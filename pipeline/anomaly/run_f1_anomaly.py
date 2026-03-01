"""
F1 Anomaly Scoring Pipeline.

Loads McCar telemetry + McDriver biometric CSVs, runs the ensemble
anomaly detection from OmniSense DataSense, and outputs per-driver
per-race per-system anomaly scores as JSON for the Fleet Overview UI.

Usage:
    python -m pipeline.anomaly.run_f1_anomaly

Output:
    pipeline/output/anomaly_scores.json
"""

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.anomaly.ensemble import (
    AnomalyDetectionEnsemble,
    AnomalyStatistics,
    severity_from_votes,
)
from pipeline.anomaly.classifier import ClassifierPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]  # f1/
MCCAR_DIR = ROOT / "f1data" / "McCar" / "2024"
MCDRIVER_DIR = ROOT / "f1data" / "McDriver" / "2024"
OUTPUT = ROOT / "pipeline" / "output" / "anomaly_scores.json"

DRIVERS = {"NOR": {"name": "Lando Norris", "number": 4},
           "PIA": {"name": "Oscar Piastri", "number": 81}}

# System groupings — map telemetry columns to vehicle systems
SYSTEM_FEATURES = {
    "Power Unit":   ["RPM", "nGear"],
    "Brakes":       ["Brake", "Speed"],  # brake application + deceleration
    "Drivetrain":   ["Throttle", "DRS"],
    "Suspension":   ["Speed", "Distance"],
    "Thermal":      ["HeartRate_bpm", "CockpitTemp_C", "AirTemp_C", "TrackTemp_C"],
    "Electronics":  ["DRS", "RPM", "nGear"],
}


def load_car_race_data(driver_code: str) -> pd.DataFrame:
    """Load and aggregate per-race car telemetry for a driver."""
    if not MCCAR_DIR.exists():
        logger.warning(f"McCar directory not found: {MCCAR_DIR}")
        return pd.DataFrame()

    rows = []
    for csv_file in sorted(MCCAR_DIR.glob("2024_*_Race.csv")):
        race_match = re.match(r"2024_(.+)_Grand_Prix_Race\.csv", csv_file.name)
        if not race_match:
            continue
        race_name = race_match.group(1).replace("_", " ")

        try:
            df = pd.read_csv(csv_file, low_memory=False)
            df = df[df["Driver"] == driver_code]
            if df.empty:
                continue

            # Aggregate per-race
            row = {"race": race_name, "driver": driver_code}
            for col in ["RPM", "Speed", "Throttle", "nGear", "Distance"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    row[f"{col}_mean"] = vals.mean() if len(vals) else 0
                    row[f"{col}_max"] = vals.max() if len(vals) else 0
                    row[f"{col}_std"] = vals.std() if len(vals) else 0
            if "Brake" in df.columns:
                brake_vals = df["Brake"].astype(str).isin(["True", "1", "true"])
                row["Brake_pct"] = brake_vals.mean() * 100
            if "DRS" in df.columns:
                drs_vals = pd.to_numeric(df["DRS"], errors="coerce").fillna(0)
                row["DRS_pct"] = (drs_vals >= 10).mean() * 100
            if "TyreLife" in df.columns:
                tl = pd.to_numeric(df["TyreLife"], errors="coerce").dropna()
                row["TyreLife_max"] = tl.max() if len(tl) else 0
            row["samples"] = len(df)
            rows.append(row)
        except Exception as e:
            logger.warning(f"Error loading {csv_file.name}: {e}")

    return pd.DataFrame(rows)


def load_bio_race_data(driver_code: str) -> pd.DataFrame:
    """Load and aggregate per-race biometric data for a driver."""
    if not MCDRIVER_DIR.exists():
        return pd.DataFrame()

    rows = []
    for csv_file in sorted(MCDRIVER_DIR.glob("*_biometrics.csv")):
        race_match = re.match(r"2024_(.+)_Grand_Prix_Race_biometrics\.csv", csv_file.name)
        if not race_match:
            continue
        race_name = race_match.group(1).replace("_", " ")

        try:
            df = pd.read_csv(csv_file, low_memory=False)
            if "Driver" in df.columns:
                df = df[df["Driver"] == driver_code]
            if df.empty:
                continue

            row = {"race": race_name, "driver": driver_code}
            for col in ["HeartRate_bpm", "CockpitTemp_C", "BattleIntensity", "AirTemp_C", "TrackTemp_C"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    row[f"{col}_mean"] = vals.mean() if len(vals) else 0
                    row[f"{col}_max"] = vals.max() if len(vals) else 0
            row["bio_samples"] = len(df)
            rows.append(row)
        except Exception as e:
            logger.warning(f"Error loading {csv_file.name}: {e}")

    return pd.DataFrame(rows)


def merge_telemetry(car_df: pd.DataFrame, bio_df: pd.DataFrame) -> pd.DataFrame:
    """Merge car + biometric data per race."""
    if car_df.empty:
        return car_df
    if bio_df.empty:
        return car_df

    merged = car_df.merge(bio_df, on=["race", "driver"], how="left")
    return merged.fillna(0)


def run_ensemble_per_system(merged_df: pd.DataFrame) -> tuple:
    """
    Run the anomaly ensemble on per-system feature groups.
    Returns (results_dict, system_col_map) for downstream classifier.
    """
    if merged_df.empty or len(merged_df) < 3:
        logger.warning("Not enough data for ensemble (need >= 3 races)")
        return {}, {}

    ensemble = AnomalyDetectionEnsemble()
    stats = AnomalyStatistics()

    # Map system features to actual column patterns in merged data
    system_col_map = {}
    for system, raw_features in SYSTEM_FEATURES.items():
        cols = []
        for feat in raw_features:
            matching = [c for c in merged_df.columns
                        if c.startswith(feat) and c != "driver" and c != "race"]
            cols.extend(matching)
        # Deduplicate preserving order
        seen = set()
        deduped = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        if deduped:
            system_col_map[system] = deduped

    results = {}
    for system, feature_cols in system_col_map.items():
        try:
            subset = merged_df[feature_cols].copy()
            subset = subset.apply(pd.to_numeric, errors="coerce").fillna(0)

            if subset.shape[1] < 2:
                logger.info(f"Skipping {system}: only {subset.shape[1]} features")
                continue

            scaler = StandardScaler()
            scaled = pd.DataFrame(
                scaler.fit_transform(subset),
                columns=subset.columns,
                index=subset.index,
            )

            _, raw_results = ensemble.run_anomaly_detection_models(subset.copy(), scaled)
            raw_results = stats.anomaly_insights(raw_results)

            results[system] = raw_results
            logger.info(f"  {system}: {len(feature_cols)} features, "
                        f"{(raw_results['Voted_Anomaly'] == 1).sum()}/{len(raw_results)} anomalies")

        except Exception as e:
            logger.error(f"  {system} failed: {e}")

    return results, system_col_map


def run_classifier_per_system(
    merged_df: pd.DataFrame,
    system_results: dict,
    system_col_map: dict,
    driver_code: str,
) -> dict:
    """Run severity classifier on each system's ensemble results."""
    pipeline = ClassifierPipeline()
    enriched = {}
    for system, result_df in system_results.items():
        feature_cols = system_col_map.get(system, [])
        try:
            enriched[system] = pipeline.train_and_predict_system(
                merged_df, system, feature_cols, result_df, driver_code,
            )
        except Exception as e:
            logger.warning(f"  Classifier failed for {system}: {e}")
            enriched[system] = result_df
    return enriched


def compute_system_health(system_results: dict, merged_df: pd.DataFrame) -> list:
    """Convert ensemble results into per-race health scores for each system."""
    races = merged_df["race"].tolist()
    per_race = []

    for i, race in enumerate(races):
        race_data = {"race": race, "systems": {}}

        for system, result_df in system_results.items():
            if i >= len(result_df):
                continue

            row = result_df.iloc[i]
            level = row.get("Anomaly_Level", "normal")
            score_mean = row.get("Anomaly_Score_Mean", 0)
            voting = row.get("Voting_Score", 0)

            # Health = 100 - (anomaly_score * 100), clamped
            health = max(10, min(100, int(100 - score_mean * 80)))

            # Count model votes
            vote_count = sum(1 for col in result_df.columns
                           if col.endswith("_Anomaly") and row.get(col, 0) == 1)
            total_models = sum(1 for col in result_df.columns if col.endswith("_Anomaly"))

            # Use vote-based severity (from broadcaster)
            vote_severity = severity_from_votes(vote_count, total_models)

            # Top deviating features
            score_cols = [c for c in result_df.columns if c.endswith("_AnomalyScore")]
            top_model = ""
            if score_cols:
                max_col = max(score_cols, key=lambda c: row.get(c, 0))
                top_model = max_col.replace("_AnomalyScore", "")

            # Get actual feature values for this system
            feature_vals = {}
            if system in SYSTEM_FEATURES:
                for feat in SYSTEM_FEATURES[system]:
                    for col in merged_df.columns:
                        if col.startswith(feat) and col.endswith("_mean"):
                            feature_vals[feat] = round(float(merged_df.iloc[i].get(col, 0)), 1)

            entry = {
                "health": health,
                "level": level,
                "vote_severity": vote_severity,
                "score_mean": round(float(score_mean), 4),
                "voting_score": round(float(voting), 3),
                "vote_count": vote_count,
                "total_models": total_models,
                "top_model": top_model,
                "features": feature_vals,
            }

            # Enrich with classifier predictions if available
            if "classifier_severity" in result_df.columns:
                ClassifierPipeline.enrich_health_entry(entry, row)

            race_data["systems"][system] = entry

        per_race.append(race_data)

    return per_race


def run_driver(driver_code: str) -> dict:
    """Full pipeline for a single driver."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {DRIVERS[driver_code]['name']} (#{DRIVERS[driver_code]['number']})")
    logger.info(f"{'='*60}")

    car_df = load_car_race_data(driver_code)
    bio_df = load_bio_race_data(driver_code)
    merged = merge_telemetry(car_df, bio_df)

    if merged.empty:
        logger.warning(f"No data for {driver_code}")
        return {}

    logger.info(f"Loaded {len(merged)} races, {merged.shape[1]} features")

    system_results, system_col_map = run_ensemble_per_system(merged)

    # Supervised classifier: train on ensemble pseudo-labels, produce
    # calibrated severity probabilities for preventative maintenance
    logger.info("Running severity classifier...")
    system_results = run_classifier_per_system(
        merged, system_results, system_col_map, driver_code,
    )

    per_race_health = compute_system_health(system_results, merged)

    # Compute overall health (latest race, average of systems)
    latest = per_race_health[-1] if per_race_health else {}
    system_healths = [s["health"] for s in latest.get("systems", {}).values()]
    overall_health = int(np.mean(system_healths)) if system_healths else 0

    levels = [s["level"] for s in latest.get("systems", {}).values()]
    if "critical" in levels:
        overall_level = "critical"
    elif "high" in levels:
        overall_level = "high"
    elif "medium" in levels:
        overall_level = "medium"
    elif "low" in levels:
        overall_level = "low"
    else:
        overall_level = "normal"

    return {
        "driver": DRIVERS[driver_code]["name"],
        "number": DRIVERS[driver_code]["number"],
        "code": driver_code,
        "overall_health": overall_health,
        "overall_level": overall_level,
        "last_race": per_race_health[-1]["race"] if per_race_health else "",
        "races": per_race_health,
        "race_count": len(per_race_health),
    }


def _push_to_mongo(output_data: dict):
    """Push anomaly scores to MongoDB for the Vercel API."""
    try:
        from dotenv import load_dotenv
        from pymongo import MongoClient

        load_dotenv(ROOT / ".env")
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            logger.warning("MONGODB_URI not set — skipping MongoDB push")
            return

        db_name = os.environ.get("MONGODB_DB", "marip_f1")
        client = MongoClient(uri)
        db = client[db_name]

        db["anomaly_scores_snapshot"].drop()
        db["anomaly_scores_snapshot"].insert_one(output_data)

        logger.info(f"Pushed to MongoDB {db_name}.anomaly_scores_snapshot")
        client.close()
    except ImportError:
        logger.warning("pymongo/dotenv not installed — skipping MongoDB push")
    except Exception as e:
        logger.warning(f"MongoDB push failed: {e}")


def main():
    """Run anomaly pipeline for all McLaren drivers."""
    logger.info("F1 Anomaly Scoring Pipeline")
    logger.info(f"McCar dir: {MCCAR_DIR}")
    logger.info(f"McDriver dir: {MCDRIVER_DIR}")

    output_data = {"drivers": [], "metadata": {
        "systems": list(SYSTEM_FEATURES.keys()),
        "models": ["IsolationForest", "OneClassSVM", "KNN", "PCA_Reconstruction"],
        "model_weights": {"IsolationForest": 1.0, "OneClassSVM": 0.6, "KNN": 0.8, "PCA_Reconstruction": 0.9},
    }}

    for code in DRIVERS:
        result = run_driver(code)
        if result:
            output_data["drivers"].append(result)

    # Write local JSON
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nOutput written to {OUTPUT}")
    logger.info(f"Drivers: {len(output_data['drivers'])}")
    for d in output_data["drivers"]:
        logger.info(f"  {d['driver']}: {d['overall_health']}% ({d['overall_level']}) — {d['race_count']} races")

    # Push to MongoDB so Vercel API serves fresh data
    _push_to_mongo(output_data)


if __name__ == "__main__":
    main()
