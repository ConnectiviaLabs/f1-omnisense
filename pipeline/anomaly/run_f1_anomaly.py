"""
F1 Anomaly Scoring Pipeline — All Drivers.

Loads race telemetry from MongoDB (fastf1_laps), runs the ensemble
anomaly detection, and outputs per-driver per-race per-system anomaly
scores as JSON for the Fleet Overview UI.

Usage:
    python -m pipeline.anomaly.run_f1_anomaly                # all grid drivers
    python -m pipeline.anomaly.run_f1_anomaly --team McLaren # specific team
    python -m pipeline.anomaly.run_f1_anomaly --driver VER   # single driver

Output:
    pipeline/output/anomaly_scores.json
"""

import argparse
import json
import logging
import os
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
from pipeline.anomaly.mongo_loader import (
    load_driver_race_telemetry,
    get_grid_drivers,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]  # f1/
OUTPUT = ROOT / "pipeline" / "output" / "anomaly_scores.json"

# System groupings — map fastf1_laps aggregated columns to vehicle systems
SYSTEM_FEATURES = {
    "Speed": ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"],
    "Lap Pace": ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"],
    "Tyre Management": ["TyreLife"],
}


def load_car_race_data(driver_code: str) -> pd.DataFrame:
    """Load per-race telemetry for any driver from MongoDB."""
    return load_driver_race_telemetry(driver_code)


def run_ensemble_per_system(merged_df: pd.DataFrame) -> tuple:
    """Run the anomaly ensemble on per-system feature groups."""
    if merged_df.empty or len(merged_df) < 3:
        logger.warning("Not enough data for ensemble (need >= 3 races)")
        return {}, {}

    ensemble = AnomalyDetectionEnsemble()
    stats = AnomalyStatistics()

    system_col_map = {}
    for system, raw_features in SYSTEM_FEATURES.items():
        cols = []
        for feat in raw_features:
            matching = [c for c in merged_df.columns
                        if c.startswith(feat) and c not in ("driver", "race", "Year", "Race")]
            cols.extend(matching)
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

            health = max(10, min(100, int(100 - score_mean * 80)))

            vote_count = sum(1 for col in result_df.columns
                           if col.endswith("_Anomaly") and row.get(col, 0) == 1)
            total_models = sum(1 for col in result_df.columns if col.endswith("_Anomaly"))

            vote_severity = severity_from_votes(vote_count, total_models)

            score_cols = [c for c in result_df.columns if c.endswith("_AnomalyScore")]
            top_model = ""
            if score_cols:
                max_col = max(score_cols, key=lambda c: row.get(c, 0))
                top_model = max_col.replace("_AnomalyScore", "")

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

            if "classifier_severity" in result_df.columns:
                ClassifierPipeline.enrich_health_entry(entry, row)

            race_data["systems"][system] = entry

        per_race.append(race_data)

    return per_race


def run_driver(driver_code: str, driver_info: dict | None = None) -> dict:
    """Full pipeline for a single driver."""
    name = driver_info.get("name", driver_code) if driver_info else driver_code
    number = driver_info.get("number", 0) if driver_info else 0
    team = driver_info.get("team", "Unknown") if driver_info else "Unknown"

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {name} ({driver_code}) — {team}")
    logger.info(f"{'='*60}")

    merged = load_car_race_data(driver_code)

    if merged.empty:
        logger.warning(f"No data for {driver_code}")
        return {}

    logger.info(f"Loaded {len(merged)} races, {merged.shape[1]} features")

    system_results, system_col_map = run_ensemble_per_system(merged)

    if not system_results:
        logger.warning(f"No system results for {driver_code}")
        return {}

    logger.info("Running severity classifier...")
    system_results = run_classifier_per_system(
        merged, system_results, system_col_map, driver_code,
    )

    per_race_health = compute_system_health(system_results, merged)

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
        "driver": name,
        "number": number,
        "code": driver_code,
        "team": team,
        "overall_health": overall_health,
        "overall_level": overall_level,
        "last_race": per_race_health[-1]["race"] if per_race_health else "",
        "races": per_race_health,
        "race_count": len(per_race_health),
    }


def _push_to_mongo(output_data: dict):
    """Push anomaly scores to MongoDB."""
    try:
        from updater._db import get_db
        db = get_db()
        db["anomaly_scores_snapshot"].drop()
        db["anomaly_scores_snapshot"].insert_one(output_data)
        logger.info("Pushed to MongoDB marip_f1.anomaly_scores_snapshot")
    except Exception as e:
        logger.warning(f"MongoDB push failed: {e}")


def main():
    """Run anomaly pipeline for all drivers (or filtered by team/driver)."""
    parser = argparse.ArgumentParser(description="F1 Anomaly Scoring Pipeline")
    parser.add_argument("--driver", help="Single driver code (e.g. VER)")
    parser.add_argument("--team", help="Filter by team (e.g. McLaren)")
    parser.add_argument("--year", type=int, default=2024, help="Grid year (default 2024)")
    args = parser.parse_args()

    logger.info("F1 Anomaly Scoring Pipeline — All Drivers")

    grid = get_grid_drivers(args.year)
    logger.info(f"Grid: {len(grid)} drivers for {args.year}")

    if args.driver:
        grid = [d for d in grid if d["code"] == args.driver.upper()]
    elif args.team:
        grid = [d for d in grid if args.team.lower() in d["team"].lower()]

    if not grid:
        logger.error("No drivers matched filters")
        return

    logger.info(f"Processing {len(grid)} driver(s): {[d['code'] for d in grid]}")

    output_data = {"drivers": [], "metadata": {
        "systems": list(SYSTEM_FEATURES.keys()),
        "models": ["IsolationForest", "OneClassSVM", "KNN", "PCA_Reconstruction"],
        "model_weights": {"IsolationForest": 1.0, "OneClassSVM": 0.6, "KNN": 0.8, "PCA_Reconstruction": 0.9},
        "year": args.year,
    }}

    for driver_info in grid:
        result = run_driver(driver_info["code"], driver_info)
        if result:
            output_data["drivers"].append(result)

    # Write local JSON
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nOutput written to {OUTPUT}")
    logger.info(f"Drivers: {len(output_data['drivers'])}")
    for d in output_data["drivers"]:
        logger.info(f"  {d['code']:5s} {d.get('team',''):20s} {d['overall_health']}% ({d['overall_level']}) — {d['race_count']} races")

    _push_to_mongo(output_data)


if __name__ == "__main__":
    main()
