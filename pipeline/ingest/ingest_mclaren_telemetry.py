"""
Ingest McLaren telemetry CSVs into MongoDB `mclaren_telemetry` collection.

Reads per-race CSVs from data/Car/mclaren_data/, computes rich per-race
features (throttle consistency, brake patterns, gear entropy, tyre-aware
degradation, late-race drops), and stores one document per (season, race, driver).

Usage:
    PYTHONPATH=pipeline:. python3 -m pipeline.ingest.ingest_mclaren_telemetry
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient

CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Car", "mclaren_data")
COLLECTION = "mclaren_telemetry"


def load_race_csv(filepath: str) -> pd.DataFrame:
    """Load a single race CSV with proper type coercion."""
    df = pd.read_csv(filepath)
    for col in ["RPM", "Speed", "Throttle", "TyreLife", "LapNumber"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Brake"] = df["Brake"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0).astype(int)
    df["DRS"] = pd.to_numeric(df["DRS"], errors="coerce").fillna(0).astype(int)
    df["nGear"] = pd.to_numeric(df["nGear"], errors="coerce").fillna(0).astype(int)
    return df


def compute_rich_features(df: pd.DataFrame) -> dict:
    """Compute per-race rich telemetry features for a single driver."""
    features = {}
    n_laps = df["LapNumber"].nunique()

    # --- Throttle ---
    features["throttle_mean"] = float(df["Throttle"].mean())
    features["throttle_std"] = float(df["Throttle"].std())
    features["full_throttle_ratio"] = float((df["Throttle"] > 95).mean())
    features["partial_throttle_ratio"] = float(((df["Throttle"] > 20) & (df["Throttle"] < 80)).mean())

    lap_throttle_std = df.groupby("LapNumber")["Throttle"].std()
    features["throttle_intra_lap_std_mean"] = float(lap_throttle_std.mean())
    features["throttle_intra_lap_std_std"] = float(lap_throttle_std.std())

    # --- Brake ---
    features["brake_ratio"] = float(df["Brake"].mean())
    brake_transitions = float(df["Brake"].diff().abs().sum())
    features["brake_transitions_per_lap"] = brake_transitions / max(n_laps, 1)
    features["brake_throttle_overlap"] = float(((df["Brake"] == 1) & (df["Throttle"] > 20)).mean())

    # --- RPM ---
    features["rpm_mean"] = float(df["RPM"].mean())
    features["rpm_std"] = float(df["RPM"].std())
    features["rpm_max"] = float(df["RPM"].max())
    features["rpm_range"] = float(df["RPM"].max() - df["RPM"].min())
    features["rpm_cv"] = float(df["RPM"].std() / max(df["RPM"].mean(), 1))

    # --- Speed ---
    features["speed_mean"] = float(df["Speed"].mean())
    features["speed_std"] = float(df["Speed"].std())
    features["speed_max"] = float(df["Speed"].max())

    lap_speed_mean = df.groupby("LapNumber")["Speed"].mean()
    features["speed_lap_to_lap_std"] = float(lap_speed_mean.std())

    # --- Gear ---
    features["gear_mean"] = float(df["nGear"].mean())
    features["gear_std"] = float(df["nGear"].std())
    gear_shifts = float((df["nGear"].diff().abs() > 0).sum())
    features["gear_shifts_per_lap"] = gear_shifts / max(n_laps, 1)
    features["top_gear_ratio"] = float((df["nGear"] >= 7).mean())

    gear_counts = df["nGear"].value_counts(normalize=True)
    features["gear_entropy"] = float(-(gear_counts * np.log2(gear_counts + 1e-10)).sum())

    # --- DRS ---
    features["drs_active_ratio"] = float((df["DRS"] >= 10).mean())
    features["drs_available_ratio"] = float((df["DRS"] >= 8).mean())

    # --- Tyre-aware degradation ---
    if "TyreLife" in df.columns and df["TyreLife"].nunique() > 3:
        fresh = df[df["TyreLife"] <= 10]
        worn = df[df["TyreLife"] > 20]
        if len(fresh) > 100 and len(worn) > 100:
            features["speed_degradation"] = float(fresh["Speed"].mean() - worn["Speed"].mean())
            features["throttle_degradation"] = float(fresh["Throttle"].mean() - worn["Throttle"].mean())
            features["rpm_degradation"] = float(fresh["RPM"].mean() - worn["RPM"].mean())
            features["brake_increase_with_wear"] = float(worn["Brake"].mean() - fresh["Brake"].mean())
        else:
            features["speed_degradation"] = 0.0
            features["throttle_degradation"] = 0.0
            features["rpm_degradation"] = 0.0
            features["brake_increase_with_wear"] = 0.0

        valid = df.dropna(subset=["TyreLife", "Speed"])
        features["tyre_speed_correlation"] = float(valid["TyreLife"].corr(valid["Speed"])) if len(valid) > 50 else 0.0
    else:
        features["speed_degradation"] = 0.0
        features["throttle_degradation"] = 0.0
        features["rpm_degradation"] = 0.0
        features["brake_increase_with_wear"] = 0.0
        features["tyre_speed_correlation"] = 0.0

    # --- Lap-to-lap stability ---
    lap_features = df.groupby("LapNumber").agg({"Speed": "mean", "RPM": "mean", "Throttle": "mean"})
    if len(lap_features) > 3:
        for col in ["Speed", "RPM", "Throttle"]:
            rolling_std = lap_features[col].rolling(3, min_periods=2).std()
            features[f"{col.lower()}_rolling_instability"] = float(rolling_std.mean())

        n = len(lap_features)
        q1 = lap_features.iloc[:n // 4]
        q4 = lap_features.iloc[-(n // 4):]
        if len(q1) > 0 and len(q4) > 0:
            features["late_race_speed_drop"] = float(q1["Speed"].mean() - q4["Speed"].mean())
            features["late_race_rpm_drop"] = float(q1["RPM"].mean() - q4["RPM"].mean())
        else:
            features["late_race_speed_drop"] = 0.0
            features["late_race_rpm_drop"] = 0.0
    else:
        features["speed_rolling_instability"] = 0.0
        features["rpm_rolling_instability"] = 0.0
        features["throttle_rolling_instability"] = 0.0
        features["late_race_speed_drop"] = 0.0
        features["late_race_rpm_drop"] = 0.0

    # Replace NaN with 0
    for k, v in features.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0

    return features


def main():
    client = MongoClient(os.environ.get("MONGODB_URI", "mongodb://localhost:27017"))
    db = client[os.environ.get("MONGODB_DB", "marip_f1")]
    coll = db[COLLECTION]

    csv_files = sorted([
        f for f in os.listdir(CSV_DIR)
        if f.endswith(".csv") and "ALL_COMBINED" not in f
        and (f.startswith("2023") or f.startswith("2024"))
    ])

    print(f"Found {len(csv_files)} McLaren race CSVs")
    total_ingested = 0

    for csv_file in csv_files:
        season = int(csv_file[:4])
        race_name = csv_file[5:].replace("_Race.csv", "").replace("_", " ")
        filepath = os.path.join(CSV_DIR, csv_file)

        df = load_race_csv(filepath)
        drivers = df["Driver"].unique()
        samples = len(df)

        for driver in drivers:
            driver_df = df[df["Driver"] == driver].copy()
            if len(driver_df) < 100:
                continue

            features = compute_rich_features(driver_df)

            # Compounds used
            compounds = []
            if "Compound" in driver_df.columns:
                compounds = driver_df["Compound"].dropna().unique().tolist()

            doc = {
                "season": season,
                "race_name": race_name,
                "driver_code": driver,
                "constructor_id": "mclaren",
                "features": features,
                "compounds_used": compounds,
                "total_samples": len(driver_df),
                "total_laps": int(driver_df["LapNumber"].nunique()),
                "source": "mclaren_csv",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }

            # Upsert by (season, race_name, driver_code)
            coll.update_one(
                {"season": season, "race_name": race_name, "driver_code": driver},
                {"$set": doc},
                upsert=True,
            )
            total_ingested += 1

        print(f"  {csv_file}: {len(drivers)} drivers, {samples:,} samples")

    # Create index
    coll.create_index([("season", 1), ("race_name", 1), ("driver_code", 1)], unique=True)
    coll.create_index([("season", 1), ("driver_code", 1)])

    print(f"\nIngested {total_ingested} documents into {COLLECTION}")
    print(f"Collection count: {coll.count_documents({})}")


if __name__ == "__main__":
    main()
