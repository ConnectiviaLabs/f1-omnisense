"""
Driver Performance Markers Pipeline
────────────────────────────────────
Reads from fastf1_laps, fastf1_weather, and local telemetry parquets.
Outputs → driver_performance_markers collection in MongoDB (marip_f1).

Usage:
    python pipeline/driver_performance_markers.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from updater._db import get_db
from data_quality_fixes import load_telemetry_from_mongo

warnings.filterwarnings("ignore")

OUT_COL = "driver_performance_markers"


# ── Marker Computation Functions ──────────────────────────────────────────────

def throttle_smoothness(series: pd.Series) -> float:
    diffs = series.diff().abs().dropna()
    return round(float(diffs.mean()), 4) if len(diffs) > 0 else np.nan


def brake_overlap_rate(throttle: pd.Series, brake: pd.Series) -> float:
    overlap = ((throttle > 10) & (brake == True)).sum()
    total = len(throttle.dropna())
    return round(float(overlap / total), 4) if total > 0 else np.nan


def lap_time_std(lap_times: pd.Series) -> float:
    clean = lap_times.dropna()
    return round(float(clean.std()), 4) if len(clean) > 1 else np.nan


def sector_consistency(sector_series: pd.Series) -> float:
    clean = sector_series.dropna()
    if clean.mean() == 0 or len(clean) < 2:
        return np.nan
    return round(float(clean.std() / clean.mean()), 4)


def degradation_slope(lap_numbers: pd.Series, lap_times: pd.Series) -> float:
    df = pd.DataFrame({"lap": lap_numbers, "time": lap_times}).dropna()
    if len(df) < 4:
        return np.nan
    slope = np.polyfit(df["lap"].values, df["time"].values, 1)[0]
    return round(float(slope), 5)


def late_race_delta(lap_times: pd.Series, lap_numbers: pd.Series) -> float:
    df = pd.DataFrame({"lap": lap_numbers, "time": lap_times}).dropna()
    if len(df) < 8:
        return np.nan
    cutoff_early = df["lap"].quantile(0.25)
    cutoff_late = df["lap"].quantile(0.75)
    early = df[df["lap"] <= cutoff_early]["time"].mean()
    late = df[df["lap"] >= cutoff_late]["time"].mean()
    return round(float(late - early), 4)


def late_race_speed_drop(speeds: pd.Series, lap_numbers: pd.Series) -> float:
    df = pd.DataFrame({"lap": lap_numbers, "speed": speeds}).dropna()
    if len(df) < 8:
        return np.nan
    per_lap = df.groupby("lap")["speed"].max()
    cutoff_early = per_lap.index.to_series().quantile(0.25)
    cutoff_late = per_lap.index.to_series().quantile(0.75)
    early = per_lap[per_lap.index <= cutoff_early].mean()
    late = per_lap[per_lap.index >= cutoff_late].mean()
    return round(float(late - early), 3)


def avg_top_speed(speeds: pd.Series) -> float:
    return round(float(speeds.quantile(0.99)), 2) if len(speeds) > 0 else np.nan


def avg_throttle_pct(throttle: pd.Series) -> float:
    return round(float(throttle.mean()), 2) if len(throttle) > 0 else np.nan


def heat_lap_delta(race_laps: pd.DataFrame, weather_df: pd.DataFrame) -> float:
    if weather_df is None or weather_df.empty:
        return np.nan
    hot = weather_df[weather_df["TrackTemp"] > 45]
    if hot.empty:
        return np.nan
    hot_races = hot["Race"].unique()
    hot_laps = race_laps[race_laps["Race"].isin(hot_races)]["LapTime"]
    all_laps = race_laps["LapTime"]
    if hot_laps.dropna().empty or all_laps.dropna().empty:
        return np.nan
    return round(float(hot_laps.median() - all_laps.median()), 4)


def humidity_lap_delta(race_laps: pd.DataFrame, weather_df: pd.DataFrame) -> float:
    if weather_df is None or weather_df.empty:
        return np.nan
    humid = weather_df[weather_df["Humidity"] > 70]
    if humid.empty:
        return np.nan
    humid_races = humid["Race"].unique()
    humid_laps = race_laps[race_laps["Race"].isin(humid_races)]["LapTime"]
    all_laps = race_laps["LapTime"]
    if humid_laps.dropna().empty or all_laps.dropna().empty:
        return np.nan
    return round(float(humid_laps.median() - all_laps.median()), 4)


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe(val):
    """Convert numpy types and NaN to Python-safe values for MongoDB."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def build_driver_number_map(db) -> dict[str, str]:
    """Map driver number (str) → three-letter code using openf1_drivers."""
    mapping = {}
    for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0}):
        mapping[str(doc["driver_number"])] = doc["name_acronym"]
    return mapping


def load_all_telemetry(db, driver_number_map: dict[str, str]) -> pd.DataFrame:
    """Load race telemetry from telemetry_compressed, remap driver numbers → codes."""
    tel_df = load_telemetry_from_mongo(
        db, session_filter="R",
        columns=["Driver", "Speed", "Throttle", "Brake", "LapNumber", "Session"],
    )
    if tel_df.empty:
        print("  ⚠ No telemetry found in telemetry_compressed")
        return pd.DataFrame()

    tel_df["Driver"] = tel_df["Driver"].astype(str).map(driver_number_map)
    tel_df = tel_df.dropna(subset=["Driver"])
    print(f"  Loaded {len(tel_df):,} telemetry rows from telemetry_compressed")
    return tel_df


def process_driver(
    driver_code: str,
    laps_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    tel_df: pd.DataFrame,
) -> dict | None:
    """Compute all markers for one driver."""

    drv_laps = laps_df[laps_df["Driver"] == driver_code].copy()
    if drv_laps.empty:
        print(f"    ⚠ No lap data for {driver_code}")
        return None

    race_laps = drv_laps[drv_laps["LapTime"].notna() & (drv_laps["LapTime"] > 60)]
    if race_laps.empty:
        print(f"    ⚠ No valid race laps for {driver_code}")
        return None

    print(f"  {driver_code}: {len(race_laps)} race laps", end="")

    markers = {
        "Driver": driver_code,
        "years_covered": sorted(int(y) for y in drv_laps["Year"].dropna().unique()),
        "total_race_laps": int(len(race_laps)),

        # Pace degradation
        "degradation_slope_s_per_lap": safe(degradation_slope(
            race_laps["LapNumber"], race_laps["LapTime"]
        )),
        "late_race_delta_s": safe(late_race_delta(
            race_laps["LapTime"], race_laps["LapNumber"]
        )),
        "lap_time_consistency_std": safe(lap_time_std(race_laps["LapTime"])),

        # Sector consistency
        "sector1_cv": safe(sector_consistency(drv_laps.get("Sector1Time", pd.Series(dtype=float)))),
        "sector2_cv": safe(sector_consistency(drv_laps.get("Sector2Time", pd.Series(dtype=float)))),
        "sector3_cv": safe(sector_consistency(drv_laps.get("Sector3Time", pd.Series(dtype=float)))),

        # Weather sensitivity
        "heat_lap_delta_s": safe(heat_lap_delta(race_laps, weather_df)),
        "humidity_lap_delta_s": safe(humidity_lap_delta(race_laps, weather_df)),

        # Telemetry markers (filled below if available)
        "throttle_smoothness": None,
        "brake_overlap_rate": None,
        "avg_top_speed_kmh": None,
        "avg_throttle_pct": None,
        "late_race_speed_drop_kmh": None,
    }

    # Telemetry markers
    if not tel_df.empty:
        drv_tel = tel_df[(tel_df["Driver"] == driver_code) & (tel_df["Session"] == "R")]
        if not drv_tel.empty:
            print(f", {len(drv_tel):,} tel rows", end="")
            markers["throttle_smoothness"] = safe(throttle_smoothness(drv_tel["Throttle"]))
            markers["brake_overlap_rate"] = safe(brake_overlap_rate(drv_tel["Throttle"], drv_tel["Brake"]))
            markers["avg_top_speed_kmh"] = safe(avg_top_speed(drv_tel["Speed"]))
            markers["avg_throttle_pct"] = safe(avg_throttle_pct(drv_tel["Throttle"]))
            markers["late_race_speed_drop_kmh"] = safe(late_race_speed_drop(
                drv_tel["Speed"], drv_tel["LapNumber"]
            ))

    # Tyre stint endurance
    if "TyreLife" in drv_laps.columns and "Stint" in drv_laps.columns:
        long_stints = drv_laps[drv_laps["TyreLife"] > 20]
        if not long_stints.empty:
            markers["long_stint_lap_delta"] = safe(
                float(long_stints["LapTime"].mean() - race_laps["LapTime"].mean())
            )
        else:
            markers["long_stint_lap_delta"] = None

        stint_lengths = drv_laps.groupby(["Year", "Race", "Stint"])["LapNumber"].count()
        markers["avg_stint_length"] = safe(float(stint_lengths.mean())) if len(stint_lengths) > 0 else None
    else:
        markers["long_stint_lap_delta"] = None
        markers["avg_stint_length"] = None

    print()
    return markers


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    db = get_db()
    print("✅ Connected to MongoDB\n")

    # Get driver list
    drivers = sorted(
        d["driver_code"]
        for d in db["opponent_profiles"].find({}, {"driver_code": 1, "_id": 0})
        if d.get("driver_code")
    )
    print(f"Drivers to process: {len(drivers)}")

    # Load all fastf1_laps for race sessions (once, not per-driver)
    print("\nLoading fastf1_laps (race sessions)...")
    laps_raw = list(db["fastf1_laps"].find(
        {"SessionType": "R"},
        {
            "Driver": 1, "LapTime": 1, "LapNumber": 1, "Race": 1, "Year": 1,
            "Compound": 1, "TyreLife": 1, "Stint": 1,
            "Sector1Time": 1, "Sector2Time": 1, "Sector3Time": 1, "_id": 0,
        },
    ))
    laps_df = pd.DataFrame(laps_raw)
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "LapNumber", "TyreLife"]:
        if col in laps_df.columns:
            laps_df[col] = pd.to_numeric(laps_df[col], errors="coerce")
    print(f"  {len(laps_df):,} lap records loaded")

    # Load weather (once)
    print("Loading fastf1_weather (race sessions)...")
    weather_raw = list(db["fastf1_weather"].find(
        {"SessionType": "R"},
        {"Race": 1, "Year": 1, "TrackTemp": 1, "Humidity": 1, "_id": 0},
    ))
    weather_df = pd.DataFrame(weather_raw) if weather_raw else None
    print(f"  {len(weather_df) if weather_df is not None else 0:,} weather records loaded")

    # Load telemetry from MongoDB (telemetry_compressed)
    print("Loading telemetry from MongoDB...")
    driver_number_map = build_driver_number_map(db)
    tel_df = load_all_telemetry(db, driver_number_map)

    # Process each driver
    print(f"\n{'=' * 60}")
    print("COMPUTING PERFORMANCE MARKERS")
    print(f"{'=' * 60}")

    results = []
    skipped = []
    for driver in drivers:
        try:
            marker = process_driver(driver, laps_df, weather_df, tel_df)
            if marker:
                results.append(marker)
        except Exception as e:
            print(f"  ❌ {driver}: {e}")
            skipped.append(driver)

    print(f"\n✅ Computed markers for {len(results)} drivers, skipped {len(skipped)}")
    if skipped:
        print(f"   Skipped: {skipped}")

    # Upsert to MongoDB
    if results:
        db[OUT_COL].create_index("Driver", unique=True)
        ops = [
            UpdateOne({"Driver": r["Driver"]}, {"$set": r}, upsert=True)
            for r in results
        ]
        result = db[OUT_COL].bulk_write(ops, ordered=False)
        print(f"✅ Upserted {result.upserted_count + result.modified_count} documents into {OUT_COL}")
    else:
        print("⚠ No results to write")

    # Sanity check
    total = db[OUT_COL].count_documents({})
    print(f"\nTotal docs in {OUT_COL}: {total}")
    sample = db[OUT_COL].find_one({"Driver": "NOR"})
    if sample:
        sample.pop("_id", None)
        print(f"\nSample (NOR):")
        for k, v in sample.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
