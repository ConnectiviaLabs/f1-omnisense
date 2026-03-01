"""
Telemetry Metrics Extractor
────────────────────────────
Reads high-frequency telemetry from telemetry_compressed (MongoDB)
and computes per-driver performance profiles.

Collections created/updated:
  - driver_telemetry_profiles : per-driver aggregated telemetry metrics

Metrics extracted:
  - Braking: avg/max braking G, brake point consistency, brake-to-throttle transition
  - Throttle: avg throttle %, smoothness (variance), full-throttle ratio
  - Speed: avg top speed, avg race speed, straight-line pace
  - DRS: DRS usage ratio, avg speed gain in DRS zones
  - Gears: avg upshift RPM, avg downshift RPM, gear distribution
  - Driving style: wet vs dry throttle delta, late-race degradation

Usage:
    python -m pipeline.enrichment.build_telemetry_profiles
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_quality_fixes import load_telemetry_from_mongo
from updater._db import get_db


def compute_braking_metrics(drv: pd.DataFrame) -> dict:
    """Compute braking performance metrics for a single driver."""
    drv = drv.sort_values("SessionTime").reset_index(drop=True)

    dt = drv["SessionTime"].diff()
    dv_kmh = drv["Speed"].diff()
    dv_ms = dv_kmh / 3.6

    braking_mask = (drv["Brake"] > 0) & (dv_ms < 0) & (dt > 0) & (dt < 5)
    if braking_mask.sum() < 10:
        return {}

    g_values = (dv_ms[braking_mask].abs() / dt[braking_mask]) / 9.81
    g_values = g_values[(g_values > 0.1) & (g_values < 8)]

    if g_values.empty:
        return {}

    # Brake-to-throttle transition time
    # Find brake release → throttle application gaps
    brake_active = drv["Brake"] > 0
    throttle_active = drv["Throttle"] > 20
    transitions = []
    in_brake = False
    brake_end_time = None
    for i in range(len(drv)):
        if brake_active.iloc[i]:
            in_brake = True
            brake_end_time = drv["SessionTime"].iloc[i]
        elif in_brake and throttle_active.iloc[i] and brake_end_time is not None:
            gap = drv["SessionTime"].iloc[i] - brake_end_time
            if 0 < gap < 2:
                transitions.append(gap)
            in_brake = False
        elif not brake_active.iloc[i]:
            in_brake = False

    # Late-race braking delta
    drv_braking = drv.loc[braking_mask].copy()
    drv_braking["g"] = (dv_ms[braking_mask].abs() / dt[braking_mask]) / 9.81
    drv_braking = drv_braking[(drv_braking["g"] > 0.1) & (drv_braking["g"] < 8)]

    early = drv_braking[drv_braking["LapNumber"].between(1, 20)]["g"]
    late = drv_braking[drv_braking["LapNumber"] >= 40]["g"]
    late_delta = None
    if len(early) > 10 and len(late) > 10:
        late_delta = round(float(late.mean() - early.mean()), 4)

    return {
        "avg_braking_g": round(float(g_values.mean()), 4),
        "max_braking_g": round(float(g_values.quantile(0.99)), 4),
        "braking_consistency": round(float(g_values.std()), 4),
        "brake_to_throttle_avg_s": round(float(np.mean(transitions)), 4) if transitions else None,
        "late_race_braking_delta": late_delta,
    }


def compute_throttle_metrics(drv: pd.DataFrame) -> dict:
    """Compute throttle application metrics."""
    throttle = drv["Throttle"].dropna()
    if len(throttle) < 100:
        return {}

    full_throttle_mask = throttle >= 95
    full_throttle_ratio = full_throttle_mask.sum() / len(throttle)

    # Throttle smoothness: std of throttle changes (lower = smoother)
    throttle_diffs = throttle.diff().dropna()
    smoothness = float(throttle_diffs.std())

    # Late race throttle comparison
    early_throttle = drv.loc[drv["LapNumber"].between(1, 20), "Throttle"].dropna()
    late_throttle = drv.loc[drv["LapNumber"] >= 40, "Throttle"].dropna()
    late_delta = None
    if len(early_throttle) > 100 and len(late_throttle) > 100:
        late_delta = round(float(late_throttle.mean() - early_throttle.mean()), 4)

    return {
        "avg_throttle_pct": round(float(throttle.mean()), 2),
        "full_throttle_ratio": round(float(full_throttle_ratio), 4),
        "throttle_smoothness": round(smoothness, 4),
        "late_race_throttle_delta": late_delta,
    }


def compute_speed_metrics(drv: pd.DataFrame) -> dict:
    """Compute speed-related metrics."""
    speed = drv["Speed"].dropna()
    if len(speed) < 100:
        return {}

    # Top speed: 99th percentile (avoids sensor spikes)
    top_speed = float(speed.quantile(0.99))

    # Average race speed
    avg_speed = float(speed.mean())

    # Late race speed drop
    early_speed = drv.loc[drv["LapNumber"].between(1, 20), "Speed"].dropna()
    late_speed = drv.loc[drv["LapNumber"] >= 40, "Speed"].dropna()
    late_drop = None
    if len(early_speed) > 100 and len(late_speed) > 100:
        late_drop = round(float(late_speed.mean() - early_speed.mean()), 2)

    return {
        "top_speed_kmh": round(top_speed, 1),
        "avg_race_speed_kmh": round(avg_speed, 1),
        "late_race_speed_drop_kmh": late_drop,
    }


def compute_drs_metrics(drv: pd.DataFrame) -> dict:
    """Compute DRS usage metrics."""
    if "DRS" not in drv.columns:
        return {}

    drs = drv["DRS"].dropna()
    if len(drs) < 100:
        return {}

    # DRS values: 0-1 = off, 10-14 = enabled/active (varies by source)
    drs_active = (drs >= 10).sum()
    drs_ratio = drs_active / len(drs)

    # Speed with DRS vs without
    drs_on = drv.loc[drv["DRS"] >= 10, "Speed"].dropna()
    drs_off = drv.loc[(drv["DRS"] < 10) & (drv["Speed"] > 200), "Speed"].dropna()
    drs_speed_gain = None
    if len(drs_on) > 50 and len(drs_off) > 50:
        drs_speed_gain = round(float(drs_on.mean() - drs_off.mean()), 1)

    return {
        "drs_usage_ratio": round(float(drs_ratio), 4),
        "drs_speed_gain_kmh": drs_speed_gain,
    }


def compute_gear_metrics(drv: pd.DataFrame) -> dict:
    """Compute gear shift patterns."""
    if "nGear" not in drv.columns or "RPM" not in drv.columns:
        return {}

    gear = drv["nGear"].dropna()
    rpm = drv["RPM"].dropna()
    if len(gear) < 100:
        return {}

    # Average gear (higher = more time at speed)
    avg_gear = float(gear[gear > 0].mean())

    # Upshift RPM: RPM just before gear increases
    gear_diff = drv["nGear"].diff()
    upshift_mask = gear_diff == 1
    upshift_rpm = drv.loc[upshift_mask, "RPM"].dropna()
    avg_upshift_rpm = float(upshift_rpm.mean()) if len(upshift_rpm) > 10 else None

    # Downshift RPM: RPM just before gear decreases
    downshift_mask = gear_diff == -1
    downshift_rpm = drv.loc[downshift_mask, "RPM"].dropna()
    avg_downshift_rpm = float(downshift_rpm.mean()) if len(downshift_rpm) > 10 else None

    return {
        "avg_gear": round(avg_gear, 2),
        "avg_upshift_rpm": round(avg_upshift_rpm) if avg_upshift_rpm else None,
        "avg_downshift_rpm": round(avg_downshift_rpm) if avg_downshift_rpm else None,
    }


def compute_wet_dry_delta(drv: pd.DataFrame) -> dict:
    """Compare driving style in wet vs dry conditions using TrackStatus."""
    if "TrackStatus" not in drv.columns:
        return {}

    # TrackStatus: "1" = green, "2" = yellow, "3" = SC, "4" = VSC, "5" = red, "6" = wet
    wet_mask = drv["TrackStatus"].astype(str).str.contains("6", na=False)
    dry_mask = drv["TrackStatus"].astype(str).isin(["1", ""])

    wet = drv.loc[wet_mask]
    dry = drv.loc[dry_mask]

    if len(wet) < 100 or len(dry) < 100:
        return {}

    wet_throttle = wet["Throttle"].mean()
    dry_throttle = dry["Throttle"].mean()
    wet_brake_ratio = (wet["Brake"] > 0).sum() / len(wet)
    dry_brake_ratio = (dry["Brake"] > 0).sum() / len(dry)

    return {
        "wet_throttle_delta": round(float(wet_throttle - dry_throttle), 2),
        "wet_brake_ratio_delta": round(float(wet_brake_ratio - dry_brake_ratio), 4),
    }


def main():
    db = get_db()
    print("✅ Connected to MongoDB")
    print("\nLoading race telemetry from telemetry_compressed...")

    tel_df = load_telemetry_from_mongo(
        db, session_filter="R",
        columns=[
            "Driver", "Speed", "Brake", "Throttle", "DRS", "nGear", "RPM",
            "SessionTime", "LapNumber", "TrackStatus",
        ],
    )

    if tel_df.empty:
        print("  ⚠ No telemetry data found")
        return

    print(f"  Loaded {len(tel_df):,} rows, {tel_df['Driver'].nunique()} drivers")

    # Build driver number → code mapping
    num_to_code = {}
    for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0}):
        num_to_code[str(doc["driver_number"])] = doc["name_acronym"]

    tel_df["Driver"] = tel_df["Driver"].map(num_to_code)
    tel_df = tel_df.dropna(subset=["Driver"])
    print(f"  After mapping: {tel_df['Driver'].nunique()} drivers with known codes")

    drivers = sorted(tel_df["Driver"].unique())
    ops = []
    now = datetime.now(timezone.utc)

    for i, driver_code in enumerate(drivers):
        drv = tel_df[tel_df["Driver"] == driver_code].copy()
        if len(drv) < 1000:
            continue

        metrics = {"driver_code": driver_code, "sample_count": len(drv)}

        # Compute all metric groups
        metrics.update(compute_braking_metrics(drv))
        metrics.update(compute_throttle_metrics(drv))
        metrics.update(compute_speed_metrics(drv))
        metrics.update(compute_drs_metrics(drv))
        metrics.update(compute_gear_metrics(drv))
        metrics.update(compute_wet_dry_delta(drv))

        metrics["updated_at"] = now

        ops.append(UpdateOne(
            {"driver_code": driver_code},
            {"$set": metrics},
            upsert=True,
        ))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(drivers)} drivers...")

    if ops:
        db["driver_telemetry_profiles"].create_index("driver_code", unique=True)
        result = db["driver_telemetry_profiles"].bulk_write(ops, ordered=False)
        count = result.upserted_count + result.modified_count
        print(f"\n  ✅ Upserted {count} driver telemetry profiles")
    else:
        print("\n  ⚠ No profiles to write")

    # Verify
    total = db["driver_telemetry_profiles"].count_documents({})
    print(f"\n  driver_telemetry_profiles: {total} drivers")

    sample = db["driver_telemetry_profiles"].find_one(
        {"driver_code": "VER"},
        {"_id": 0, "updated_at": 0},
    )
    if sample:
        print(f"\n  Sample (VER):")
        for k, v in sorted(sample.items()):
            print(f"    {k:35s}  {v}")


if __name__ == "__main__":
    main()
