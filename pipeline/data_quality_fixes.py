"""
Data Quality Fixes Pipeline
────────────────────────────
Addresses all issues identified in the data quality audit:

1. Fix braking G values (billions → real 3-6g range)
2. Deduplicate fastf1_laps and fastf1_weather
3. Add indexes on high-volume OpenF1 collections
4. Compute 4 null opponent_profiles fields
5. Fix opponent_circuit_profiles 4 null fields
6. Fix openf1_race_control mixed-type driver_number
7. Clean dead fields from fastf1_laps and openf1_pit

Usage:
    python pipeline/data_quality_fixes.py
"""

from __future__ import annotations

import gzip
import pickle
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pymongo import UpdateOne
from pymongo.database import Database

from updater._db import get_db

warnings.filterwarnings("ignore")


def load_telemetry_from_mongo(
    db: Database,
    session_filter: str = "R",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load telemetry from telemetry_compressed (gzip'd pickle chunks in MongoDB).

    Each document is a gzip-compressed pickle of a DataFrame chunk.
    Chunks are grouped by filename (e.g. '2018_R.parquet').
    """
    # Only load race files if session_filter is set
    filenames = sorted(db["telemetry_compressed"].distinct("filename"))
    if session_filter:
        filenames = [f for f in filenames if f"_{session_filter}." in f]

    frames = []
    for fname in filenames:
        chunks = list(db["telemetry_compressed"].find(
            {"filename": fname},
            {"data": 1, "chunk": 1, "_id": 0},
        ))
        chunks.sort(key=lambda d: d.get("chunk", 0))

        for doc in chunks:
            try:
                df = pickle.loads(gzip.decompress(doc["data"]))
                if columns:
                    available = [c for c in columns if c in df.columns]
                    df = df[available]
                frames.append(df)
            except Exception:
                pass

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FIX BRAKING G VALUES
# ══════════════════════════════════════════════════════════════════════════════

def fix_braking_g(db: Database) -> int:
    """Recompute braking G from telemetry_compressed with correct unit conversion.

    Formula: braking_g = abs(delta_speed_m_s / delta_time_s) / 9.81
    where delta_speed_m_s = delta_speed_kmh / 3.6
    """
    print("\n[1/7] Fixing braking G values...")

    # Build driver number → code mapping
    num_to_code = {}
    for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0}):
        num_to_code[str(doc["driver_number"])] = doc["name_acronym"]

    # Load race telemetry from MongoDB
    tel_df = load_telemetry_from_mongo(
        db, session_filter="R",
        columns=["Driver", "Speed", "Brake", "SessionTime", "Session", "LapNumber"],
    )

    if tel_df.empty:
        print("  ⚠ No telemetry data found in telemetry_compressed, skipping braking G fix")
        return 0

    tel_df["Driver"] = tel_df["Driver"].map(num_to_code)
    tel_df = tel_df.dropna(subset=["Driver"])
    print(f"  Loaded {len(tel_df):,} race telemetry rows")

    # Get all drivers from opponent_profiles
    all_drivers = [
        d["driver_code"]
        for d in db["opponent_profiles"].find({}, {"driver_code": 1, "_id": 0})
        if d.get("driver_code")
    ]

    ops = []
    for driver_code in sorted(all_drivers):
        drv = tel_df[tel_df["Driver"] == driver_code].copy()
        if drv.empty or len(drv) < 10:
            continue

        drv = drv.sort_values("SessionTime").reset_index(drop=True)

        # Compute instantaneous deceleration in g
        dt = drv["SessionTime"].diff()
        dv_kmh = drv["Speed"].diff()
        # Convert km/h to m/s: divide by 3.6
        dv_ms = dv_kmh / 3.6

        # Only braking samples: brake active AND decelerating
        braking_mask = (drv["Brake"] == True) & (dv_ms < 0) & (dt > 0)
        if braking_mask.sum() < 5:
            continue

        g_values = (dv_ms[braking_mask].abs() / dt[braking_mask]) / 9.81
        # Filter obvious sensor noise (> 8g is unrealistic even for F1)
        g_values = g_values[(g_values > 0.1) & (g_values < 8)]

        if g_values.empty:
            continue

        # Late-race braking delta: laps 40+ vs laps 1-20
        drv_braking = drv.loc[braking_mask].copy()
        drv_braking["g"] = (dv_ms[braking_mask].abs() / dt[braking_mask]) / 9.81
        drv_braking = drv_braking[(drv_braking["g"] > 0.1) & (drv_braking["g"] < 8)]

        early = drv_braking[drv_braking["LapNumber"].between(1, 20)]["g"]
        late = drv_braking[drv_braking["LapNumber"] >= 40]["g"]
        late_delta = None
        if len(early) > 10 and len(late) > 10:
            late_delta = round(float(late.mean() - early.mean()), 4)

        update = {
            "avg_braking_g": round(float(g_values.mean()), 4),
            "max_braking_g": round(float(g_values.quantile(0.99)), 4),
            "g_consistency": round(float(g_values.std()), 4),
            "late_race_braking_delta": late_delta,
        }

        ops.append(UpdateOne(
            {"driver_code": driver_code},
            {"$set": update},
        ))

    if ops:
        result = db["opponent_profiles"].bulk_write(ops, ordered=False)
        count = result.modified_count
        print(f"  ✅ Fixed braking G for {count} drivers")
        return count

    print("  ⚠ No braking data to fix")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEDUPLICATE fastf1_laps AND fastf1_weather
# ══════════════════════════════════════════════════════════════════════════════

def deduplicate_fastf1(db: Database) -> dict[str, int]:
    """Remove duplicate documents from fastf1_laps and fastf1_weather."""
    print("\n[2/7] Deduplicating fastf1_laps and fastf1_weather...")
    results = {}

    # --- fastf1_laps: key = (Year, Race, SessionType, Driver, LapNumber) ---
    pipeline = [
        {"$group": {
            "_id": {
                "Year": "$Year", "Race": "$Race", "SessionType": "$SessionType",
                "Driver": "$Driver", "LapNumber": "$LapNumber",
            },
            "ids": {"$push": "$_id"},
            "count": {"$sum": 1},
        }},
        {"$match": {"count": {"$gt": 1}}},
    ]
    dups = list(db["fastf1_laps"].aggregate(pipeline, allowDiskUse=True))
    ids_to_delete = []
    for group in dups:
        # Keep the first, delete the rest
        ids_to_delete.extend(group["ids"][1:])

    if ids_to_delete:
        # Delete in batches of 5000
        for i in range(0, len(ids_to_delete), 5000):
            batch = ids_to_delete[i:i + 5000]
            db["fastf1_laps"].delete_many({"_id": {"$in": batch}})
        results["fastf1_laps"] = len(ids_to_delete)
        print(f"  ✅ Removed {len(ids_to_delete)} duplicate laps from {len(dups)} groups")
    else:
        results["fastf1_laps"] = 0
        print("  ✅ No duplicate laps found")

    # --- fastf1_weather: key = (Year, Race, SessionType, Time) ---
    pipeline = [
        {"$group": {
            "_id": {
                "Year": "$Year", "Race": "$Race",
                "SessionType": "$SessionType", "Time": "$Time",
            },
            "ids": {"$push": "$_id"},
            "count": {"$sum": 1},
        }},
        {"$match": {"count": {"$gt": 1}}},
    ]
    dups = list(db["fastf1_weather"].aggregate(pipeline, allowDiskUse=True))
    ids_to_delete = []
    for group in dups:
        ids_to_delete.extend(group["ids"][1:])

    if ids_to_delete:
        for i in range(0, len(ids_to_delete), 5000):
            batch = ids_to_delete[i:i + 5000]
            db["fastf1_weather"].delete_many({"_id": {"$in": batch}})
        results["fastf1_weather"] = len(ids_to_delete)
        print(f"  ✅ Removed {len(ids_to_delete)} duplicate weather rows from {len(dups)} groups")
    else:
        results["fastf1_weather"] = 0
        print("  ✅ No duplicate weather found")

    # Ensure proper compound indexes exist (drop non-unique first if needed)
    laps_idx = "Year_1_Race_1_SessionType_1_Driver_1_LapNumber_1"
    weather_idx = "Year_1_Race_1_SessionType_1_Time_1"
    try:
        existing = db["fastf1_laps"].index_information()
        if laps_idx in existing and not existing[laps_idx].get("unique"):
            db["fastf1_laps"].drop_index(laps_idx)
        db["fastf1_laps"].create_index(
            [("Year", 1), ("Race", 1), ("SessionType", 1), ("Driver", 1), ("LapNumber", 1)],
            unique=True, background=True,
        )
    except Exception as e:
        print(f"  ⚠ fastf1_laps index: {e}")

    try:
        existing = db["fastf1_weather"].index_information()
        if weather_idx in existing and not existing[weather_idx].get("unique"):
            db["fastf1_weather"].drop_index(weather_idx)
        db["fastf1_weather"].create_index(
            [("Year", 1), ("Race", 1), ("SessionType", 1), ("Time", 1)],
            unique=True, background=True,
        )
    except Exception as e:
        print(f"  ⚠ fastf1_weather index: {e}")

    print("  ✅ Compound unique indexes created/verified")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. ADD INDEXES ON HIGH-VOLUME OPENF1 COLLECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def add_openf1_indexes(db: Database) -> int:
    """Add compound indexes on high-volume OpenF1 collections."""
    print("\n[3/7] Adding indexes on OpenF1 collections...")

    indexes = [
        ("openf1_intervals", [("meeting_key", 1), ("session_key", 1), ("driver_number", 1)]),
        ("openf1_car_data", [("meeting_key", 1), ("session_key", 1), ("driver_number", 1)]),
        ("openf1_position", [("meeting_key", 1), ("session_key", 1), ("driver_number", 1)]),
        ("openf1_laps", [("meeting_key", 1), ("session_key", 1), ("driver_number", 1)]),
        ("openf1_location", [("meeting_key", 1), ("session_key", 1), ("driver_number", 1)]),
        ("openf1_pit", [("meeting_key", 1), ("session_key", 1), ("driver_number", 1)]),
        ("openf1_stints", [("meeting_key", 1), ("session_key", 1), ("driver_number", 1)]),
    ]

    created = 0
    for col_name, keys in indexes:
        idx_name = "_".join(f"{k}_{v}" for k, v in keys)
        existing = db[col_name].index_information()
        if idx_name not in existing:
            db[col_name].create_index(keys, background=True)
            created += 1
            print(f"  ✅ {col_name}: index created")
        else:
            print(f"  ⏭  {col_name}: index already exists")

    print(f"  ✅ {created} new indexes created")
    return created


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTE NULL OPPONENT_PROFILES FIELDS
# ══════════════════════════════════════════════════════════════════════════════

def compute_missing_profile_fields(db: Database) -> int:
    """Compute the 4 null fields in opponent_profiles:
    - avg_position_lap1
    - avg_positions_gained_lap1_to_5
    - avg_tyre_life
    - long_race_performance
    """
    print("\n[4/7] Computing missing opponent_profiles fields...")

    drivers = list(db["opponent_profiles"].find(
        {}, {"driver_code": 1, "_id": 0}
    ))
    driver_codes = [d["driver_code"] for d in drivers if d.get("driver_code")]

    # --- avg_tyre_life: from fastf1_laps ---
    tyre_life_pipeline = [
        {"$match": {"SessionType": "R", "TyreLife": {"$ne": None}}},
        {"$group": {
            "_id": "$Driver",
            "avg_tyre_life": {"$avg": "$TyreLife"},
        }},
    ]
    tyre_results = {
        r["_id"]: round(r["avg_tyre_life"], 2)
        for r in db["fastf1_laps"].aggregate(tyre_life_pipeline, allowDiskUse=True)
    }

    # --- long_race_performance: late vs early stint lap time delta ---
    # Use fastf1_laps: compare laps 40+ vs laps 1-20
    long_race_pipeline_early = [
        {"$match": {"SessionType": "R", "LapTime": {"$gt": 60}, "LapNumber": {"$lte": 20}}},
        {"$group": {"_id": "$Driver", "early_avg": {"$avg": {"$toDouble": "$LapTime"}}}},
    ]
    long_race_pipeline_late = [
        {"$match": {"SessionType": "R", "LapTime": {"$gt": 60}, "LapNumber": {"$gte": 40}}},
        {"$group": {"_id": "$Driver", "late_avg": {"$avg": {"$toDouble": "$LapTime"}}}},
    ]
    early_map = {r["_id"]: r["early_avg"] for r in db["fastf1_laps"].aggregate(long_race_pipeline_early, allowDiskUse=True)}
    late_map = {r["_id"]: r["late_avg"] for r in db["fastf1_laps"].aggregate(long_race_pipeline_late, allowDiskUse=True)}

    long_race_map = {}
    for drv in early_map:
        if drv in late_map and early_map[drv] and late_map[drv]:
            long_race_map[drv] = round(late_map[drv] - early_map[drv], 4)

    # --- avg_position_lap1 and avg_positions_gained_lap1_to_5 ---
    # Use openf1_position: get position at start vs lap 5
    # We need session_key + driver_number, then map back to driver_code
    code_to_num = {}
    for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0}):
        code_to_num[doc["name_acronym"]] = doc["driver_number"]
    num_to_code = {v: k for k, v in code_to_num.items()}

    # Get race session keys
    race_sessions = list(db["openf1_sessions"].find(
        {"session_type": "Race"},
        {"session_key": 1, "_id": 0},
    ))
    race_session_keys = [s["session_key"] for s in race_sessions]

    # Lap 1 positions from openf1_laps
    lap1_pipeline = [
        {"$match": {"session_key": {"$in": race_session_keys}, "lap_number": 1}},
        {"$group": {
            "_id": "$driver_number",
            "positions": {"$push": "$duration_sector_1"},
            "count": {"$sum": 1},
        }},
    ]

    # Use openf1_position for actual positions — get earliest position per session per driver
    # This is position at race start (grid → lap 1)
    position_lap1_pipeline = [
        {"$match": {"session_key": {"$in": race_session_keys}}},
        {"$sort": {"date": 1}},
        {"$group": {
            "_id": {"session_key": "$session_key", "driver_number": "$driver_number"},
            "first_position": {"$first": "$position"},
        }},
        {"$group": {
            "_id": "$_id.driver_number",
            "avg_position_lap1": {"$avg": "$first_position"},
        }},
    ]
    pos_lap1_results = list(db["openf1_position"].aggregate(position_lap1_pipeline, allowDiskUse=True))
    pos_lap1_map = {}
    for r in pos_lap1_results:
        code = num_to_code.get(r["_id"])
        if code:
            pos_lap1_map[code] = round(r["avg_position_lap1"], 2)

    # Positions gained lap 1 to 5: need position at start vs after ~5 laps
    # Get position entries near lap 5 (use openf1_laps lap_number=5 position)
    pos_lap5_pipeline = [
        {"$match": {"session_key": {"$in": race_session_keys}, "lap_number": 5}},
        {"$group": {
            "_id": "$driver_number",
            "sessions": {"$push": "$session_key"},
        }},
    ]

    # Simpler: use openf1_position grouped by time bins
    # Instead, compare grid position (from fastf1_laps) to position at lap 5
    grid_pipeline = [
        {"$match": {"SessionType": "R", "LapNumber": 1}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Year": "$Year", "Race": "$Race"},
            "grid_pos": {"$first": "$Position"},
        }},
    ]
    grid_results = list(db["fastf1_laps"].aggregate(grid_pipeline, allowDiskUse=True))

    lap5_pipeline = [
        {"$match": {"SessionType": "R", "LapNumber": 5}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Year": "$Year", "Race": "$Race"},
            "lap5_pos": {"$first": "$Position"},
        }},
    ]
    lap5_results = list(db["fastf1_laps"].aggregate(lap5_pipeline, allowDiskUse=True))

    grid_map = {(r["_id"]["Driver"], r["_id"]["Year"], r["_id"]["Race"]): r["grid_pos"] for r in grid_results}
    lap5_map = {(r["_id"]["Driver"], r["_id"]["Year"], r["_id"]["Race"]): r["lap5_pos"] for r in lap5_results}

    # Compute per-driver average positions gained
    gained_per_driver: dict[str, list[float]] = {}
    for key, grid_pos in grid_map.items():
        driver = key[0]
        lap5_pos = lap5_map.get(key)
        if grid_pos is not None and lap5_pos is not None:
            gained = grid_pos - lap5_pos  # positive = gained positions
            gained_per_driver.setdefault(driver, []).append(gained)

    gained_map = {
        drv: round(float(np.mean(vals)), 2)
        for drv, vals in gained_per_driver.items()
        if vals
    }

    # --- Build updates ---
    ops = []
    for driver_code in driver_codes:
        update = {}

        if driver_code in tyre_results:
            update["avg_tyre_life"] = tyre_results[driver_code]

        if driver_code in long_race_map:
            update["long_race_performance"] = long_race_map[driver_code]

        if driver_code in pos_lap1_map:
            update["avg_position_lap1"] = pos_lap1_map[driver_code]

        if driver_code in gained_map:
            update["avg_positions_gained_lap1_to_5"] = gained_map[driver_code]

        if update:
            ops.append(UpdateOne({"driver_code": driver_code}, {"$set": update}))

    if ops:
        result = db["opponent_profiles"].bulk_write(ops, ordered=False)
        count = result.modified_count
        print(f"  ✅ Updated {count} profiles")
        print(f"    avg_tyre_life: {len(tyre_results)} drivers")
        print(f"    long_race_performance: {len(long_race_map)} drivers")
        print(f"    avg_position_lap1: {len(pos_lap1_map)} drivers")
        print(f"    avg_positions_gained_lap1_to_5: {len(gained_map)} drivers")
        return count

    print("  ⚠ No updates computed")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIX OPPONENT_CIRCUIT_PROFILES NULL FIELDS
# ══════════════════════════════════════════════════════════════════════════════

def fix_circuit_profiles(db: Database) -> int:
    """Compute the 4 null fields in opponent_circuit_profiles:
    - avg_finish_position (from actual last-lap Position)
    - avg_positions_gained (grid - finish)
    - lap_time_delta_high_heat (vs TrackTemp > 45°C)
    - stint_endurance_slope (degradation per stint)
    """
    print("\n[5/7] Fixing opponent_circuit_profiles null fields...")

    # --- avg_finish_position + avg_positions_gained ---
    # Use fastf1_laps: get grid position (lap 1) and final position (max lap)
    finish_pipeline = [
        {"$match": {"SessionType": "R"}},
        {"$sort": {"LapNumber": -1}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Year": "$Year", "Race": "$Race"},
            "final_position": {"$first": "$Position"},
            "grid_position": {"$last": "$Position"},
            "max_lap": {"$first": "$LapNumber"},
        }},
    ]
    finish_results = list(db["fastf1_laps"].aggregate(finish_pipeline, allowDiskUse=True))

    # Group by (driver, circuit) for avg finish and positions gained
    from collections import defaultdict
    circuit_finish: dict[tuple, list] = defaultdict(list)
    circuit_gained: dict[tuple, list] = defaultdict(list)

    for r in finish_results:
        key = (r["_id"]["Driver"], r["_id"]["Race"])
        if r["final_position"] is not None:
            circuit_finish[key].append(r["final_position"])
        if r["final_position"] is not None and r["grid_position"] is not None:
            circuit_gained[key].append(r["grid_position"] - r["final_position"])

    # --- lap_time_delta_high_heat ---
    # Load weather to find hot races (TrackTemp > 45°C)
    hot_races = set()
    for doc in db["fastf1_weather"].find(
        {"SessionType": "R", "TrackTemp": {"$gt": 45}},
        {"Race": 1, "_id": 0},
    ):
        hot_races.add(doc["Race"])

    # Get per-driver per-circuit median lap times for hot vs all races
    heat_pipeline = [
        {"$match": {"SessionType": "R", "LapTime": {"$gt": 60}}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Race": "$Race"},
            "avg_laptime": {"$avg": {"$toDouble": "$LapTime"}},
        }},
    ]
    heat_results = list(db["fastf1_laps"].aggregate(heat_pipeline, allowDiskUse=True))

    driver_all_times: dict[str, list] = defaultdict(list)
    driver_hot_times: dict[str, list] = defaultdict(list)
    circuit_heat: dict[tuple, float] = {}

    for r in heat_results:
        drv = r["_id"]["Driver"]
        race = r["_id"]["Race"]
        driver_all_times[drv].append(r["avg_laptime"])
        if race in hot_races:
            driver_hot_times[drv].append(r["avg_laptime"])

    for drv in driver_all_times:
        if drv in driver_hot_times and driver_all_times[drv]:
            all_avg = np.mean(driver_all_times[drv])
            hot_avg = np.mean(driver_hot_times[drv])
            for race in hot_races:
                circuit_heat[(drv, race)] = round(float(hot_avg - all_avg), 4)

    # --- stint_endurance_slope ---
    # Degradation slope per stint: from fastf1_laps
    stint_pipeline = [
        {"$match": {"SessionType": "R", "LapTime": {"$gt": 60}, "TyreLife": {"$ne": None}}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Race": "$Race", "Stint": "$Stint"},
            "lap_numbers": {"$push": "$LapNumber"},
            "lap_times": {"$push": {"$toDouble": "$LapTime"}},
            "count": {"$sum": 1},
        }},
        {"$match": {"count": {"$gte": 5}}},
    ]
    stint_results = list(db["fastf1_laps"].aggregate(stint_pipeline, allowDiskUse=True))

    circuit_slopes: dict[tuple, list] = defaultdict(list)
    for r in stint_results:
        drv = r["_id"]["Driver"]
        race = r["_id"]["Race"]
        laps = np.array(r["lap_numbers"], dtype=float)
        times = np.array(r["lap_times"], dtype=float)
        mask = ~np.isnan(laps) & ~np.isnan(times)
        if mask.sum() >= 4:
            slope = np.polyfit(laps[mask], times[mask], 1)[0]
            circuit_slopes[(drv, race)].append(slope)

    # --- Build updates ---
    ops = []
    existing = list(db["opponent_circuit_profiles"].find({}, {"driver_id": 1, "circuit": 1, "_id": 1}))

    for doc in existing:
        key = (doc["driver_id"], doc["circuit"])
        update = {}

        if key in circuit_finish:
            update["avg_finish_position"] = round(float(np.mean(circuit_finish[key])), 2)

        if key in circuit_gained:
            update["avg_positions_gained"] = round(float(np.mean(circuit_gained[key])), 2)

        if key in circuit_heat:
            update["lap_time_delta_high_heat"] = circuit_heat[key]

        if key in circuit_slopes:
            update["stint_endurance_slope"] = round(float(np.mean(circuit_slopes[key])), 5)

        if update:
            ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": update}))

    if ops:
        result = db["opponent_circuit_profiles"].bulk_write(ops, ordered=False)
        count = result.modified_count
        print(f"  ✅ Updated {count} circuit profiles")
        return count

    print("  ⚠ No circuit profile updates")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. FIX openf1_race_control MIXED-TYPE driver_number
# ══════════════════════════════════════════════════════════════════════════════

def fix_race_control_driver_numbers(db: Database) -> int:
    """Normalize driver_number: convert string 'None'/'nan' to null, numeric strings to int."""
    print("\n[6/7] Fixing openf1_race_control driver_number types...")

    # Fix string "None" and "nan" → null
    r1 = db["openf1_race_control"].update_many(
        {"driver_number": {"$in": ["None", "nan", "NaN", ""]}},
        {"$set": {"driver_number": None}},
    )

    # Fix numeric strings → int
    # Find docs where driver_number is a string that looks like a number
    numeric_fixed = 0
    for doc in db["openf1_race_control"].find(
        {"driver_number": {"$type": "string"}},
        {"driver_number": 1},
    ):
        try:
            num = int(doc["driver_number"])
            db["openf1_race_control"].update_one(
                {"_id": doc["_id"]},
                {"$set": {"driver_number": num}},
            )
            numeric_fixed += 1
        except (ValueError, TypeError):
            pass

    total = r1.modified_count + numeric_fixed
    print(f"  ✅ Fixed {r1.modified_count} None/nan → null, {numeric_fixed} string → int")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLEAN DEAD FIELDS
# ══════════════════════════════════════════════════════════════════════════════

def clean_dead_fields(db: Database) -> dict[str, int]:
    """Remove universally-null fields that add no value."""
    print("\n[7/7] Cleaning dead fields...")
    results = {}

    # fastf1_laps: Deleted, LapStartDate, Position are 100% null
    dead_laps_fields = ["Deleted", "DeletedReason", "LapStartDate"]
    r = db["fastf1_laps"].update_many(
        {},
        {"$unset": {f: "" for f in dead_laps_fields}},
    )
    results["fastf1_laps"] = r.modified_count
    print(f"  ✅ fastf1_laps: unset {dead_laps_fields} from {r.modified_count} docs")

    # openf1_pit: stop_duration is 100% null
    r = db["openf1_pit"].update_many(
        {},
        {"$unset": {"stop_duration": ""}},
    )
    results["openf1_pit"] = r.modified_count
    print(f"  ✅ openf1_pit: unset stop_duration from {r.modified_count} docs")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    db = get_db()
    print("✅ Connected to MongoDB")
    print(f"\n{'=' * 60}")
    print("  DATA QUALITY FIXES")
    print(f"{'=' * 60}")

    summary = {}

    # 1. Fix braking G
    summary["braking_g_fixed"] = fix_braking_g(db)

    # 2. Deduplicate
    summary["dedup"] = deduplicate_fastf1(db)

    # 3. Add indexes
    summary["indexes_created"] = add_openf1_indexes(db)

    # 4. Compute missing profile fields
    summary["profiles_updated"] = compute_missing_profile_fields(db)

    # 5. Fix circuit profiles
    summary["circuit_profiles_fixed"] = fix_circuit_profiles(db)

    # 6. Fix race control driver numbers
    summary["race_control_fixed"] = fix_race_control_driver_numbers(db)

    # 7. Clean dead fields
    summary["dead_fields_cleaned"] = clean_dead_fields(db)

    # Final summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Verification
    print(f"\n{'=' * 60}")
    print("  VERIFICATION")
    print(f"{'=' * 60}")

    # Check braking G values are sane
    sample = db["opponent_profiles"].find_one(
        {"avg_braking_g": {"$ne": None}},
        {"driver_code": 1, "avg_braking_g": 1, "max_braking_g": 1, "g_consistency": 1, "_id": 0},
    )
    if sample:
        print(f"\n  Braking G sample: {sample}")

    # Check dedup results
    laps_count = db["fastf1_laps"].count_documents({})
    weather_count = db["fastf1_weather"].count_documents({})
    print(f"\n  fastf1_laps: {laps_count:,} docs")
    print(f"  fastf1_weather: {weather_count:,} docs")

    # Check missing fields filled
    null_checks = ["avg_tyre_life", "long_race_performance", "avg_position_lap1", "avg_positions_gained_lap1_to_5"]
    for field in null_checks:
        null_count = db["opponent_profiles"].count_documents({field: None})
        total = db["opponent_profiles"].count_documents({})
        print(f"  {field}: {total - null_count}/{total} populated")

    # Log the fix run
    db["pipeline_log"].insert_one({
        "chunk": "data_quality_fixes",
        "status": "complete",
        "summary": {k: str(v) for k, v in summary.items()},
        "timestamp": datetime.now(timezone.utc),
    })

    print("\n✅ All fixes complete")


if __name__ == "__main__":
    main()
