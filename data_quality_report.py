#!/usr/bin/env python3
"""
Data Quality Analysis for marip_f1 MongoDB database.
Checks cross-collection consistency, completeness, freshness, and anomalies.
"""

import os
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv("/home/pedrad/javier_project_folder/f1/.env")

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = "marip_f1"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ── Discover collections ──
collections = sorted(db.list_collection_names())
print("=" * 90)
print(f"  DATA QUALITY REPORT -- database: {DB_NAME}")
print(f"  Generated: {datetime.now(timezone.utc).isoformat()}")
print(f"  Collections found ({len(collections)}): {', '.join(collections)}")
print("=" * 90)

# Helper: count docs
for coll_name in collections:
    cnt = db[coll_name].estimated_document_count()
    print(f"  {coll_name:40s}  {cnt:>10,} docs")
print()


def safe_distinct(coll_name, field, filt=None):
    """Use aggregation $group instead of distinct to avoid 16MB cap."""
    pipeline = []
    if filt:
        pipeline.append({"$match": filt})
    pipeline.append({"$group": {"_id": f"${field}"}})
    results = list(db[coll_name].aggregate(pipeline, allowDiskUse=True))
    return [r["_id"] for r in results]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CROSS-COLLECTION CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("  1. CROSS-COLLECTION CONSISTENCY")
print("=" * 90)

# --- 1a. Unique drivers per collection ---
# Explicit field map based on actual schemas
driver_fields_map = {
    "fastf1_laps": "Driver",
    "fastf1_weather": None,
    "openf1_car_data": "driver_number",
    "openf1_drivers": "driver_number",
    "openf1_intervals": "driver_number",
    "openf1_laps": "driver_number",
    "openf1_location": "driver_number",
    "openf1_pit": "driver_number",
    "openf1_position": "driver_number",
    "openf1_race_control": "driver_number",
    "openf1_stints": "driver_number",
    "openf1_weather": None,
    "openf1_sessions": None,
    "opponent_profiles": "driver_id",
    "opponent_circuit_profiles": None,
    "opponent_compound_profiles": None,
    "driver_performance_markers": "Driver",
    "circuit_pit_loss_times": None,
    "pipeline_log": None,
    "telemetry_compressed": None,
}

print("\n  1a. Unique DRIVERS per collection")
print("  " + "-" * 70)
all_driver_sets = {}
for coll_name in collections:
    field = driver_fields_map.get(coll_name, "__auto__")
    if field == "__auto__":
        sample = db[coll_name].find_one()
        if sample:
            for f in ["Driver", "driver", "driver_number", "driver_id", "DriverNumber"]:
                if f in sample:
                    field = f
                    break
            else:
                field = None
    if field:
        vals = safe_distinct(coll_name, field)
        non_null = [v for v in vals if v is not None]
        all_driver_sets[coll_name] = set(str(v) for v in non_null)
        print(f"    {coll_name:40s}  field='{field}'  unique={len(non_null)}")
    else:
        print(f"    {coll_name:40s}  (no driver field)")

# Cross-check overlap (only meaningful pairs)
key_pairs = [
    ("fastf1_laps", "driver_performance_markers"),
    ("fastf1_laps", "opponent_profiles"),
    ("driver_performance_markers", "opponent_profiles"),
    ("openf1_laps", "openf1_intervals"),
    ("openf1_laps", "openf1_stints"),
    ("openf1_laps", "openf1_pit"),
]
print("\n  Driver overlap (selected pairs):")
for c1, c2 in key_pairs:
    if c1 in all_driver_sets and c2 in all_driver_sets:
        s1, s2 = all_driver_sets[c1], all_driver_sets[c2]
        overlap = s1 & s2
        only1 = s1 - s2
        only2 = s2 - s1
        print(f"    {c1} <-> {c2}:")
        print(f"      shared={len(overlap)}  only-in-1={len(only1)}  only-in-2={len(only2)}")

# --- 1b. Unique races/sessions per collection ---
print("\n  1b. Unique RACES / SESSIONS per collection")
print("  " + "-" * 70)
session_fields_map = {
    "fastf1_laps": ["EventName", "SessionType"],
    "fastf1_weather": ["EventName", "SessionType"],
    "openf1_intervals": ["meeting_key", "session_key"],
    "openf1_laps": ["meeting_key", "session_key"],
    "openf1_car_data": ["meeting_key", "session_key"],
    "openf1_location": ["meeting_key"],
    "openf1_pit": ["meeting_key"],
    "openf1_position": ["meeting_key"],
    "openf1_race_control": ["meeting_key"],
    "openf1_sessions": ["meeting_key"],
    "openf1_stints": ["meeting_key"],
    "openf1_weather": ["meeting_key"],
}

for coll_name in collections:
    fields = session_fields_map.get(coll_name)
    if not fields:
        sample = db[coll_name].find_one()
        if sample:
            for candidates in [["EventName"], ["meeting_key"], ["race"], ["Race"], ["session_key"]]:
                if all(c in sample for c in candidates):
                    fields = candidates
                    break
    if fields:
        pipeline = [
            {"$group": {"_id": {f: f"${f}" for f in fields}}},
            {"$count": "total"}
        ]
        res = list(db[coll_name].aggregate(pipeline, allowDiskUse=True))
        count = res[0]["total"] if res else 0
        print(f"    {coll_name:40s}  keys={fields}  unique_combos={count}")
    else:
        print(f"    {coll_name:40s}  (no session/race field found)")

# --- 1c. Year coverage per collection ---
print("\n  1c. YEAR coverage per collection")
print("  " + "-" * 70)
year_field_candidates = ["Year", "year", "date_start", "Date", "date"]

for coll_name in collections:
    sample = db[coll_name].find_one()
    if not sample:
        print(f"    {coll_name:40s}  (empty collection)")
        continue
    year_field = None
    for yf in year_field_candidates:
        if yf in sample:
            year_field = yf
            break
    # Also check years_covered (list field in driver_performance_markers)
    if year_field is None and "years_covered" in sample:
        # Unwind and get distinct years
        pipeline = [
            {"$unwind": "$years_covered"},
            {"$group": {"_id": "$years_covered"}},
            {"$sort": {"_id": 1}}
        ]
        res = list(db[coll_name].aggregate(pipeline))
        years = sorted([r["_id"] for r in res if r["_id"] is not None])
        print(f"    {coll_name:40s}  field='years_covered'  years={years}")
        continue
    # Check seasons field (opponent_profiles)
    if year_field is None and "seasons" in sample:
        pipeline = [
            {"$unwind": "$seasons"},
            {"$group": {"_id": "$seasons"}},
            {"$sort": {"_id": 1}}
        ]
        res = list(db[coll_name].aggregate(pipeline))
        years = sorted([r["_id"] for r in res if r["_id"] is not None])
        print(f"    {coll_name:40s}  field='seasons'  years={years}")
        continue

    if year_field:
        # Use aggregation to extract years safely
        sample_val = sample.get(year_field)
        if isinstance(sample_val, (int, float)):
            # Numeric year field -- safe to distinct
            vals = safe_distinct(coll_name, year_field)
            years = sorted([int(v) for v in vals if v is not None])
            print(f"    {coll_name:40s}  field='{year_field}'  years={years}")
        elif isinstance(sample_val, str):
            # String date -- extract year via $substr aggregation
            pipeline = [
                {"$group": {"_id": {"$substr": [f"${year_field}", 0, 4]}}},
                {"$sort": {"_id": 1}}
            ]
            res = list(db[coll_name].aggregate(pipeline, allowDiskUse=True))
            years = sorted([r["_id"] for r in res if r["_id"] and r["_id"].isdigit()])
            print(f"    {coll_name:40s}  field='{year_field}'  years={years}")
        elif isinstance(sample_val, datetime):
            # datetime -- extract year
            pipeline = [
                {"$group": {"_id": {"$year": f"${year_field}"}}},
                {"$sort": {"_id": 1}}
            ]
            res = list(db[coll_name].aggregate(pipeline, allowDiskUse=True))
            years = sorted([r["_id"] for r in res if r["_id"] is not None])
            print(f"    {coll_name:40s}  field='{year_field}'  years={years}")
        else:
            print(f"    {coll_name:40s}  field='{year_field}'  (unhandled type: {type(sample_val).__name__})")
    else:
        print(f"    {coll_name:40s}  (no year/date field found)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA COMPLETENESS (null rates)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  2. DATA COMPLETENESS (null / missing rates)")
print("=" * 90)


def null_rate_report(coll_name, fields, group_by=None):
    """Compute null rates for given fields, optionally grouped."""
    coll = db[coll_name]
    total = coll.estimated_document_count()
    if total == 0:
        print(f"    {coll_name}: empty collection, skipping")
        return

    if group_by:
        groups = safe_distinct(coll_name, group_by)
        for g in sorted(str(x) for x in groups if x is not None):
            filt = {group_by: g}
            grp_total = coll.count_documents(filt)
            print(f"\n    [{coll_name}] {group_by}={g}  (n={grp_total:,})")
            for f in fields:
                null_count = coll.count_documents({**filt, "$or": [{f: None}, {f: {"$exists": False}}]})
                pct = (null_count / grp_total * 100) if grp_total else 0
                bar = "#" * int(pct // 5) + "." * (20 - int(pct // 5))
                print(f"      {f:25s}  null={null_count:>8,} / {grp_total:>8,}  ({pct:5.1f}%)  |{bar}|")
    else:
        print(f"\n    [{coll_name}]  (n={total:,})")
        for f in fields:
            null_count = coll.count_documents({"$or": [{f: None}, {f: {"$exists": False}}]})
            pct = (null_count / total * 100) if total else 0
            bar = "#" * int(pct // 5) + "." * (20 - int(pct // 5))
            print(f"      {f:25s}  null={null_count:>8,} / {total:>8,}  ({pct:5.1f}%)  |{bar}|")


# 2a. fastf1_laps
if "fastf1_laps" in collections:
    null_rate_report("fastf1_laps",
                     ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
                      "Compound", "TyreLife", "Stint"],
                     group_by="SessionType")

# 2b. fastf1_weather
if "fastf1_weather" in collections:
    null_rate_report("fastf1_weather",
                     ["TrackTemp", "Humidity", "AirTemp"])

# 2c. openf1_intervals
if "openf1_intervals" in collections:
    null_rate_report("openf1_intervals",
                     ["gap_to_leader", "interval"])

# 2d. openf1_laps
if "openf1_laps" in collections:
    null_rate_report("openf1_laps",
                     ["lap_duration", "is_pit_out_lap"])

# 2e. opponent_profiles: find which fields have most nulls
if "opponent_profiles" in collections:
    print(f"\n    [opponent_profiles] -- fields with most nulls")
    sample = db["opponent_profiles"].find_one()
    if sample:
        total = db["opponent_profiles"].estimated_document_count()
        field_nulls = []
        for key in sample.keys():
            if key == "_id":
                continue
            nc = db["opponent_profiles"].count_documents(
                {"$or": [{key: None}, {key: {"$exists": False}}]})
            if nc > 0:
                field_nulls.append((key, nc, nc / total * 100))
        field_nulls.sort(key=lambda x: -x[1])
        for fname, nc, pct in field_nulls[:20]:
            bar = "#" * int(pct // 5) + "." * (20 - int(pct // 5))
            print(f"      {fname:40s}  null={nc:>6,} / {total:>6,}  ({pct:5.1f}%)  |{bar}|")
        if not field_nulls:
            print("      (no null fields found)")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA FRESHNESS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  3. DATA FRESHNESS")
print("=" * 90)

ts_fields = ["ingested_at", "updated_at", "created_at", "timestamp", "career_stats_updated_at"]

for coll_name in collections:
    sample = db[coll_name].find_one()
    if not sample:
        print(f"    {coll_name:40s}  (empty)")
        continue

    latest_ts = None
    ts_field_used = None
    for tf in ts_fields:
        if tf in sample:
            doc = db[coll_name].find_one(sort=[(tf, -1)])
            if doc and doc.get(tf):
                latest_ts = doc[tf]
                ts_field_used = tf
                break

    if latest_ts is None:
        # fallback to _id timestamp
        doc = db[coll_name].find_one(sort=[("_id", -1)])
        if doc:
            latest_ts = doc["_id"].generation_time
            ts_field_used = "_id (ObjectId)"

    # Most recent year/race
    year_field = None
    for yf in ["Year", "year"]:
        if yf in sample:
            year_field = yf
            break
    race_field = None
    for rf in ["EventName", "race", "Race", "meeting_key"]:
        if rf in sample:
            race_field = rf
            break

    latest_year = None
    latest_race = None
    if year_field:
        vals = safe_distinct(coll_name, year_field)
        numeric = [int(v) for v in vals if v is not None and str(v).isdigit()]
        if numeric:
            latest_year = max(numeric)
    if race_field and year_field and latest_year:
        races = safe_distinct(coll_name, race_field, {year_field: latest_year})
        latest_race = races[-1] if races else None
    elif race_field:
        doc = db[coll_name].find_one(sort=[("_id", -1)])
        latest_race = doc.get(race_field) if doc else None

    print(f"    {coll_name:40s}  latest_ts={latest_ts}  (via {ts_field_used})")
    if latest_year or latest_race:
        print(f"    {'':40s}  latest_year={latest_year}  latest_race={latest_race}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  4. ANOMALIES")
print("=" * 90)

# --- 4a. Duplicate documents by natural key ---
print("\n  4a. Duplicate documents (by natural key)")
print("  " + "-" * 70)

natural_keys = {
    "fastf1_laps": ["Year", "EventName", "SessionType", "Driver", "LapNumber"],
    "fastf1_weather": ["Year", "EventName", "SessionType", "Time"],
    "openf1_intervals": ["session_key", "driver_number", "date"],
    "openf1_laps": ["session_key", "driver_number", "lap_number"],
    "openf1_stints": ["session_key", "driver_number", "stint_number"],
    "openf1_pit": ["session_key", "driver_number", "lap_number"],
    "openf1_position": ["session_key", "driver_number", "date"],
    "opponent_profiles": ["driver_id"],
    "driver_performance_markers": ["Driver"],
}

# Skip collections > 500k docs for dupe check (Atlas free-tier memory limit)
DUPE_SKIP_THRESHOLD = 500_000

for coll_name in collections:
    keys = natural_keys.get(coll_name)
    if not keys:
        sample = db[coll_name].find_one()
        if sample:
            print(f"    {coll_name:40s}  (no natural key defined, skipping)")
        continue
    doc_count = db[coll_name].estimated_document_count()
    if doc_count > DUPE_SKIP_THRESHOLD:
        print(f"    {coll_name:40s}  (skipped: {doc_count:,} docs exceeds memory-safe threshold)")
        continue
    # Verify keys exist
    sample = db[coll_name].find_one()
    if not sample:
        continue
    valid_keys = [k for k in keys if k in sample]
    if len(valid_keys) < 1:
        print(f"    {coll_name:40s}  (natural key fields missing, skipping)")
        continue

    try:
        pipeline = [
            {"$group": {
                "_id": {k: f"${k}" for k in valid_keys},
                "count": {"$sum": 1}
            }},
            {"$match": {"count": {"$gt": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        dupes = list(db[coll_name].aggregate(pipeline, allowDiskUse=True))
        total_dupe_pipeline = [
            {"$group": {
                "_id": {k: f"${k}" for k in valid_keys},
                "count": {"$sum": 1}
            }},
            {"$match": {"count": {"$gt": 1}}},
            {"$count": "total_dupe_groups"}
        ]
        dupe_count_res = list(db[coll_name].aggregate(total_dupe_pipeline, allowDiskUse=True))
        dupe_groups = dupe_count_res[0]["total_dupe_groups"] if dupe_count_res else 0

        if dupes:
            print(f"    {coll_name:40s}  WARNING: {dupe_groups} duplicate groups!  key={valid_keys}")
            for d in dupes[:3]:
                print(f"      example: {d['_id']}  count={d['count']}")
        else:
            print(f"    {coll_name:40s}  OK: No duplicates  key={valid_keys}")
    except Exception as e:
        print(f"    {coll_name:40s}  ERROR: {e}")

# --- 4b. Obviously wrong values ---
print("\n  4b. Obviously wrong values")
print("  " + "-" * 70)

# Negative lap times
if "fastf1_laps" in collections:
    neg_lap = db["fastf1_laps"].count_documents({"LapTime": {"$lt": 0}})
    neg_s1 = db["fastf1_laps"].count_documents({"Sector1Time": {"$lt": 0}})
    neg_s2 = db["fastf1_laps"].count_documents({"Sector2Time": {"$lt": 0}})
    neg_s3 = db["fastf1_laps"].count_documents({"Sector3Time": {"$lt": 0}})
    print(f"    fastf1_laps  negative LapTime={neg_lap}  S1={neg_s1}  S2={neg_s2}  S3={neg_s3}")

    # Speeds > 400
    speed_fields = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
    for sf in speed_fields:
        cnt = db["fastf1_laps"].count_documents({sf: {"$gt": 400}})
        if cnt > 0:
            print(f"    fastf1_laps  {sf} > 400 km/h: {cnt} docs")
        else:
            print(f"    fastf1_laps  {sf} > 400 km/h: 0 (OK)")

if "openf1_laps" in collections:
    neg_dur = db["openf1_laps"].count_documents({"lap_duration": {"$lt": 0}})
    print(f"    openf1_laps  negative lap_duration: {neg_dur}")
    short = db["openf1_laps"].count_documents({"lap_duration": {"$lt": 30, "$gt": 0}})
    long_ = db["openf1_laps"].count_documents({"lap_duration": {"$gt": 300}})
    print(f"    openf1_laps  lap_duration < 30s: {short}   lap_duration > 300s: {long_}")

if "fastf1_weather" in collections:
    neg_temp = db["fastf1_weather"].count_documents({"TrackTemp": {"$lt": -20}})
    high_temp = db["fastf1_weather"].count_documents({"TrackTemp": {"$gt": 80}})
    print(f"    fastf1_weather  TrackTemp < -20C: {neg_temp}   TrackTemp > 80C: {high_temp}")

if "opponent_profiles" in collections:
    # Check for obviously wrong braking values (billions of g's)
    absurd_braking = db["opponent_profiles"].count_documents({"avg_braking_g": {"$gt": 100}})
    absurd_max_braking = db["opponent_profiles"].count_documents({"max_braking_g": {"$gt": 100}})
    print(f"    opponent_profiles  avg_braking_g > 100: {absurd_braking}  max_braking_g > 100: {absurd_max_braking}")
    if absurd_braking > 0:
        examples = list(db["opponent_profiles"].find(
            {"avg_braking_g": {"$gt": 100}},
            {"driver_id": 1, "avg_braking_g": 1, "max_braking_g": 1, "_id": 0}
        ).limit(3))
        for ex in examples:
            print(f"      example: {ex}")

# --- 4c. driver_performance_markers: null telemetry markers ---
print("\n  4c. driver_performance_markers -- null telemetry markers")
print("  " + "-" * 70)

if "driver_performance_markers" in collections:
    sample = db["driver_performance_markers"].find_one()
    if sample:
        total = db["driver_performance_markers"].estimated_document_count()
        # Telemetry-related fields based on actual schema
        telem_fields = [k for k in sample.keys()
                        if any(t in k.lower() for t in ["speed", "throttle",
                                                         "brake", "drs", "gear", "rpm"])]
        if not telem_fields:
            # Broader search: any field that sounds telemetry-like
            telem_fields = [k for k in sample.keys()
                            if k != "_id" and k != "Driver" and k != "years_covered"
                            and k != "total_race_laps"]
            print(f"    (Using all metric fields as telemetry proxy)")

        print(f"    Total docs: {total:,}")
        print(f"    Telemetry/metric fields ({len(telem_fields)}): {telem_fields}")
        null_summary = []
        for tf in telem_fields:
            nc = db["driver_performance_markers"].count_documents(
                {"$or": [{tf: None}, {tf: {"$exists": False}}]})
            pct = nc / total * 100 if total else 0
            null_summary.append((tf, nc, pct))
            print(f"      {tf:40s}  null={nc:>4,} / {total:>4,}  ({pct:5.1f}%)")

        # Drivers with ALL telemetry fields null
        strict_telem = [k for k in sample.keys()
                        if any(t in k.lower() for t in ["speed", "throttle", "brake"])]
        if strict_telem:
            null_filter = {"$and": [
                {"$or": [{tf: None}, {tf: {"$exists": False}}]} for tf in strict_telem
            ]}
            all_null = db["driver_performance_markers"].count_documents(null_filter)
            drivers_null = safe_distinct("driver_performance_markers", "Driver", null_filter)
            print(f"\n    Drivers with ALL of {strict_telem} null:")
            print(f"      {all_null} docs / {len(drivers_null)} drivers")
            if drivers_null:
                print(f"      Drivers: {sorted(str(d) for d in drivers_null[:30])}")
    else:
        print("    (empty collection)")
else:
    print("    (collection not found)")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  END OF DATA QUALITY REPORT")
print("=" * 90)

client.close()
