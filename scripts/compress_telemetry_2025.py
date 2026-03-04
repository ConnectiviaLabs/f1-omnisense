#!/usr/bin/env python3
"""Fetch 2025 FastF1 telemetry, compress, and upload to MongoDB.

Usage (on RunPod):
    MONGODB_URI=mongodb://localhost:27017/marip_f1 python3 scripts/compress_telemetry_2025.py

Takes ~30-60 min for a full season.
"""

import gzip
import os
import pickle
import time

import fastf1
import pandas as pd
from pymongo import MongoClient, UpdateOne

# ── Config ─────────────────────────────────────────────────────
YEAR = 2025
SESSION_TYPES = ["R"]
CHUNK_SIZE = 500_000
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/marip_f1")
DB_NAME = os.environ.get("MONGODB_DB", "marip_f1")

session_code_map = {"R": "Race", "Q": "Qualifying", "S": "Sprint"}


def main():
    print(f"Year: {YEAR}")
    print(f"Sessions: {SESSION_TYPES}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")
    print(f"MongoDB: {MONGODB_URI[:50]}...")

    # Connect
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    print(f"Connected to {DB_NAME}")

    for st in SESSION_TYPES:
        fname = f"{YEAR}_{st}.parquet"
        existing = db["telemetry_compressed"].count_documents({"filename": fname})
        print(f"  {fname}: {existing} chunks already exist")

    # FastF1 cache
    cache_dir = "/tmp/fastf1_cache"
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    # Get schedule
    schedule = fastf1.get_event_schedule(YEAR)
    events = schedule[schedule["EventFormat"] != "testing"]
    print(f"\n{YEAR} schedule: {len(events)} events")
    for _, ev in events.iterrows():
        print(f"  Round {ev['RoundNumber']:2d}: {ev['EventName']}")

    # ── Fetch telemetry ────────────────────────────────────────
    all_telemetry = []
    race_summaries = []
    failed = []

    for _, event in events.iterrows():
        round_num = int(event["RoundNumber"])
        event_name = event["EventName"]

        for st_code in SESSION_TYPES:
            st_name = session_code_map[st_code]
            label = f"Round {round_num:2d} {event_name} ({st_name})"

            try:
                t0 = time.time()
                session = fastf1.get_session(YEAR, event_name, st_name)
                session.load(telemetry=True, messages=False, weather=False)

                laps = session.laps
                if laps.empty:
                    print(f"  SKIP {label} — no laps")
                    continue

                frames = []
                for drv in laps["DriverNumber"].unique():
                    drv_laps = laps.pick_driver(drv)
                    for _, lap in drv_laps.iterrows():
                        try:
                            car = lap.get_car_data().add_distance()
                        except Exception:
                            continue
                        if car.empty:
                            continue

                        car["Driver"] = str(drv)
                        car["LapNumber"] = int(lap["LapNumber"]) if pd.notna(lap.get("LapNumber")) else 0

                        lt = lap.get("LapTime")
                        if pd.notna(lt):
                            car["LapTime_s"] = lt.total_seconds() if hasattr(lt, "total_seconds") else float(lt)
                        else:
                            car["LapTime_s"] = None

                        car["Compound"] = lap.get("Compound", "")
                        car["TyreLife"] = int(lap["TyreLife"]) if pd.notna(lap.get("TyreLife")) else 0
                        car["Stint"] = int(lap["Stint"]) if pd.notna(lap.get("Stint")) else 0
                        car["TrackStatus"] = str(lap.get("TrackStatus", ""))
                        frames.append(car)

                if not frames:
                    print(f"  SKIP {label} — no car data")
                    continue

                df = pd.concat(frames, ignore_index=True)
                df["Year"] = YEAR
                df["Round"] = round_num
                df["Race"] = event_name
                df["Session"] = st_code

                for col in ["Date", "Time", "SessionTime"]:
                    if col in df.columns:
                        if hasattr(df[col].dtype, "tz"):
                            df[col] = df[col].dt.tz_localize(None)
                        if pd.api.types.is_timedelta64_dtype(df[col]):
                            df[col] = df[col].dt.total_seconds()
                        elif pd.api.types.is_datetime64_any_dtype(df[col]):
                            df[col] = df[col].astype(str)

                expected_cols = [
                    "Date", "RPM", "Speed", "nGear", "Throttle", "Brake", "DRS",
                    "Source", "Time", "SessionTime", "Distance",
                    "Year", "Round", "Race", "Session", "Driver",
                    "LapNumber", "LapTime_s", "Compound", "TyreLife", "Stint", "TrackStatus",
                ]
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = None
                df = df[expected_cols]

                all_telemetry.append(df)
                elapsed = time.time() - t0

                # Build race summary per driver
                for drv_num, grp in df.groupby("Driver"):
                    speeds = grp["Speed"].dropna()
                    rpms = grp["RPM"].dropna()
                    n = len(grp)
                    if n < 100:
                        continue
                    race_summaries.append({
                        "Race": event_name,
                        "Year": YEAR,
                        "Driver": str(drv_num),
                        "avg_speed": round(speeds.mean(), 1) if len(speeds) else 0,
                        "top_speed": int(speeds.max()) if len(speeds) else 0,
                        "avg_rpm": int(rpms.mean()) if len(rpms) else 0,
                        "max_rpm": int(rpms.max()) if len(rpms) else 0,
                        "avg_throttle": round(grp["Throttle"].dropna().mean(), 1),
                        "brake_pct": round((grp["Brake"].sum() / n) * 100, 1),
                        "drs_pct": round((grp["DRS"].dropna().ge(10).sum() / n) * 100, 1),
                        "compounds": list(grp["Compound"].dropna().unique()),
                        "samples": n,
                    })

                total_rows = sum(len(d) for d in all_telemetry)
                print(f"  OK {label}: {len(df):,} rows ({elapsed:.0f}s) — total: {total_rows:,}")

            except Exception as e:
                print(f"  FAIL {label}: {e}")
                failed.append((event_name, st_code, str(e)))

    total_rows = sum(len(d) for d in all_telemetry)
    print(f"\nDone fetching: {len(all_telemetry)} races, {total_rows:,} total rows")
    if failed:
        print(f"Failed: {len(failed)}")
        for name, st, err in failed:
            print(f"  {name} ({st}): {err}")

    # ── Compress and upload ────────────────────────────────────
    if not all_telemetry:
        print("No telemetry to upload!")
        return

    combined = pd.concat(all_telemetry, ignore_index=True)
    print(f"\nCombined: {combined.shape}")

    for st_code in SESSION_TYPES:
        subset = combined[combined["Session"] == st_code]
        if subset.empty:
            continue

        filename = f"{YEAR}_{st_code}.parquet"
        n_chunks = (len(subset) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"\nUploading {filename}: {len(subset):,} rows in {n_chunks} chunks")

        deleted = db["telemetry_compressed"].delete_many({"filename": filename})
        if deleted.deleted_count:
            print(f"  Deleted {deleted.deleted_count} existing chunks")

        for i in range(n_chunks):
            chunk_df = subset.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE].copy()
            blob = gzip.compress(pickle.dumps(chunk_df), compresslevel=6)
            size_mb = len(blob) / (1024 * 1024)

            db["telemetry_compressed"].insert_one({
                "filename": filename,
                "chunk": i,
                "data": blob,
                "rows": len(chunk_df),
            })
            print(f"  Chunk {i}/{n_chunks-1}: {len(chunk_df):,} rows, {size_mb:.1f} MB compressed")

        uploaded = db["telemetry_compressed"].count_documents({"filename": filename})
        print(f"  Verified: {uploaded} chunks")

    # ── Upload race summaries ──────────────────────────────────
    num_to_code = {}
    for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0}):
        num_to_code[str(doc["driver_number"])] = doc["name_acronym"]
    print(f"\nDriver map: {len(num_to_code)} entries")

    mapped = []
    for s in race_summaries:
        code = num_to_code.get(str(s["Driver"]))
        if not code:
            continue
        s["Driver"] = code
        s["_source_file"] = f"{YEAR}_{s['Race'].replace(' ', '_')}_Race.csv"
        mapped.append(s)

    print(f"Race summaries: {len(mapped)}")

    if mapped:
        ops = [UpdateOne(
            {"Year": s["Year"], "Race": s["Race"], "Driver": s["Driver"]},
            {"$set": s}, upsert=True
        ) for s in mapped]
        result = db["telemetry_race_summary"].bulk_write(ops, ordered=False)
        print(f"  Upserted: {result.upserted_count}, Modified: {result.modified_count}")

    # ── Verify ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERIFICATION")
    for st in SESSION_TYPES:
        fname = f"{YEAR}_{st}.parquet"
        chunks = db["telemetry_compressed"].count_documents({"filename": fname})
        print(f"  {fname}: {chunks} chunks")

    count = db["telemetry_race_summary"].count_documents({"Year": YEAR})
    print(f"  telemetry_race_summary (2025): {count} docs")

    test = db["telemetry_compressed"].find_one({"filename": f"{YEAR}_R.parquet", "chunk": 0})
    if test:
        test_df = pickle.loads(gzip.decompress(test["data"]))
        print(f"  Decompression test: {test_df.shape}")
    print("\nDone! Restart the server to see 2025 Race Detail data.")


if __name__ == "__main__":
    main()
