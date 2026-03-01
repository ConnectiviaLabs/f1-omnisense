"""
Populate Telemetry Collection
─────────────────────────────
Decompresses telemetry_compressed chunks and inserts into the `telemetry`
collection that chat_server.py endpoints depend on.

Transforms:
  - Driver number → 3-letter code (e.g. "44" → "HAM")
  - LapTime_s → LapTime (timedelta string)
  - Adds _source_file field (e.g. "2024_Abu_Dhabi_Grand_Prix_Race.csv")
  - Only race sessions (Session == "R" or filename contains "_R.")

Usage:
    python -m pipeline.enrichment.populate_telemetry
"""

from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path

import pandas as pd
from pymongo import InsertOne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from updater._db import get_db

BATCH_SIZE = 50_000  # docs per bulk insert


def build_driver_map(db) -> dict[str, str]:
    """Build driver number → 3-letter code mapping."""
    num_to_code = {}
    for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0}):
        num_to_code[str(doc["driver_number"])] = doc["name_acronym"]
    return num_to_code


def make_source_file(year: int, race: str) -> str:
    """Create _source_file string matching chat_server's expected format."""
    # "Australian Grand Prix" → "2018_Australian_Grand_Prix_Race.csv"
    race_clean = race.replace(" ", "_")
    return f"{year}_{race_clean}_Race.csv"


def make_laptime_str(seconds: float | None) -> str | None:
    """Convert lap time in seconds to timedelta-like string."""
    if seconds is None or pd.isna(seconds):
        return None
    m, s = divmod(seconds, 60)
    return f"0 days 00:{int(m):02d}:{s:06.3f}"


def main():
    db = get_db()
    print("✅ Connected to MongoDB")

    num_to_code = build_driver_map(db)
    print(f"  Driver map: {len(num_to_code)} entries")

    # Check existing count
    existing = db["telemetry"].estimated_document_count()
    if existing > 0:
        print(f"  ⚠ telemetry collection already has {existing:,} docs")
        resp = input("  Drop and re-populate? [y/N] ")
        if resp.lower() != "y":
            print("  Aborted.")
            return
        db["telemetry"].drop()
        print("  Dropped telemetry collection")

    # Only race files
    filenames = sorted(db["telemetry_compressed"].distinct("filename"))
    race_files = [f for f in filenames if "_R." in f]
    print(f"\n  Race files to process: {len(race_files)}")

    total_inserted = 0

    for fi, fname in enumerate(race_files):
        chunks = list(db["telemetry_compressed"].find(
            {"filename": fname},
            {"data": 1, "chunk": 1, "_id": 0},
        ).sort("chunk", 1))

        frames = []
        for doc in chunks:
            try:
                df = pickle.loads(gzip.decompress(doc["data"]))
                frames.append(df)
            except Exception:
                pass

        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True)

        # Map driver numbers to codes
        df["Driver"] = df["Driver"].astype(str).map(num_to_code)
        df = df.dropna(subset=["Driver"])

        if df.empty:
            continue

        # Extract year and race for _source_file
        year = int(df["Year"].iloc[0]) if "Year" in df.columns else 0
        race = str(df["Race"].iloc[0]) if "Race" in df.columns else "Unknown"
        source_file = make_source_file(year, race)
        df["_source_file"] = source_file

        # Convert LapTime_s → LapTime
        if "LapTime_s" in df.columns:
            df["LapTime"] = df["LapTime_s"].apply(make_laptime_str)
            df = df.drop(columns=["LapTime_s"])

        # Drop columns chat_server doesn't need (keep all useful ones)
        # Convert to records
        records = df.to_dict("records")

        # Batch insert
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            ops = [InsertOne(r) for r in batch]
            db["telemetry"].bulk_write(ops, ordered=False)

        total_inserted += len(records)
        if (fi + 1) % 5 == 0 or fi == 0:
            print(f"  [{fi + 1}/{len(race_files)}] {fname}: {len(records):,} rows → telemetry  (total: {total_inserted:,})")

    # Create indexes for chat_server query patterns
    print("\n  Creating indexes...")
    db["telemetry"].create_index("_source_file")
    db["telemetry"].create_index([("Driver", 1), ("Year", 1)])
    db["telemetry"].create_index([("Driver", 1), ("Year", 1), ("Race", 1)])
    db["telemetry"].create_index([("Year", 1), ("Race", 1), ("LapNumber", 1)])

    total = db["telemetry"].estimated_document_count()
    print(f"\n  ✅ Inserted {total_inserted:,} docs into telemetry collection")
    print(f"  telemetry: {total:,} total docs")

    # Sample
    sample = db["telemetry"].find_one({"Driver": "VER"}, {"_id": 0})
    if sample:
        print(f"\n  Sample (VER):")
        for k in sorted(sample.keys()):
            print(f"    {k:25s}  {sample[k]}")


if __name__ == "__main__":
    main()
