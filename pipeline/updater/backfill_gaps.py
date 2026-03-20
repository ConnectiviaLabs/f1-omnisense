"""Backfill data gaps across all time-series collections.

Identified gaps:
  1. openf1_car_data     — only 2 sessions ingested (needs 2023-2026)
  2. openf1_location     — only 2 sessions ingested (needs 2023-2026)
  3. openf1_pit          — 2023 missing ~7 races
  4. race_air_density    — gaps in some years

Strategy:
  - car_data: Load from local .ff1pkl files first (20 races), then OpenF1 API
  - location: OpenF1 API only (no local source)
  - pit: OpenF1 API
  - air_density: Open-Meteo API

Usage:
    python -m pipeline.updater.backfill_gaps                    # backfill all gaps
    python -m pipeline.updater.backfill_gaps --target openf1    # only OpenF1 gaps
    python -m pipeline.updater.backfill_gaps --target air       # only air density
    python -m pipeline.updater.backfill_gaps --dry-run          # show what would be fetched
"""

from __future__ import annotations

import argparse
import glob
import pickle
import re
import time
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from pymongo import UpdateOne
from pymongo.database import Database

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from updater._db import get_db
from updater.openf1_fetcher import _api_get, _bulk_upsert, BASE_URL, REQUEST_DELAY

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CAR_DATA = PROJECT_ROOT / "data" / "Car" / "mclaren_data"

# ─────────────────────────────────────────────────────────────
# OpenF1 high-frequency endpoints (car_data + location)
# ─────────────────────────────────────────────────────────────

HF_ENDPOINTS = [
    ("car_data", "openf1_car_data", ["session_key", "driver_number", "date"]),
    ("location", "openf1_location", ["session_key", "driver_number", "date"]),
]

PIT_ENDPOINT = ("pit", "openf1_pit", ["session_key", "driver_number", "lap_number"])

# Map local race folder names to openf1 circuit_short_name
RACE_NAME_TO_CIRCUIT = {
    "Abu_Dhabi_Grand_Prix": "Yas Marina Circuit",
    "Australian_Grand_Prix": "Melbourne",
    "Austrian_Grand_Prix": "Spielberg",
    "Azerbaijan_Grand_Prix": "Baku",
    "Bahrain_Grand_Prix": "Sakhir",
    "Belgian_Grand_Prix": "Spa-Francorchamps",
    "British_Grand_Prix": "Silverstone",
    "Canadian_Grand_Prix": "Montreal",
    "Dutch_Grand_Prix": "Zandvoort",
    "Emilia_Romagna_Grand_Prix": "Imola",
    "Hungarian_Grand_Prix": "Hungaroring",
    "Italian_Grand_Prix": "Monza",
    "Japanese_Grand_Prix": "Suzuka",
    "Las_Vegas_Grand_Prix": "Las Vegas",
    "Mexico_City_Grand_Prix": "Mexico City",
    "Miami_Grand_Prix": "Miami",
    "Monaco_Grand_Prix": "Monte Carlo",
    "Saudi_Arabian_Grand_Prix": "Jeddah",
    "Singapore_Grand_Prix": "Singapore",
    "Spanish_Grand_Prix": "Catalunya",
}


def _get_sessions_to_backfill(db: Database, collection: str, years: list[int]) -> list[dict]:
    """Find sessions that have no data in the target collection."""
    pipeline = [
        {"$match": {"year": {"$in": years}, "session_type": "Race"}},
        {"$group": {
            "_id": "$session_key",
            "session_key": {"$first": "$session_key"},
            "year": {"$first": "$year"},
            "circuit_short_name": {"$first": "$circuit_short_name"},
            "session_name": {"$first": "$session_name"},
            "meeting_key": {"$first": "$meeting_key"},
        }},
        {"$sort": {"year": 1, "circuit_short_name": 1}},
    ]
    all_sessions = list(db["openf1_sessions"].aggregate(pipeline))
    existing_keys = set(db[collection].distinct("session_key"))
    return [s for s in all_sessions if s["session_key"] not in existing_keys]


# ─────────────────────────────────────────────────────────────
# Local pickle loader for car_data
# ─────────────────────────────────────────────────────────────

def _find_local_pickles() -> dict[tuple[int, str], Path]:
    """Find all local car_data.ff1pkl files, keyed by (year, circuit_short_name)."""
    found = {}
    pattern = str(LOCAL_CAR_DATA / "**/car_data.ff1pkl")
    for pkl_path in glob.glob(pattern, recursive=True):
        # Path like: .../2023/2023-10-29_Mexico_City_Grand_Prix/2023-10-29_Race/car_data.ff1pkl
        parts = Path(pkl_path).parts
        # Find year from directory name
        for part in parts:
            m = re.match(r"^(\d{4})$", part)
            if m:
                year = int(m.group(1))
                break
        else:
            continue

        # Find race name from directory like "2023-10-29_Mexico_City_Grand_Prix"
        for part in parts:
            m = re.match(r"^\d{4}-\d{2}-\d{2}_(.+)$", part)
            if m:
                race_name = m.group(1)
                circuit = RACE_NAME_TO_CIRCUIT.get(race_name)
                if circuit:
                    found[(year, circuit)] = Path(pkl_path)
                break

    return found


def _load_pickle_to_docs(pkl_path: Path, session_key: int, meeting_key: int) -> list[dict]:
    """Load a car_data.ff1pkl file and convert to MongoDB docs matching OpenF1 schema."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    data = obj.get("data", {})
    if not isinstance(data, dict):
        return []

    now = datetime.now(timezone.utc)
    docs = []

    for driver_number_str, df in data.items():
        driver_number = int(driver_number_str)
        for _, row in df.iterrows():
            date_val = row.get("Date")
            if date_val is None:
                continue

            doc = {
                "date": str(date_val),
                "session_key": session_key,
                "driver_number": driver_number,
                "meeting_key": meeting_key,
                "speed": int(row.get("Speed", 0)),
                "rpm": int(row.get("RPM", 0)),
                "n_gear": int(row.get("nGear", 0)),
                "throttle": int(row.get("Throttle", 0)),
                "brake": int(row.get("Brake", 0)) if isinstance(row.get("Brake"), (int, float)) else (100 if row.get("Brake") else 0),
                "drs": int(row.get("DRS", 0)),
                "ingested_at": now,
            }
            docs.append(doc)

    return docs


def backfill_openf1_hf(db: Database, dry_run: bool = False):
    """Backfill openf1_car_data and openf1_location for 2023-2026 Race sessions."""
    years = [2023, 2024, 2025, 2026]
    local_pickles = _find_local_pickles()

    if local_pickles:
        print(f"\n  Found {len(local_pickles)} local car_data pickle files")
        for (y, c), p in sorted(local_pickles.items()):
            print(f"    {y} {c}")

    for endpoint, collection, key_fields in HF_ENDPOINTS:
        print(f"\n{'─'*60}")
        print(f"  Backfilling {collection}")
        print(f"{'─'*60}")

        missing = _get_sessions_to_backfill(db, collection, years)
        print(f"  Missing sessions: {len(missing)}")

        if dry_run:
            for s in missing:
                year = s["year"]
                circuit = s.get("circuit_short_name", "?")
                source = "LOCAL" if (endpoint == "car_data" and (year, circuit) in local_pickles) else "API"
                print(f"    [{source}] {year} {circuit} (sk={s['session_key']})")
            continue

        # Create indexes
        db[collection].create_index([("meeting_key", 1), ("session_key", 1), ("driver_number", 1)])
        db[collection].create_index([("session_key", 1), ("driver_number", 1), ("date", 1)])

        for idx, session in enumerate(missing, 1):
            sk = session["session_key"]
            mk = session.get("meeting_key", 0)
            circuit = session.get("circuit_short_name", "?")
            year = session["year"]

            # Try local pickle first (car_data only)
            if endpoint == "car_data" and (year, circuit) in local_pickles:
                pkl_path = local_pickles[(year, circuit)]
                print(f"\n  [{idx}/{len(missing)}] {year} {circuit} (sk={sk}) — LOCAL")
                docs = _load_pickle_to_docs(pkl_path, sk, mk)
                if docs:
                    count = _bulk_upsert(db, collection, docs, key_fields)
                    print(f"    Loaded {len(docs)} records from pickle, {count} upserted")
                    continue
                else:
                    print(f"    Pickle empty, falling back to API")

            # Fetch from OpenF1 API
            print(f"\n  [{idx}/{len(missing)}] {year} {circuit} (sk={sk}) — API")

            drivers = db["openf1_drivers"].distinct("driver_number", {"session_key": sk})
            if not drivers:
                driver_docs = _api_get("drivers", {"session_key": sk})
                drivers = list({d["driver_number"] for d in driver_docs if "driver_number" in d})
                time.sleep(REQUEST_DELAY)

            if not drivers:
                print(f"    No drivers found, skipping")
                continue

            print(f"    Drivers: {len(drivers)}")
            session_total = 0

            for drv in sorted(drivers):
                time.sleep(REQUEST_DELAY)
                docs = _api_get(endpoint, {"session_key": sk, "driver_number": drv})
                if docs:
                    count = _bulk_upsert(db, collection, docs, key_fields)
                    session_total += count
                    print(f"    Driver {drv}: {len(docs)} fetched, {count} upserted")
                else:
                    print(f"    Driver {drv}: 0 (no data)")

            print(f"    Session total: {session_total}")


def backfill_openf1_pit(db: Database, dry_run: bool = False):
    """Backfill missing openf1_pit data for 2023."""
    print(f"\n{'─'*60}")
    print(f"  Backfilling openf1_pit (2023 gaps)")
    print(f"{'─'*60}")

    endpoint, collection, key_fields = PIT_ENDPOINT
    missing = _get_sessions_to_backfill(db, collection, [2023])
    print(f"  Missing 2023 sessions: {len(missing)}")

    if dry_run:
        for s in missing:
            print(f"    Would fetch: {s.get('circuit_short_name')} (sk={s['session_key']})")
        return

    for idx, session in enumerate(missing, 1):
        sk = session["session_key"]
        circuit = session.get("circuit_short_name", "?")
        print(f"  [{idx}/{len(missing)}] 2023 {circuit} (sk={sk})")

        time.sleep(REQUEST_DELAY)
        docs = _api_get(endpoint, {"session_key": sk})
        if docs:
            count = _bulk_upsert(db, collection, docs, key_fields)
            print(f"    {len(docs)} fetched, {count} upserted")
        else:
            print(f"    0 (no data)")


# ─────────────────────────────────────────────────────────────
# Air density backfill
# ─────────────────────────────────────────────────────────────

def backfill_air_density(db: Database, dry_run: bool = False):
    """Backfill missing race_air_density records."""
    print(f"\n{'─'*60}")
    print(f"  Backfilling race_air_density gaps")
    print(f"{'─'*60}")

    from enrichment.fetch_air_density import (
        RACE_TO_SLUG, CIRCUIT_COORDS,
        fetch_race_day_weather, get_elevation,
    )

    existing = set()
    for doc in db["race_air_density"].find({}, {"year": 1, "race": 1, "_id": 0}):
        existing.add((doc.get("year"), doc.get("race")))

    all_combos = set()
    for doc in db["fastf1_laps"].aggregate([
        {"$match": {"SessionType": "R"}},
        {"$group": {"_id": {"Year": "$Year", "Race": "$Race"}}},
    ]):
        all_combos.add((doc["_id"]["Year"], doc["_id"]["Race"]))

    missing = [(y, r) for y, r in all_combos if (y, r) not in existing and r in RACE_TO_SLUG]
    missing.sort()

    print(f"  Existing: {len(existing)}, Missing: {len(missing)}")

    if dry_run:
        for y, r in missing:
            print(f"    Would fetch: {y} {r}")
        return

    ops = []
    now = datetime.now(timezone.utc)
    elevations = {}

    for idx, (year, race) in enumerate(missing, 1):
        slug = RACE_TO_SLUG[race]
        if slug not in CIRCUIT_COORDS:
            continue

        lat, lon = CIRCUIT_COORDS[slug]

        jolpica_doc = db["jolpica_race_results"].find_one(
            {"season": year, "race_name": race},
            {"date": 1, "_id": 0},
        )
        if not jolpica_doc or not jolpica_doc.get("date"):
            print(f"  [{idx}/{len(missing)}] {year} {race} — no date found, skipping")
            continue

        date_str = jolpica_doc["date"][:10]

        if slug not in elevations:
            elevations[slug] = get_elevation(lat, lon)
            time.sleep(0.2)

        weather = fetch_race_day_weather(lat, lon, date_str)
        if not weather:
            print(f"  [{idx}/{len(missing)}] {year} {race} — no weather data")
            continue

        doc = {
            "year": year,
            "race": race,
            "circuit_slug": slug,
            "race_date": date_str,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elevations.get(slug),
            **weather,
            "ingested_at": now,
        }
        ops.append(UpdateOne({"year": year, "race": race}, {"$set": doc}, upsert=True))
        print(f"  [{idx}/{len(missing)}] {year} {race} — {weather['air_density_kg_m3']} kg/m3")
        time.sleep(0.3)

    if ops:
        result = db["race_air_density"].bulk_write(ops, ordered=False)
        print(f"\n  Upserted {result.upserted_count + result.modified_count} air density records")
    else:
        print("\n  No new records to insert")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backfill data gaps")
    parser.add_argument("--target", choices=["openf1", "air", "all"],
                        default="all", help="Which gaps to fill")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without fetching")
    args = parser.parse_args()

    db = get_db()
    print(f"Connected to MongoDB ({db.name})")

    if args.target in ("openf1", "all"):
        backfill_openf1_hf(db, dry_run=args.dry_run)
        backfill_openf1_pit(db, dry_run=args.dry_run)

    if args.target in ("air", "all"):
        backfill_air_density(db, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("  Backfill complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
