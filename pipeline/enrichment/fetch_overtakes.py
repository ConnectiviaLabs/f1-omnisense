"""
Overtake Data Fetcher
─────────────────────
Fetches per-race overtake events from OpenF1 /v1/overtakes endpoint.
Computes per-driver overtake stats and upserts to MongoDB.

Collections created:
  - openf1_overtakes        : raw overtake events per race
  - driver_overtake_profiles : aggregated per-driver overtake stats

Usage:
    python -m pipeline.enrichment.fetch_overtakes
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from pymongo import UpdateOne

# Add parent to path for _db import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from updater._db import get_db

BASE_URL = "https://api.openf1.org/v1"


def fetch_race_session_keys(db) -> list[dict]:
    """Get all race session keys from openf1_sessions."""
    return list(db["openf1_sessions"].find(
        {"session_type": "Race"},
        {"session_key": 1, "meeting_key": 1, "circuit_short_name": 1,
         "year": 1, "country_name": 1, "_id": 0},
    ).sort("session_key", 1))


def fetch_overtakes(session_key: int) -> list[dict]:
    """Fetch overtake events from OpenF1 for a single race session."""
    url = f"{BASE_URL}/overtakes"
    params = {"session_key": session_key}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    ⚠ Failed to fetch overtakes for session {session_key}: {e}")
        return []


def build_driver_number_map(db) -> dict[int, str]:
    """Map driver_number → name_acronym."""
    return {
        doc["driver_number"]: doc["name_acronym"]
        for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0})
    }


def ingest_overtakes(db) -> int:
    """Fetch and store all overtake events."""
    print("\n[1/2] Fetching overtake events from OpenF1...")

    sessions = fetch_race_session_keys(db)
    print(f"  Race sessions to process: {len(sessions)}")

    # Check which sessions we already have
    existing = set(db["openf1_overtakes"].distinct("session_key"))
    to_fetch = [s for s in sessions if s["session_key"] not in existing]
    print(f"  Already ingested: {len(existing)}, new to fetch: {len(to_fetch)}")

    total_inserted = 0
    now = datetime.now(timezone.utc)

    for i, session in enumerate(to_fetch):
        sk = session["session_key"]
        label = f"{session.get('year', '?')} {session.get('circuit_short_name', sk)}"

        events = fetch_overtakes(sk)
        if not events:
            print(f"  [{i+1}/{len(to_fetch)}] {label}: no overtake data")
            continue

        docs = []
        for evt in events:
            doc = {
                "session_key": sk,
                "meeting_key": evt.get("meeting_key", session.get("meeting_key")),
                "date": evt.get("date"),
                "overtaking_driver_number": evt.get("overtaking_driver_number"),
                "overtaken_driver_number": evt.get("overtaken_driver_number"),
                "position": evt.get("position"),
                "year": session.get("year"),
                "circuit": session.get("circuit_short_name"),
                "ingested_at": now,
            }
            docs.append(doc)

        db["openf1_overtakes"].insert_many(docs)
        total_inserted += len(docs)
        print(f"  [{i+1}/{len(to_fetch)}] {label}: {len(docs)} overtakes")

        # Respect rate limits (longer delay to avoid 429s)
        time.sleep(3)

    print(f"  ✅ Inserted {total_inserted} overtake events")
    return total_inserted


def compute_overtake_profiles(db) -> int:
    """Aggregate per-driver overtake stats from raw events."""
    print("\n[2/2] Computing driver overtake profiles...")

    num_to_code = build_driver_number_map(db)

    pipeline = [
        {"$facet": {
            "as_overtaker": [
                {"$group": {
                    "_id": "$overtaking_driver_number",
                    "total_overtakes": {"$sum": 1},
                    "races_with_overtakes": {"$addToSet": "$session_key"},
                }},
            ],
            "as_overtaken": [
                {"$group": {
                    "_id": "$overtaken_driver_number",
                    "total_times_overtaken": {"$sum": 1},
                    "races_overtaken_in": {"$addToSet": "$session_key"},
                }},
            ],
        }},
    ]

    result = list(db["openf1_overtakes"].aggregate(pipeline, allowDiskUse=True))
    if not result:
        print("  ⚠ No overtake data to aggregate")
        return 0

    # Build maps
    overtaker_map = {}
    for r in result[0]["as_overtaker"]:
        dn = r["_id"]
        overtaker_map[dn] = {
            "total_overtakes": r["total_overtakes"],
            "races_with_overtakes": len(r["races_with_overtakes"]),
        }

    overtaken_map = {}
    for r in result[0]["as_overtaken"]:
        dn = r["_id"]
        overtaken_map[dn] = {
            "total_times_overtaken": r["total_times_overtaken"],
            "races_overtaken_in": len(r["races_overtaken_in"]),
        }

    # Get total races per driver from openf1_position
    race_sessions = set(db["openf1_sessions"].distinct("session_key", {"session_type": "Race"}))
    driver_race_counts = {}
    for doc in db["openf1_position"].aggregate([
        {"$match": {"session_key": {"$in": list(race_sessions)}}},
        {"$group": {"_id": "$driver_number", "races": {"$addToSet": "$session_key"}}},
    ], allowDiskUse=True):
        driver_race_counts[doc["_id"]] = len(doc["races"])

    # Merge and compute rates
    all_drivers = set(overtaker_map.keys()) | set(overtaken_map.keys())
    ops = []
    now = datetime.now(timezone.utc)

    for dn in all_drivers:
        code = num_to_code.get(dn)
        if not code:
            continue

        ot = overtaker_map.get(dn, {"total_overtakes": 0, "races_with_overtakes": 0})
        od = overtaken_map.get(dn, {"total_times_overtaken": 0, "races_overtaken_in": 0})
        total_races = driver_race_counts.get(dn, 1)

        doc = {
            "driver_code": code,
            "driver_number": dn,
            "total_overtakes_made": ot["total_overtakes"],
            "total_times_overtaken": od["total_times_overtaken"],
            "overtake_net": ot["total_overtakes"] - od["total_times_overtaken"],
            "overtakes_per_race": round(ot["total_overtakes"] / max(total_races, 1), 2),
            "times_overtaken_per_race": round(od["total_times_overtaken"] / max(total_races, 1), 2),
            "overtake_ratio": round(
                ot["total_overtakes"] / max(od["total_times_overtaken"], 1), 3
            ),
            "races_analysed": total_races,
            "updated_at": now,
        }

        ops.append(UpdateOne(
            {"driver_code": code},
            {"$set": doc},
            upsert=True,
        ))

    if ops:
        db["driver_overtake_profiles"].create_index("driver_code", unique=True)
        result = db["driver_overtake_profiles"].bulk_write(ops, ordered=False)
        count = result.upserted_count + result.modified_count
        print(f"  ✅ Upserted {count} driver overtake profiles")
        return count

    return 0


def main():
    db = get_db()
    print("✅ Connected to MongoDB")

    ingest_overtakes(db)
    compute_overtake_profiles(db)

    # Verify
    total = db["openf1_overtakes"].count_documents({})
    profiles = db["driver_overtake_profiles"].count_documents({})
    print(f"\n  openf1_overtakes: {total:,} events")
    print(f"  driver_overtake_profiles: {profiles} drivers")

    sample = db["driver_overtake_profiles"].find_one(
        {"driver_code": "VER"},
        {"_id": 0, "updated_at": 0},
    )
    if sample:
        print(f"\n  Sample (VER): {sample}")


if __name__ == "__main__":
    main()
