"""Recompute opponent profiles after new data lands.

Uses the existing OpponentProfiler from pipeline.opponents
plus direct MongoDB aggregation for derived fields.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pymongo import UpdateOne
from pymongo.database import Database

logger = logging.getLogger(__name__)


def _compute_compound_profiles(db: Database) -> int:
    """Recompute opponent_compound_profiles from fastf1_laps."""
    pipeline = [
        {"$match": {"SessionType": "R", "Compound": {"$nin": [None, "", "UNKNOWN"]}}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Compound": "$Compound"},
            "total_laps": {"$sum": 1},
            "avg_tyre_life": {"$avg": "$TyreLife"},
            "avg_lap_time_s": {"$avg": {"$toDouble": "$LapTime"}},
            "std_lap_time_s": {"$stdDevPop": {"$toDouble": "$LapTime"}},
        }},
    ]

    results = list(db["fastf1_laps"].aggregate(pipeline, allowDiskUse=True))
    if not results:
        return 0

    now = datetime.now(timezone.utc)
    ops = []
    for r in results:
        doc = {
            "driver_id": r["_id"]["Driver"],
            "compound": r["_id"]["Compound"],
            "total_laps": r["total_laps"],
            "avg_tyre_life": r["avg_tyre_life"],
            "avg_lap_time_s": r["avg_lap_time_s"],
            "std_lap_time_s": r["std_lap_time_s"],
            "updated_at": now,
        }
        ops.append(UpdateOne(
            {"driver_id": doc["driver_id"], "compound": doc["compound"]},
            {"$set": doc},
            upsert=True,
        ))

    if ops:
        result = db["opponent_compound_profiles"].bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count
    return 0


def _compute_circuit_profiles(db: Database) -> int:
    """Recompute opponent_circuit_profiles from fastf1_laps."""
    pipeline = [
        {"$match": {"SessionType": "R"}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Race": "$Race"},
            "races": {"$addToSet": {"$concat": [{"$toString": "$Year"}, "-", "$Race"]}},
            "avg_finish_position": {"$avg": "$Position"},
            "avg_top_speed": {"$max": "$SpeedST"},
        }},
    ]

    results = list(db["fastf1_laps"].aggregate(pipeline, allowDiskUse=True))
    if not results:
        return 0

    now = datetime.now(timezone.utc)
    ops = []
    for r in results:
        doc = {
            "driver_id": r["_id"]["Driver"],
            "circuit": r["_id"]["Race"],
            "races": len(r.get("races", [])),
            "avg_finish_position": r["avg_finish_position"],
            "avg_top_speed": r["avg_top_speed"],
            "updated_at": now,
        }
        ops.append(UpdateOne(
            {"driver_id": doc["driver_id"], "circuit": doc["circuit"]},
            {"$set": doc},
            upsert=True,
        ))

    if ops:
        result = db["opponent_circuit_profiles"].bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count
    return 0


def refresh(db: Database) -> dict[str, int]:
    """Recompute all derived profile collections.

    Returns dict mapping collection name -> count of docs written.
    """
    print(f"\n{'='*60}")
    print("  Profile Refresh")
    print(f"{'='*60}")

    results = {}

    print("\n  Recomputing compound profiles...")
    results["opponent_compound_profiles"] = _compute_compound_profiles(db)
    print(f"    {results['opponent_compound_profiles']} docs")

    print("  Recomputing circuit profiles...")
    results["opponent_circuit_profiles"] = _compute_circuit_profiles(db)
    print(f"    {results['opponent_circuit_profiles']} docs")

    # Log
    db["pipeline_log"].insert_one({
        "chunk": "profile_refresh",
        "status": "complete",
        "results": results,
        "timestamp": datetime.now(timezone.utc),
    })

    print(f"\n  Refresh complete.")
    return results
