"""Fetch full-grid race results from Jolpica/Ergast API and patch career stats.

Jolpica API: https://api.jolpi.ca/ergast/f1/
Mirror of the deprecated Ergast API. Free, no auth.
"""

from __future__ import annotations

import time
import logging
from collections import defaultdict
from datetime import datetime, timezone

import requests
from pymongo import UpdateOne
from pymongo.database import Database

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"


def _fetch_results(year: int) -> list[dict]:
    """Fetch all race results for a year from Jolpica API."""
    all_results = []
    offset = 0
    while True:
        url = f"{BASE_URL}/{year}/results/"
        resp = requests.get(url, params={"limit": 100, "offset": offset}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            break
        all_results.extend(races)
        total = int(data["MRData"]["total"])
        offset += 100
        if offset >= total:
            break
        time.sleep(0.3)
    return all_results


def sync(
    db: Database,
    years: list[int] | None = None,
) -> int:
    """Fetch race results and patch career stats into opponent_profiles.

    Args:
        db: MongoDB database
        years: Years to fetch. None = all years in opponent_profiles.seasons.

    Returns:
        Number of profiles patched.
    """
    print(f"\n{'='*60}")
    print("  Jolpica Career Stats Sync")
    print(f"{'='*60}")

    # Determine which driver_ids we care about
    profiles = list(db["opponent_profiles"].find({}, {"_id": 0, "driver_id": 1, "seasons": 1}))
    driver_ids = {p["driver_id"] for p in profiles}

    if years is None:
        # Get all seasons covered
        all_seasons = set()
        for p in profiles:
            all_seasons.update(p.get("seasons", []))
        years = sorted(all_seasons)

    print(f"  Drivers: {len(driver_ids)}, Years: {years[0]}–{years[-1]}")

    # Fetch and aggregate
    stats = defaultdict(lambda: {
        "wins": 0, "podiums": 0, "positions": [], "grids": [],
        "points": 0.0, "races": 0, "dnfs": 0,
    })

    for year in years:
        races = _fetch_results(year)
        for race in races:
            for res in race.get("Results", []):
                did = res["Driver"]["driverId"]
                if did not in driver_ids:
                    continue
                s = stats[did]
                s["races"] += 1
                s["points"] += float(res.get("points", 0))
                grid = int(res.get("grid", 0))
                s["grids"].append(grid)
                try:
                    pos = int(res.get("position", ""))
                    s["positions"].append(pos)
                    if pos == 1:
                        s["wins"] += 1
                    if pos <= 3:
                        s["podiums"] += 1
                except (ValueError, TypeError):
                    s["dnfs"] += 1
        print(f"  {year}: fetched")

    # Build upsert operations
    ops = []
    now = datetime.now(timezone.utc)
    for did, s in stats.items():
        avg_finish = round(sum(s["positions"]) / len(s["positions"]), 2) if s["positions"] else None
        avg_grid = round(sum(s["grids"]) / len(s["grids"]), 2) if s["grids"] else None
        avg_gained = round(avg_grid - avg_finish, 2) if avg_grid and avg_finish else None

        patch = {
            "total_wins": s["wins"],
            "total_podiums": s["podiums"],
            "total_races": s["races"],
            "avg_finish_position": avg_finish,
            "avg_grid_position": avg_grid,
            "avg_positions_gained": avg_gained,
            "avg_points_per_race": round(s["points"] / s["races"], 2) if s["races"] else 0,
            "dnf_rate": round(s["dnfs"] / s["races"], 4) if s["races"] else 0,
            "career_stats_updated_at": now,
        }
        ops.append(UpdateOne({"driver_id": did}, {"$set": patch}))

    if ops:
        result = db["opponent_profiles"].bulk_write(ops)
        patched = result.modified_count + result.upserted_count
    else:
        patched = 0

    print(f"\n  Patched {patched} profiles.")
    return patched
