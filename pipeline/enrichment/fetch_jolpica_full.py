"""
Jolpica Full-Grid Fetcher
─────────────────────────
Fetches complete race results, qualifying, pit stops, sprint results,
and constructor standings from the Jolpica (Ergast) API for ALL drivers
and teams (2018-2025).

Collections created/updated:
  - jolpica_race_results       : per-driver per-race finish data
  - jolpica_qualifying         : per-driver Q1/Q2/Q3 times
  - jolpica_pit_stops          : per-driver per-race pit stop events
  - jolpica_sprint_results     : per-driver sprint finishes
  - jolpica_constructor_standings : end-of-season team rankings

Derived enrichments:
  - Patches opponent_profiles with quali-race delta, constructor-adjusted perf
  - Expands circuit_pit_loss_times with historical pit data

Usage:
    python -m pipeline.enrichment.fetch_jolpica_full
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from updater._db import get_db

BASE_URL = "https://api.jolpi.ca/ergast/f1"
YEARS = list(range(2018, 2026))


# ── API helpers ──────────────────────────────────────────────────────────

def _api_get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET with retry on 429."""
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"    ⏳ Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


def _paginate(url: str, limit: int = 100) -> list[dict]:
    """Paginate through Jolpica API results."""
    all_races = []
    offset = 0
    while True:
        data = _api_get(url, params={"limit": limit, "offset": offset})
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            break
        all_races.extend(races)
        total = int(data["MRData"]["total"])
        offset += limit
        if offset >= total:
            break
        time.sleep(1.5)
    return all_races


# ── 1. Race Results ─────────────────────────────────────────────────────

def ingest_race_results(db) -> int:
    print("\n[1/5] Fetching full-grid race results...")

    existing_keys = set()
    for doc in db["jolpica_race_results"].find({}, {"season": 1, "round": 1, "_id": 0}):
        existing_keys.add((doc["season"], doc["round"]))

    total = 0
    now = datetime.now(timezone.utc)

    for year in YEARS:
        races = _paginate(f"{BASE_URL}/{year}/results/")
        if not races:
            continue

        docs = []
        for race in races:
            season = int(race["season"])
            rnd = int(race["round"])
            if (season, rnd) in existing_keys:
                continue

            circuit_id = race["Circuit"]["circuitId"]
            race_name = race["raceName"]

            for res in race.get("Results", []):
                driver = res["Driver"]
                constructor = res["Constructor"]

                try:
                    pos = int(res["position"])
                except (ValueError, TypeError, KeyError):
                    pos = None

                try:
                    grid = int(res.get("grid", 0))
                except (ValueError, TypeError):
                    grid = None

                fl = res.get("FastestLap", {})

                doc = {
                    "season": season,
                    "round": rnd,
                    "race_name": race_name,
                    "circuit_id": circuit_id,
                    "date": race.get("date"),
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code", ""),
                    "driver_name": f"{driver['givenName']} {driver['familyName']}",
                    "constructor_id": constructor["constructorId"],
                    "constructor_name": constructor["name"],
                    "grid": grid,
                    "position": pos,
                    "position_text": res.get("positionText", ""),
                    "points": float(res.get("points", 0)),
                    "laps": int(res.get("laps", 0)),
                    "status": res.get("status", ""),
                    "fastest_lap_rank": fl.get("rank"),
                    "fastest_lap_time": fl.get("Time", {}).get("time"),
                    "fastest_lap_speed_kph": fl.get("AverageSpeed", {}).get("speed"),
                    "positions_gained": (grid - pos) if grid and pos else None,
                    "ingested_at": now,
                }
                docs.append(doc)

        if docs:
            db["jolpica_race_results"].insert_many(docs)
            total += len(docs)
        print(f"  {year}: {len(docs)} results")
        time.sleep(0.5)

    db["jolpica_race_results"].create_index([("season", 1), ("round", 1), ("driver_id", 1)], unique=True)
    db["jolpica_race_results"].create_index("driver_id")
    db["jolpica_race_results"].create_index("circuit_id")
    db["jolpica_race_results"].create_index("constructor_id")
    print(f"  Inserted {total} race result records")
    return total


# ── 2. Qualifying ───────────────────────────────────────────────────────

def ingest_qualifying(db) -> int:
    print("\n[2/5] Fetching qualifying results...")

    existing_keys = set()
    for doc in db["jolpica_qualifying"].find({}, {"season": 1, "round": 1, "_id": 0}):
        existing_keys.add((doc["season"], doc["round"]))

    total = 0
    now = datetime.now(timezone.utc)

    for year in YEARS:
        races = _paginate(f"{BASE_URL}/{year}/qualifying/")
        if not races:
            continue

        docs = []
        for race in races:
            season = int(race["season"])
            rnd = int(race["round"])
            if (season, rnd) in existing_keys:
                continue

            circuit_id = race["Circuit"]["circuitId"]

            for res in race.get("QualifyingResults", []):
                driver = res["Driver"]
                constructor = res["Constructor"]

                doc = {
                    "season": season,
                    "round": rnd,
                    "race_name": race["raceName"],
                    "circuit_id": circuit_id,
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code", ""),
                    "constructor_id": constructor["constructorId"],
                    "position": int(res.get("position", 0)),
                    "q1": res.get("Q1"),
                    "q2": res.get("Q2"),
                    "q3": res.get("Q3"),
                    "ingested_at": now,
                }
                docs.append(doc)

        if docs:
            db["jolpica_qualifying"].insert_many(docs)
            total += len(docs)
        print(f"  {year}: {len(docs)} qualifying entries")
        time.sleep(0.5)

    db["jolpica_qualifying"].create_index([("season", 1), ("round", 1), ("driver_id", 1)], unique=True)
    db["jolpica_qualifying"].create_index("driver_id")
    print(f"  Inserted {total} qualifying records")
    return total


# ── 3. Pit Stops ────────────────────────────────────────────────────────

def ingest_pit_stops(db) -> int:
    print("\n[3/5] Fetching pit stop data...")

    existing_keys = set()
    for doc in db["jolpica_pit_stops"].find({}, {"season": 1, "round": 1, "_id": 0}):
        existing_keys.add((doc["season"], doc["round"]))

    # First get all race rounds per year
    total = 0
    now = datetime.now(timezone.utc)

    # Get all race rounds from already-ingested race results
    round_pipeline = [
        {"$group": {"_id": {"season": "$season", "round": "$round"}}},
        {"$sort": {"_id.season": 1, "_id.round": 1}},
    ]
    all_rounds = list(db["jolpica_race_results"].aggregate(round_pipeline))
    rounds_by_year = defaultdict(list)
    for r in all_rounds:
        rounds_by_year[r["_id"]["season"]].append(r["_id"]["round"])

    for year in YEARS:
        rounds = sorted(rounds_by_year.get(year, []))
        if not rounds:
            continue

        year_docs = []
        for rnd in rounds:
            if (year, rnd) in existing_keys:
                continue

            try:
                data = _api_get(f"{BASE_URL}/{year}/{rnd}/pitstops/", params={"limit": 100})
                pit_races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            except Exception as e:
                print(f"    ⚠ {year} R{rnd}: {e}")
                continue

            if not pit_races:
                continue

            race = pit_races[0]
            circuit_id = race["Circuit"]["circuitId"]
            race_name = race["raceName"]

            for pit in race.get("PitStops", []):
                try:
                    duration = float(pit["duration"])
                except (ValueError, TypeError, KeyError):
                    duration = None

                doc = {
                    "season": year,
                    "round": rnd,
                    "race_name": race_name,
                    "circuit_id": circuit_id,
                    "driver_id": pit["driverId"],
                    "stop": int(pit.get("stop", 0)),
                    "lap": int(pit.get("lap", 0)),
                    "duration_s": duration,
                    "time_of_day": pit.get("time"),
                    "ingested_at": now,
                }
                year_docs.append(doc)

            time.sleep(1.5)

        if year_docs:
            db["jolpica_pit_stops"].insert_many(year_docs)
            total += len(year_docs)
        print(f"  {year}: {len(year_docs)} pit stops across {len(rounds)} races")

    db["jolpica_pit_stops"].create_index([("season", 1), ("round", 1), ("driver_id", 1), ("stop", 1)])
    db["jolpica_pit_stops"].create_index("circuit_id")
    db["jolpica_pit_stops"].create_index("driver_id")
    print(f"  Inserted {total} pit stop records")
    return total


# ── 4. Sprint Results ───────────────────────────────────────────────────

def ingest_sprints(db) -> int:
    print("\n[4/5] Fetching sprint results...")

    existing_keys = set()
    for doc in db["jolpica_sprint_results"].find({}, {"season": 1, "round": 1, "_id": 0}):
        existing_keys.add((doc["season"], doc["round"]))

    total = 0
    now = datetime.now(timezone.utc)

    for year in YEARS:
        if year < 2021:
            continue  # Sprints started in 2021

        races = _paginate(f"{BASE_URL}/{year}/sprint/")
        if not races:
            print(f"  {year}: no sprint data")
            continue

        docs = []
        for race in races:
            season = int(race["season"])
            rnd = int(race["round"])
            if (season, rnd) in existing_keys:
                continue

            circuit_id = race["Circuit"]["circuitId"]

            for res in race.get("SprintResults", []):
                driver = res["Driver"]
                constructor = res["Constructor"]

                try:
                    pos = int(res["position"])
                except (ValueError, TypeError, KeyError):
                    pos = None
                try:
                    grid = int(res.get("grid", 0))
                except (ValueError, TypeError):
                    grid = None

                doc = {
                    "season": season,
                    "round": rnd,
                    "race_name": race["raceName"],
                    "circuit_id": circuit_id,
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code", ""),
                    "constructor_id": constructor["constructorId"],
                    "constructor_name": constructor["name"],
                    "grid": grid,
                    "position": pos,
                    "points": float(res.get("points", 0)),
                    "laps": int(res.get("laps", 0)),
                    "status": res.get("status", ""),
                    "positions_gained": (grid - pos) if grid and pos else None,
                    "ingested_at": now,
                }
                docs.append(doc)

        if docs:
            db["jolpica_sprint_results"].insert_many(docs)
            total += len(docs)
        print(f"  {year}: {len(docs)} sprint entries")
        time.sleep(0.5)

    db["jolpica_sprint_results"].create_index([("season", 1), ("round", 1), ("driver_id", 1)], unique=True)
    db["jolpica_sprint_results"].create_index("driver_id")
    print(f"  Inserted {total} sprint records")
    return total


# ── 5. Constructor Standings ────────────────────────────────────────────

def ingest_constructor_standings(db) -> int:
    print("\n[5/5] Fetching constructor standings...")

    existing_years = set(db["jolpica_constructor_standings"].distinct("season"))
    total = 0
    now = datetime.now(timezone.utc)

    for year in YEARS:
        if year in existing_years:
            continue

        try:
            resp = requests.get(
                f"{BASE_URL}/{year}/constructorStandings/",
                params={"limit": 100},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            standings_lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        except Exception as e:
            print(f"  {year}: ⚠ {e}")
            continue

        if not standings_lists:
            continue

        docs = []
        for sl in standings_lists:
            for cs in sl.get("ConstructorStandings", []):
                doc = {
                    "season": year,
                    "position": int(cs.get("position", 0)),
                    "points": float(cs.get("points", 0)),
                    "wins": int(cs.get("wins", 0)),
                    "constructor_id": cs["Constructor"]["constructorId"],
                    "constructor_name": cs["Constructor"]["name"],
                    "nationality": cs["Constructor"]["nationality"],
                    "ingested_at": now,
                }
                docs.append(doc)

        if docs:
            db["jolpica_constructor_standings"].insert_many(docs)
            total += len(docs)
        print(f"  {year}: {len(docs)} constructor standings")
        time.sleep(0.5)

    db["jolpica_constructor_standings"].create_index([("season", 1), ("constructor_id", 1)], unique=True)
    print(f"  Inserted {total} constructor standing records")
    return total


def ingest_driver_standings(db) -> int:
    print("\n[6/6] Fetching driver standings...")

    existing_years = set(db["jolpica_driver_standings"].distinct("season"))
    total = 0
    now = datetime.now(timezone.utc)

    for year in YEARS:
        if year in existing_years:
            continue

        try:
            resp = requests.get(
                f"{BASE_URL}/{year}/driverStandings/",
                params={"limit": 100},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            standings_lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        except Exception as e:
            print(f"  {year}: ⚠ {e}")
            continue

        if not standings_lists:
            continue

        docs = []
        for sl in standings_lists:
            for ds in sl.get("DriverStandings", []):
                driver = ds.get("Driver", {})
                constructors = ds.get("Constructors", [])
                team = constructors[0] if constructors else {}
                doc = {
                    "season": year,
                    "position": int(ds.get("position", 0)),
                    "points": float(ds.get("points", 0)),
                    "wins": int(ds.get("wins", 0)),
                    "driver_id": driver.get("driverId", ""),
                    "driver_code": driver.get("code", ""),
                    "driver_name": f'{driver.get("givenName", "")} {driver.get("familyName", "")}',
                    "nationality": driver.get("nationality", ""),
                    "constructor_id": team.get("constructorId", ""),
                    "constructor_name": team.get("name", ""),
                    "ingested_at": now,
                }
                docs.append(doc)

        if docs:
            db["jolpica_driver_standings"].insert_many(docs)
            total += len(docs)
        print(f"  {year}: {len(docs)} driver standings")
        time.sleep(0.5)

    db["jolpica_driver_standings"].create_index([("season", 1), ("driver_id", 1)], unique=True)
    db["jolpica_driver_standings"].create_index("driver_code")
    print(f"  Inserted {total} driver standing records")
    return total


# ── Derived Metrics ─────────────────────────────────────────────────────

def _parse_laptime_s(t: str | None) -> float | None:
    """Convert '1:23.456' to seconds."""
    if not t:
        return None
    try:
        parts = t.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def compute_derived_metrics(db):
    """Compute and patch derived metrics into opponent_profiles."""
    print("\n[Derived] Computing enriched metrics...")

    # Get all driver_ids in opponent_profiles
    profiles = {
        p["driver_id"]: p
        for p in db["opponent_profiles"].find({}, {"_id": 0, "driver_id": 1, "seasons": 1})
    }
    driver_ids = set(profiles.keys())

    # ── 1. Quali vs Race pace delta ──────────────────────────────────
    print("  Computing quali-race deltas...")
    quali_docs = list(db["jolpica_qualifying"].find(
        {"driver_id": {"$in": list(driver_ids)}},
        {"_id": 0, "season": 1, "round": 1, "driver_id": 1, "q1": 1, "q2": 1, "q3": 1},
    ))
    race_docs = list(db["jolpica_race_results"].find(
        {"driver_id": {"$in": list(driver_ids)}},
        {"_id": 0, "season": 1, "round": 1, "driver_id": 1, "fastest_lap_time": 1,
         "position": 1, "grid": 1, "constructor_id": 1, "points": 1, "status": 1},
    ))

    # Index race docs by (season, round, driver_id)
    race_index = {}
    for rd in race_docs:
        key = (rd["season"], rd["round"], rd["driver_id"])
        race_index[key] = rd

    # Per-driver quali-race delta
    driver_deltas = defaultdict(list)
    driver_q_positions = defaultdict(list)

    for qd in quali_docs:
        did = qd["driver_id"]
        best_q = None
        for q_field in ["q3", "q2", "q1"]:
            t = _parse_laptime_s(qd.get(q_field))
            if t:
                best_q = t
                break

        key = (qd["season"], qd["round"], did)
        rd = race_index.get(key)
        if rd and best_q:
            fl = _parse_laptime_s(rd.get("fastest_lap_time"))
            if fl and fl > 0 and best_q > 0:
                delta_pct = ((fl - best_q) / best_q) * 100
                driver_deltas[did].append(delta_pct)

    # ── 2. Constructor-adjusted performance ──────────────────────────
    print("  Computing constructor-adjusted performance...")
    # Get constructor position per season
    constructor_rank = {}
    for doc in db["jolpica_constructor_standings"].find(
        {}, {"season": 1, "constructor_id": 1, "position": 1, "_id": 0}
    ):
        constructor_rank[(doc["season"], doc["constructor_id"])] = doc["position"]

    # Per-driver: average (finish_pos - constructor_rank) = how much better/worse than car
    driver_car_adj = defaultdict(list)
    driver_grid_vs_finish = defaultdict(list)
    driver_points = defaultdict(lambda: {"points": 0, "races": 0, "dnfs": 0})
    driver_status = defaultdict(list)

    for rd in race_docs:
        did = rd["driver_id"]
        pos = rd.get("position")
        grid = rd.get("grid")
        cid = rd.get("constructor_id")
        season = rd["season"]

        if pos and cid:
            car_rank = constructor_rank.get((season, cid))
            if car_rank:
                # Negative = outperforming car, positive = underperforming
                driver_car_adj[did].append(pos - car_rank)

        if grid and pos:
            driver_grid_vs_finish[did].append(grid - pos)

        dp = driver_points[did]
        dp["races"] += 1
        dp["points"] += rd.get("points", 0)
        status = rd.get("status", "")
        if status and status not in ("Finished", "+1 Lap", "+2 Laps", "+3 Laps"):
            dp["dnfs"] += 1

    # ── 3. Sprint stats ─────────────────────────────────────────────
    print("  Computing sprint performance...")
    sprint_docs = list(db["jolpica_sprint_results"].find(
        {"driver_id": {"$in": list(driver_ids)}},
        {"_id": 0, "driver_id": 1, "position": 1, "grid": 1, "positions_gained": 1, "points": 1},
    ))

    driver_sprint = defaultdict(lambda: {"races": 0, "positions": [], "gained": [], "points": 0})
    for sd in sprint_docs:
        did = sd["driver_id"]
        ds = driver_sprint[did]
        ds["races"] += 1
        if sd.get("position"):
            ds["positions"].append(sd["position"])
        if sd.get("positions_gained") is not None:
            ds["gained"].append(sd["positions_gained"])
        ds["points"] += sd.get("points", 0)

    # ── Build patches ────────────────────────────────────────────────
    print("  Patching opponent_profiles...")
    ops = []
    now = datetime.now(timezone.utc)

    for did in driver_ids:
        patch = {}

        # Quali-race delta
        deltas = driver_deltas.get(did, [])
        if deltas:
            patch["quali_race_pace_delta_pct"] = round(sum(deltas) / len(deltas), 3)

        # Constructor-adjusted
        adj = driver_car_adj.get(did, [])
        if adj:
            patch["constructor_adjusted_finish"] = round(sum(adj) / len(adj), 2)

        # Full-grid career stats (more complete than current)
        dp = driver_points.get(did)
        if dp and dp["races"] > 0:
            patch["jolpica_total_races"] = dp["races"]
            patch["jolpica_avg_points_per_race"] = round(dp["points"] / dp["races"], 2)
            patch["jolpica_dnf_rate"] = round(dp["dnfs"] / dp["races"], 4)

        # Grid vs finish
        gvf = driver_grid_vs_finish.get(did, [])
        if gvf:
            patch["avg_positions_gained_jolpica"] = round(sum(gvf) / len(gvf), 2)

        # Sprint
        ds = driver_sprint.get(did)
        if ds and ds["races"] > 0:
            patch["sprint_races"] = ds["races"]
            patch["sprint_avg_finish"] = round(sum(ds["positions"]) / len(ds["positions"]), 2) if ds["positions"] else None
            patch["sprint_avg_gained"] = round(sum(ds["gained"]) / len(ds["gained"]), 2) if ds["gained"] else None
            patch["sprint_points"] = ds["points"]

        if patch:
            patch["jolpica_enriched_at"] = now
            ops.append(UpdateOne({"driver_id": did}, {"$set": patch}))

    if ops:
        result = db["opponent_profiles"].bulk_write(ops)
        count = result.modified_count
        print(f"  Patched {count} opponent profiles")
    return len(ops)


def expand_circuit_pit_losses(db):
    """Expand circuit_pit_loss_times with historical Jolpica pit data."""
    print("\n[Derived] Expanding circuit pit loss times...")

    # Map Jolpica circuit_id → our circuit slugs
    from enrichment.fetch_circuits import GEOJSON_TO_SLUG

    JOLPICA_TO_SLUG = {
        "albert_park": "albert_park",
        "americas": "americas",
        "bahrain": "bahrain",
        "baku": "baku",
        "catalunya": "catalunya",
        "hockenheimring": "hockenheimring",
        "hungaroring": "hungaroring",
        "imola": "imola",
        "interlagos": "interlagos",
        "istanbul": "istanbul",
        "jeddah": "jeddah",
        "losail": "losail",
        "marina_bay": "marina_bay",
        "miami": "miami",
        "monaco": "monaco",
        "monza": "monza",
        "mugello": "mugello",
        "nurburgring": "nurburgring",
        "portimao": "portimao",
        "red_bull_ring": "red_bull_ring",
        "ricard": "ricard",
        "rodriguez": "rodriguez",
        "shanghai": "shanghai",
        "silverstone": "silverstone",
        "sochi": "sochi",
        "spa": "spa",
        "suzuka": "suzuka",
        "vegas": "vegas",
        "villeneuve": "villeneuve",
        "yas_marina": "yas_marina",
        "zandvoort": "zandvoort",
    }

    pipeline = [
        {"$match": {"duration_s": {"$ne": None, "$gt": 0, "$lt": 120}}},
        {"$group": {
            "_id": "$circuit_id",
            "avg_duration_s": {"$avg": "$duration_s"},
            "median_values": {"$push": "$duration_s"},
            "sample_count": {"$sum": 1},
            "min_duration_s": {"$min": "$duration_s"},
            "max_duration_s": {"$max": "$duration_s"},
        }},
    ]

    results = list(db["jolpica_pit_stops"].aggregate(pipeline, allowDiskUse=True))
    print(f"  Aggregated pit data for {len(results)} circuits")

    ops = []
    now = datetime.now(timezone.utc)

    for r in results:
        circuit_id = r["_id"]
        slug = JOLPICA_TO_SLUG.get(circuit_id)
        if not slug:
            continue

        values = sorted(r["median_values"])
        median = values[len(values) // 2] if values else None

        patch = {
            "jolpica_avg_pit_duration_s": round(r["avg_duration_s"], 2),
            "jolpica_median_pit_duration_s": round(median, 2) if median else None,
            "jolpica_pit_sample_count": r["sample_count"],
            "jolpica_min_pit_s": round(r["min_duration_s"], 2),
            "jolpica_max_pit_s": round(r["max_duration_s"], 2),
            "jolpica_pit_updated_at": now,
        }

        ops.append(UpdateOne(
            {"circuit": slug},
            {"$set": patch},
            upsert=False,
        ))

    if ops:
        result = db["circuit_pit_loss_times"].bulk_write(ops)
        print(f"  Enriched {result.modified_count} circuits with Jolpica pit data")

    return len(ops)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    db = get_db()
    print("Connected to MongoDB")

    ingest_race_results(db)
    ingest_qualifying(db)
    ingest_pit_stops(db)
    ingest_sprints(db)
    ingest_constructor_standings(db)
    ingest_driver_standings(db)

    compute_derived_metrics(db)
    expand_circuit_pit_losses(db)

    # Final stats
    print("\n" + "=" * 50)
    print("  Final Collection Counts")
    print("=" * 50)
    for col in [
        "jolpica_race_results", "jolpica_qualifying",
        "jolpica_pit_stops", "jolpica_sprint_results",
        "jolpica_constructor_standings",
        "jolpica_driver_standings",
    ]:
        count = db[col].count_documents({})
        print(f"  {col}: {count:,}")

    # Show sample enriched profile
    sample = db["opponent_profiles"].find_one(
        {"driver_id": "hamilton"},
        {"_id": 0, "driver_id": 1, "quali_race_pace_delta_pct": 1,
         "constructor_adjusted_finish": 1, "sprint_races": 1,
         "sprint_avg_finish": 1, "jolpica_total_races": 1,
         "avg_positions_gained_jolpica": 1},
    )
    if sample:
        print(f"\n  Sample (hamilton): {sample}")


if __name__ == "__main__":
    main()
