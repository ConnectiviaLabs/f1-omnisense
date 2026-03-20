"""
Build 2025+ Race Summaries from OpenF1 car_data API
────────────────────────────────────────────────────
Fetches car_data (speed, rpm, throttle, brake, drs) per driver per race
from the OpenF1 API and aggregates into telemetry_race_summary.

This supplements the FastF1-based build_telemetry_summaries.py for years
where FastF1 data is unavailable.

Usage:
    python -m pipeline.enrichment.build_openf1_race_summaries [--year 2025]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from updater._db import get_db

API_BASE = "https://api.openf1.org/v1"

# circuit_short_name → GP race name (matching telemetry_race_summary convention)
CIRCUIT_TO_RACE = {
    "Melbourne": "Australian Grand Prix",
    "Shanghai": "Chinese Grand Prix",
    "Suzuka": "Japanese Grand Prix",
    "Sakhir": "Bahrain Grand Prix",
    "Jeddah": "Saudi Arabian Grand Prix",
    "Miami": "Miami Grand Prix",
    "Imola": "Emilia Romagna Grand Prix",
    "Monte Carlo": "Monaco Grand Prix",
    "Catalunya": "Spanish Grand Prix",
    "Montreal": "Canadian Grand Prix",
    "Spielberg": "Austrian Grand Prix",
    "Silverstone": "British Grand Prix",
    "Spa-Francorchamps": "Belgian Grand Prix",
    "Hungaroring": "Hungarian Grand Prix",
    "Zandvoort": "Dutch Grand Prix",
    "Monza": "Italian Grand Prix",
    "Baku": "Azerbaijan Grand Prix",
    "Singapore": "Singapore Grand Prix",
    "Austin": "United States Grand Prix",
    "Mexico City": "Mexico City Grand Prix",
    "Interlagos": "São Paulo Grand Prix",
    "Las Vegas": "Las Vegas Grand Prix",
    "Lusail": "Qatar Grand Prix",
    "Yas Marina Circuit": "Abu Dhabi Grand Prix",
}


def fetch_car_data(session_key: int, driver_number: int) -> list[dict]:
    """Fetch car_data from OpenF1 API for one driver in one session."""
    for attempt in range(3):
        try:
            resp = requests.get(f"{API_BASE}/car_data", params={
                "session_key": session_key,
                "driver_number": driver_number,
            }, timeout=60)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return []


def aggregate_car_data(records: list[dict]) -> dict:
    """Aggregate raw car_data records into summary stats."""
    speeds = [r["speed"] for r in records if r.get("speed") and r["speed"] > 0]
    rpms = [r["rpm"] for r in records if r.get("rpm") and r["rpm"] > 0]
    throttles = [r["throttle"] for r in records if r.get("throttle") is not None and 0 <= r["throttle"] <= 100]
    brakes = [r["brake"] for r in records if r.get("brake") is not None]
    drs_vals = [r["drs"] for r in records if r.get("drs") is not None]

    doc = {"samples": len(records)}

    if speeds:
        doc["avg_speed"] = round(sum(speeds) / len(speeds), 1)
        sorted_speeds = sorted(speeds)
        idx99 = int(len(sorted_speeds) * 0.99)
        doc["top_speed"] = round(sorted_speeds[min(idx99, len(sorted_speeds) - 1)], 1)

    if rpms:
        doc["avg_rpm"] = round(sum(rpms) / len(rpms), 0)
        sorted_rpms = sorted(rpms)
        idx99 = int(len(sorted_rpms) * 0.99)
        doc["max_rpm"] = round(sorted_rpms[min(idx99, len(sorted_rpms) - 1)], 0)

    if throttles:
        doc["avg_throttle"] = round(sum(throttles) / len(throttles), 1)

    if brakes:
        brake_on = sum(1 for b in brakes if b > 0)
        doc["brake_pct"] = round(brake_on / len(brakes) * 100, 1)

    if drs_vals:
        drs_on = sum(1 for d in drs_vals if d >= 10)
        doc["drs_pct"] = round(drs_on / len(drs_vals) * 100, 1)

    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    year = args.year

    db = get_db()
    print(f"✅ Connected to MongoDB")
    print(f"\nBuilding telemetry_race_summary for {year} from OpenF1 car_data API...")

    # Get race sessions (deduplicated, exclude sprints)
    sessions_raw = list(db["openf1_sessions"].find(
        {"year": year, "session_type": "Race", "session_name": "Race"},
        {"session_key": 1, "circuit_short_name": 1, "date_start": 1, "_id": 0},
    ).sort("date_start", 1))

    # Deduplicate by session_key
    seen_keys = set()
    sessions = []
    for s in sessions_raw:
        if s["session_key"] not in seen_keys:
            seen_keys.add(s["session_key"])
            sessions.append(s)

    print(f"  Found {len(sessions)} unique Race sessions for {year}")

    # Check what already exists
    existing = set()
    for doc in db["telemetry_race_summary"].find(
        {"Year": year},
        {"Driver": 1, "Race": 1, "_id": 0},
    ):
        existing.add((doc["Driver"], doc["Race"]))

    # Driver number → code mapping
    num_to_code = {}
    for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0}):
        num_to_code[doc["driver_number"]] = doc["name_acronym"]

    ops = []
    total_fetched = 0

    for si, session in enumerate(sessions):
        sk = session["session_key"]
        circuit = session["circuit_short_name"]
        race_name = CIRCUIT_TO_RACE.get(circuit)
        if not race_name:
            print(f"  ⚠ Unknown circuit: {circuit}, skipping")
            continue

        # Get drivers for this session
        driver_nums = db["openf1_drivers"].distinct("driver_number", {"session_key": sk})
        if not driver_nums:
            continue

        # Get compounds from stints
        stint_compounds = {}
        for stint in db["openf1_stints"].find(
            {"session_key": sk},
            {"driver_number": 1, "compound": 1, "_id": 0},
        ):
            dn = stint["driver_number"]
            comp = stint.get("compound")
            if comp and comp not in ("UNKNOWN", "None", ""):
                stint_compounds.setdefault(dn, set()).add(comp)

        skipped = 0
        fetched = 0

        for dn in sorted(driver_nums):
            code = num_to_code.get(dn, f"#{dn}")
            if (code, race_name) in existing:
                skipped += 1
                continue

            car_data = fetch_car_data(sk, dn)
            if not car_data or len(car_data) < 500:
                continue

            stats = aggregate_car_data(car_data)
            if stats.get("avg_speed", 0) < 30:
                continue

            compounds = sorted(stint_compounds.get(dn, []))

            doc = {
                "Driver": code,
                "Year": year,
                "Race": race_name,
                "_source_file": f"{year}_{race_name.replace(' ', '_')}_Race_openf1",
                **stats,
            }
            if compounds:
                doc["compounds"] = compounds

            ops.append(UpdateOne(
                {"Driver": code, "Year": year, "Race": race_name},
                {"$set": doc},
                upsert=True,
            ))
            fetched += 1
            total_fetched += 1
            time.sleep(0.3)

        print(f"  [{si+1}/{len(sessions)}] {circuit:20s} → {race_name}: {fetched} drivers fetched, {skipped} skipped")

        # Bulk write after each session
        if ops:
            result = db["telemetry_race_summary"].bulk_write(ops, ordered=False)
            print(f"    → Wrote {result.upserted_count + result.modified_count} docs")
            ops = []

    # Final flush
    if ops:
        result = db["telemetry_race_summary"].bulk_write(ops, ordered=False)
        print(f"  → Wrote {result.upserted_count + result.modified_count} docs")

    # Verify
    count_2025 = db["telemetry_race_summary"].count_documents({"Year": year})
    total = db["telemetry_race_summary"].count_documents({})
    print(f"\n✅ Done. telemetry_race_summary: {count_2025} docs for {year}, {total} total")


if __name__ == "__main__":
    main()
