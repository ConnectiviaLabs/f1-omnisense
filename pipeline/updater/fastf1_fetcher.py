"""Fetch lap and weather data via the FastF1 library.

FastF1 is optional — if not installed, this module logs a warning and skips.
Install with: pip install fastf1
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pymongo import UpdateOne
from pymongo.database import Database

logger = logging.getLogger(__name__)

try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False


def _get_existing_races(db: Database) -> set[tuple[int, str]]:
    """Return set of (Year, Race) already in fastf1_laps."""
    pipeline = [
        {"$group": {"_id": {"Year": "$Year", "Race": "$Race"}}},
    ]
    results = db["fastf1_laps"].aggregate(pipeline)
    return {(r["_id"]["Year"], r["_id"]["Race"]) for r in results}


def _session_laps_to_docs(session, year: int) -> list[dict]:
    """Convert a FastF1 session's laps to MongoDB docs."""
    laps = session.laps
    if laps.empty:
        return []

    docs = []
    race_name = session.event["EventName"]
    session_type = session.name  # "Race", "Qualifying", etc.

    for _, lap in laps.iterrows():
        doc = {
            "Year": year,
            "Race": race_name,
            "SessionType": session_type,
            "Driver": lap.get("Driver"),
            "DriverNumber": str(lap.get("DriverNumber", "")),
            "LapNumber": int(lap["LapNumber"]) if not _isna(lap.get("LapNumber")) else None,
            "LapTime": str(lap.get("LapTime", "")),
            "Compound": lap.get("Compound"),
            "TyreLife": int(lap["TyreLife"]) if not _isna(lap.get("TyreLife")) else None,
            "FreshTyre": bool(lap.get("FreshTyre")) if not _isna(lap.get("FreshTyre")) else None,
            "Stint": int(lap["Stint"]) if not _isna(lap.get("Stint")) else None,
            "Team": lap.get("Team"),
            "Position": int(lap["Position"]) if not _isna(lap.get("Position")) else None,
            "Sector1Time": str(lap.get("Sector1Time", "")),
            "Sector2Time": str(lap.get("Sector2Time", "")),
            "Sector3Time": str(lap.get("Sector3Time", "")),
            "SpeedI1": float(lap["SpeedI1"]) if not _isna(lap.get("SpeedI1")) else None,
            "SpeedI2": float(lap["SpeedI2"]) if not _isna(lap.get("SpeedI2")) else None,
            "SpeedFL": float(lap["SpeedFL"]) if not _isna(lap.get("SpeedFL")) else None,
            "SpeedST": float(lap["SpeedST"]) if not _isna(lap.get("SpeedST")) else None,
            "IsAccurate": bool(lap.get("IsAccurate")) if not _isna(lap.get("IsAccurate")) else None,
            "TrackStatus": str(lap.get("TrackStatus", "")),
        }
        docs.append(doc)
    return docs


def _session_weather_to_docs(session, year: int) -> list[dict]:
    """Convert a FastF1 session's weather to MongoDB docs."""
    weather = session.weather_data
    if weather is None or weather.empty:
        return []

    docs = []
    race_name = session.event["EventName"]
    session_type = session.name

    for _, row in weather.iterrows():
        doc = {
            "Year": year,
            "Race": race_name,
            "SessionType": session_type,
            "AirTemp": float(row["AirTemp"]) if not _isna(row.get("AirTemp")) else None,
            "Humidity": float(row["Humidity"]) if not _isna(row.get("Humidity")) else None,
            "Pressure": float(row["Pressure"]) if not _isna(row.get("Pressure")) else None,
            "Rainfall": bool(row.get("Rainfall")) if not _isna(row.get("Rainfall")) else None,
            "TrackTemp": float(row["TrackTemp"]) if not _isna(row.get("TrackTemp")) else None,
            "WindDirection": int(row["WindDirection"]) if not _isna(row.get("WindDirection")) else None,
            "WindSpeed": float(row["WindSpeed"]) if not _isna(row.get("WindSpeed")) else None,
        }
        docs.append(doc)
    return docs


def _isna(val) -> bool:
    """Check if a value is NaN/None/NaT."""
    if val is None:
        return True
    try:
        import pandas as pd
        return pd.isna(val)
    except (ImportError, TypeError, ValueError):
        return False


def _bulk_upsert_laps(db: Database, docs: list[dict]) -> int:
    """Upsert lap docs into fastf1_laps."""
    if not docs:
        return 0
    now = datetime.now(timezone.utc)
    ops = []
    for doc in docs:
        filt = {
            "Year": doc["Year"],
            "Race": doc["Race"],
            "SessionType": doc["SessionType"],
            "Driver": doc["Driver"],
            "LapNumber": doc["LapNumber"],
        }
        doc["ingested_at"] = now
        ops.append(UpdateOne(filt, {"$set": doc}, upsert=True))

    total = 0
    for i in range(0, len(ops), 1000):
        result = db["fastf1_laps"].bulk_write(ops[i : i + 1000], ordered=False)
        total += result.upserted_count + result.modified_count
    return total


def _bulk_upsert_weather(db: Database, docs: list[dict]) -> int:
    """Upsert weather docs into fastf1_weather."""
    if not docs:
        return 0
    now = datetime.now(timezone.utc)
    ops = []
    for doc in docs:
        filt = {
            "Year": doc["Year"],
            "Race": doc["Race"],
            "SessionType": doc["SessionType"],
        }
        doc["ingested_at"] = now
        ops.append(UpdateOne(filt, {"$set": doc}, upsert=True))

    total = 0
    for i in range(0, len(ops), 1000):
        result = db["fastf1_weather"].bulk_write(ops[i : i + 1000], ordered=False)
        total += result.upserted_count + result.modified_count
    return total


def sync(
    db: Database,
    year: int | None = None,
    session_types: list[str] | None = None,
) -> dict[str, int]:
    """Fetch new laps/weather via FastF1 and upsert into MongoDB.

    Args:
        db: MongoDB database
        year: Year to sync. None = current year.
        session_types: Session types to fetch (default: ["R"] = race only).

    Returns:
        Dict with counts: {"fastf1_laps": N, "fastf1_weather": N}
    """
    if not FASTF1_AVAILABLE:
        logger.warning("FastF1 not installed. Skipping fastf1 sync. Install with: pip install fastf1")
        print("  [SKIP] FastF1 not installed. Run: pip install fastf1")
        return {}

    if year is None:
        year = datetime.now().year
    if session_types is None:
        session_types = ["R"]

    print(f"\n{'='*60}")
    print(f"  FastF1 Sync — {year}")
    print(f"{'='*60}")

    existing = _get_existing_races(db)
    schedule = fastf1.get_event_schedule(year)
    results = {"fastf1_laps": 0, "fastf1_weather": 0}

    session_type_map = {"R": "Race", "Q": "Qualifying", "FP1": "Practice 1",
                        "FP2": "Practice 2", "FP3": "Practice 3", "S": "Sprint"}

    for _, event in schedule.iterrows():
        event_name = event["EventName"]

        for st_code in session_types:
            if (year, event_name) in existing and st_code == "R":
                continue

            st_name = session_type_map.get(st_code, st_code)
            try:
                session = fastf1.get_session(year, event_name, st_name)
                session.load(telemetry=False, messages=False)
            except Exception as e:
                logger.debug("FastF1 skip %s %s: %s", event_name, st_name, e)
                continue

            lap_docs = _session_laps_to_docs(session, year)
            weather_docs = _session_weather_to_docs(session, year)

            laps_n = _bulk_upsert_laps(db, lap_docs)
            weather_n = _bulk_upsert_weather(db, weather_docs)

            results["fastf1_laps"] += laps_n
            results["fastf1_weather"] += weather_n
            print(f"  {event_name} {st_name}: {laps_n} laps, {weather_n} weather")

    print(f"\n  FastF1 sync complete: {results['fastf1_laps']} laps, {results['fastf1_weather']} weather")
    return results
