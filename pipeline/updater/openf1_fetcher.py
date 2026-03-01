"""Fetch data from the OpenF1 API and upsert into MongoDB.

OpenF1 API docs: https://openf1.org
Base URL: https://api.openf1.org/v1/
No auth required. Rate-limited by courtesy (0.5s between requests).
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timezone

import requests
from pymongo import UpdateOne
from pymongo.database import Database

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
REQUEST_DELAY = 0.5  # seconds between API calls

# Mapping: (api_endpoint, mongodb_collection, upsert_key_fields)
ENDPOINTS = [
    ("laps", "openf1_laps", ["session_key", "driver_number", "lap_number"]),
    ("intervals", "openf1_intervals", ["session_key", "driver_number", "date"]),
    ("position", "openf1_position", ["session_key", "driver_number", "date"]),
    ("stints", "openf1_stints", ["session_key", "driver_number", "stint_number"]),
    ("pit", "openf1_pit", ["session_key", "driver_number", "lap_number"]),
    ("race_control", "openf1_race_control", ["session_key", "date"]),
    ("weather", "openf1_weather", ["session_key", "date"]),
    ("drivers", "openf1_drivers", ["session_key", "driver_number"]),
]


def _api_get(endpoint: str, params: dict) -> list[dict]:
    """GET from OpenF1 API with retry (skip retry on 404)."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return []  # no data, not an error
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("OpenF1 %s attempt %d failed: %s", endpoint, attempt + 1, e)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return []


def _bulk_upsert(db: Database, collection: str, docs: list[dict], key_fields: list[str]) -> int:
    """Upsert docs into MongoDB collection."""
    if not docs:
        return 0
    ops = []
    now = datetime.now(timezone.utc)
    for doc in docs:
        filt = {k: doc[k] for k in key_fields if k in doc}
        if not filt:
            continue
        doc["ingested_at"] = now
        ops.append(UpdateOne(filt, {"$set": doc}, upsert=True))

    if not ops:
        return 0

    # Batch in groups of 1000
    total = 0
    for i in range(0, len(ops), 1000):
        result = db[collection].bulk_write(ops[i : i + 1000], ordered=False)
        total += result.upserted_count + result.modified_count
    return total


def fetch_sessions(year: int) -> list[dict]:
    """Fetch all sessions for a year from OpenF1."""
    return _api_get("sessions", {"year": year})


def get_existing_session_keys(db: Database, year: int | None = None) -> set[int]:
    """Get session_keys already in MongoDB."""
    filt = {"year": year} if year else {}
    return set(db["openf1_sessions"].distinct("session_key", filt))


def sync(
    db: Database,
    year: int | None = None,
    full_refresh: bool = False,
    session_types: list[str] | None = None,
) -> dict[str, int]:
    """Sync OpenF1 data for a year (or auto-detect current year).

    Args:
        db: MongoDB database
        year: Year to sync. None = current year.
        full_refresh: If True, re-fetch even if session_key exists.
        session_types: Filter to specific types (e.g. ["Race", "Qualifying"]).

    Returns:
        Dict mapping collection name to count of upserted docs.
    """
    if year is None:
        year = datetime.now().year

    print(f"\n{'='*60}")
    print(f"  OpenF1 Sync — {year}")
    print(f"{'='*60}")

    # 1. Fetch sessions from API
    api_sessions = fetch_sessions(year)
    if not api_sessions:
        print(f"  No sessions found for {year}")
        return {}

    if session_types:
        api_sessions = [s for s in api_sessions if s.get("session_type") in session_types]

    # 2. Find new sessions
    existing_keys = get_existing_session_keys(db, year)
    if full_refresh:
        new_sessions = api_sessions
    else:
        new_sessions = [s for s in api_sessions if s.get("session_key") not in existing_keys]

    print(f"  API sessions: {len(api_sessions)}, existing: {len(existing_keys)}, new: {len(new_sessions)}")

    if not new_sessions:
        print("  Everything up to date.")
        return {}

    # 3. Upsert session metadata
    session_count = _bulk_upsert(
        db, "openf1_sessions", api_sessions,
        ["session_key"],
    )
    print(f"  openf1_sessions: {session_count} upserted")

    # 4. For each new session, fetch all data types
    results: dict[str, int] = {"openf1_sessions": session_count}
    total_sessions = len(new_sessions)

    for idx, session in enumerate(new_sessions, 1):
        sk = session["session_key"]
        sname = session.get("session_name", "?")
        circuit = session.get("circuit_short_name", "?")
        print(f"\n  [{idx}/{total_sessions}] {circuit} — {sname} (session_key={sk})")

        for endpoint, collection, key_fields in ENDPOINTS:
            time.sleep(REQUEST_DELAY)
            docs = _api_get(endpoint, {"session_key": sk})
            if docs:
                count = _bulk_upsert(db, collection, docs, key_fields)
                results[collection] = results.get(collection, 0) + count
                print(f"    {collection}: {len(docs)} fetched, {count} upserted")
            else:
                print(f"    {collection}: 0 (no data)")

    print(f"\n  Sync complete for {year}.")
    return results
