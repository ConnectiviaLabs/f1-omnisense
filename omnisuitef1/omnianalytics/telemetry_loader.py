"""
Shared telemetry loader with feature_store caching.

Replaces independent MongoDB car_data queries that were duplicated across:
- telemetry_anomaly.py
- predictive_maintenance.py
- omni_analytics_router.py
- omni_health_router.py

All of those modules can now call ``load_session_telemetry()`` instead of
querying car_data / openf1_car_data directly, gaining transparent caching
through the feature_store.
"""

import logging
from typing import Optional

import pandas as pd

from omnianalytics import feature_store

logger = logging.getLogger(__name__)


def load_session_telemetry(
    db,
    session_key: int,
    driver_number: Optional[int] = None,
) -> pd.DataFrame:
    """Load car telemetry for a session (and optionally a single driver).

    Checks the feature_store cache first.  On a miss, queries ``car_data``
    (falling back to ``openf1_car_data``), caches the result, and returns
    a DataFrame sorted by ``date``.
    """
    if db is None:
        return pd.DataFrame()

    cache_driver = driver_number if driver_number is not None else 0

    # ── cache check ──────────────────────────────────────────────────
    cached = feature_store.get(db, session_key, cache_driver, "telemetry_raw")
    if cached is not None:
        logger.debug("telemetry cache hit  session=%s driver=%s", session_key, cache_driver)
        return pd.DataFrame(cached)

    # ── build query ──────────────────────────────────────────────────
    query: dict = {"session_key": session_key}
    if driver_number is not None:
        query["driver_number"] = driver_number

    projection = {"_id": 0}

    # ── primary collection ───────────────────────────────────────────
    docs = list(db["car_data"].find(query, projection).sort("date", 1))

    # ── fallback collection ──────────────────────────────────────────
    if not docs:
        docs = list(db["openf1_car_data"].find(query, projection).sort("date", 1))

    if not docs:
        logger.warning(
            "No telemetry found for session_key=%s driver_number=%s",
            session_key,
            driver_number,
        )
        return pd.DataFrame()

    # ── build DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame(docs)

    # ── cache write (non-fatal) ──────────────────────────────────────
    try:
        records = df.copy()
        # Convert datetime columns to strings so they survive BSON round-trip
        for col in records.select_dtypes(include=["datetime", "datetimetz"]).columns:
            records[col] = records[col].astype(str)
        feature_store.put(db, session_key, cache_driver, "telemetry_raw", records.to_dict("records"))
    except Exception:
        logger.exception("Failed to cache telemetry for session=%s driver=%s", session_key, cache_driver)

    return df
