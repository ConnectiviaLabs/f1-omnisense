"""
feature_store  --  MongoDB-backed cache for cross-model computation sharing.

Cached computation types
------------------------
- anomaly_scores   : per-driver anomaly z-scores from the anomaly pipeline
- shap_features    : SHAP feature-importance vectors
- forecasts:{column}:{horizon} : forecast results keyed by target column and
  horizon length (e.g. forecasts:speed:5)
- health_report    : composite car-health / reliability reports
- telemetry_raw    : cleaned telemetry slices used as input by multiple models

Every entry is keyed by (session_key, driver_number, computation).
Historical race data is immutable, so cached results never expire.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_key(session_key: int, driver_number: int, computation: str) -> dict:
    """Build the compound key used for lookups and upserts."""
    return {
        "session_key": session_key,
        "driver_number": driver_number,
        "computation": computation,
    }


COLLECTION = "feature_store"


# ── public API ───────────────────────────────────────────────────────────────

def get(db, session_key: int, driver_number: int, computation: str):
    """Return the cached *result* for the given key, or ``None``."""
    if db is None:
        return None
    try:
        doc = db[COLLECTION].find_one(_make_key(session_key, driver_number, computation))
        if doc is not None:
            return doc.get("result")
        return None
    except Exception:
        logger.exception("feature_store.get failed for %s/%s/%s",
                         session_key, driver_number, computation)
        return None


def put(db, session_key: int, driver_number: int, computation: str, result) -> bool:
    """Upsert a computation result.  Returns ``True`` on success."""
    if db is None or result is None:
        return False
    try:
        key = _make_key(session_key, driver_number, computation)
        db[COLLECTION].update_one(
            key,
            {
                "$set": {**key, "result": result},
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )
        return True
    except Exception:
        logger.exception("feature_store.put failed for %s/%s/%s",
                         session_key, driver_number, computation)
        return False


def ensure_indexes(db) -> None:
    """Create the compound unique index if it doesn't already exist."""
    if db is None:
        return
    try:
        db[COLLECTION].create_index(
            [("session_key", 1), ("driver_number", 1), ("computation", 1)],
            unique=True,
            name="ux_session_driver_computation",
        )
    except Exception:
        logger.exception("feature_store.ensure_indexes failed")
