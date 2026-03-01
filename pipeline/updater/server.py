"""FastAPI router for triggering and monitoring data updates.

Standalone: python pipeline/updater/server.py (port 8103)
Or include in chat_server.py:
    from pipeline.updater.server import router as updater_router
    app.include_router(updater_router)
"""

import os
import logging
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

from .updater import LiveUpdater

logger = logging.getLogger(__name__)

_updater: LiveUpdater | None = None
_sync_running = False
_last_result: dict | None = None


def _get_updater() -> LiveUpdater:
    global _updater
    if _updater is None:
        _updater = LiveUpdater()
    return _updater


router = APIRouter(prefix="/api/updater", tags=["Data Updater"])


@router.get("/status")
def get_status():
    """Current data state: collection counts, year coverage, gaps."""
    status = _get_updater().status()
    status["sync_running"] = _sync_running
    return status


@router.get("/gaps")
def get_gaps():
    """Show missing years/sessions per collection."""
    status = _get_updater().status()
    return {
        "gaps": status.get("gaps", {}),
        "openf1_years": status.get("openf1_years", []),
        "fastf1_years": status.get("fastf1_years", []),
    }


@router.post("/sync")
def trigger_sync(
    year: int | None = Query(default=None, description="Year to sync"),
    full_refresh: bool = Query(default=False),
    skip_fastf1: bool = Query(default=True, description="Skip FastF1 (slow)"),
    skip_profiles: bool = Query(default=False),
):
    """Trigger a data sync in a background thread.

    Returns immediately with status. Check /status for progress.
    """
    global _sync_running, _last_result

    if _sync_running:
        return {"status": "already_running", "message": "A sync is already in progress"}

    def _run():
        global _sync_running, _last_result
        _sync_running = True
        try:
            _last_result = _get_updater().sync(
                year=year,
                full_refresh=full_refresh,
                skip_fastf1=skip_fastf1,
                skip_profiles=skip_profiles,
            )
        except Exception as e:
            logger.error("Sync failed: %s", e)
            _last_result = {"error": str(e)}
        finally:
            _sync_running = False

    thread = Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started", "year": year or "current"}


@router.get("/last-result")
def get_last_result():
    """Get the result of the last sync run."""
    if _last_result is None:
        return {"status": "no_runs_yet"}
    return {"status": "complete", "result": _last_result}


# ── Standalone mode ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    app = FastAPI(title="F1 Data Updater API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "f1-updater"}

    port = int(os.getenv("UPDATER_PORT", "8103"))
    logger.info("Starting F1 Updater API on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
