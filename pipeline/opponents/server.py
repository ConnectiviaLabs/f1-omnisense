"""FastAPI router for opponent profile queries.

Can run standalone on port 8102 or be included as a router in chat_server.py:
    from pipeline.opponents.server import router as opponents_router
    app.include_router(opponents_router)
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

from .profiler import OpponentProfiler

logger = logging.getLogger(__name__)

_profiler: OpponentProfiler | None = None


def _get_profiler() -> OpponentProfiler:
    global _profiler
    if _profiler is None:
        _profiler = OpponentProfiler()
    return _profiler


def init_profiler_with_db(db) -> None:
    """Inject a shared MongoDB database to avoid a second MongoClient."""
    global _profiler
    _profiler = OpponentProfiler(db=db)


# ── Router ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/opponents", tags=["Opponent Profiles"])


@router.get("/stats")
def get_stats():
    """Collection counts and available metrics."""
    return _get_profiler().stats()


# ── Driver endpoints ───────────────────────────────────────────────────

@router.get("/drivers")
def list_drivers():
    """Summary list of all profiled drivers."""
    drivers = _get_profiler().get_all_drivers()
    return {"drivers": drivers, "count": len(drivers)}


@router.get("/drivers/{driver_id}")
def get_driver(driver_id: str):
    """Full opponent profile for one driver."""
    doc = _get_profiler().get_driver(driver_id)
    if doc is None:
        raise HTTPException(404, f"No profile for '{driver_id}'")
    return doc


@router.get("/drivers/{driver_id}/circuits")
def get_driver_circuits(
    driver_id: str,
    circuit: str | None = Query(default=None),
):
    """Circuit breakdowns for a driver, optionally filtered."""
    docs = _get_profiler().get_driver_circuits(driver_id, circuit)
    if not docs:
        raise HTTPException(404, f"No circuit profiles for '{driver_id}'")
    return {"driver_id": driver_id, "profiles": docs, "count": len(docs)}


@router.get("/drivers/{driver_id}/compounds")
def get_driver_compounds(driver_id: str):
    """Tyre compound profiles for a driver."""
    docs = _get_profiler().get_driver_compounds(driver_id)
    if not docs:
        raise HTTPException(404, f"No compound profiles for '{driver_id}'")
    return {"driver_id": driver_id, "compounds": docs, "count": len(docs)}


# ── Comparison / leaderboard ───────────────────────────────────────────

@router.get("/compare")
def compare_drivers(
    drivers: str = Query(
        ..., description="Comma-separated driver_ids (e.g. hamilton,max_verstappen,norris)"
    ),
    metrics: str | None = Query(
        default=None, description="Comma-separated metric names to include"
    ),
):
    """Side-by-side comparison of multiple drivers."""
    ids = [d.strip() for d in drivers.split(",") if d.strip()]
    if len(ids) < 2:
        raise HTTPException(400, "Provide at least 2 driver_ids")
    metric_list = (
        [m.strip() for m in metrics.split(",") if m.strip()]
        if metrics
        else None
    )
    docs = _get_profiler().compare_drivers(ids, metric_list)
    return {"drivers": docs, "count": len(docs)}


@router.get("/leaderboard/{metric}")
def get_leaderboard(
    metric: str,
    top_n: int = Query(default=10, ge=1, le=40),
    ascending: bool = Query(default=False),
):
    """Top N drivers by any numeric metric."""
    docs = _get_profiler().get_leaderboard(metric, top_n, ascending)
    if not docs:
        raise HTTPException(404, f"No data for metric '{metric}'")
    return {"metric": metric, "ascending": ascending, "drivers": docs}


# ── Circuit endpoints ──────────────────────────────────────────────────

@router.get("/circuits/pit-loss")
def list_pit_losses():
    """All circuits sorted by median pit loss."""
    docs = _get_profiler().get_all_pit_losses()
    return {"circuits": docs, "count": len(docs)}


@router.get("/circuits/{circuit}/pit-loss")
def get_pit_loss(circuit: str):
    """Pit stop loss time stats for a circuit."""
    doc = _get_profiler().get_pit_loss(circuit)
    if doc is None:
        raise HTTPException(404, f"No pit loss data for '{circuit}'")
    return doc


@router.get("/circuits/{circuit}/drivers")
def get_circuit_drivers(circuit: str):
    """All drivers at a specific circuit — comparison view."""
    overview = _get_profiler().get_circuit_overview(circuit)
    if not overview["drivers"]:
        raise HTTPException(404, f"No data for circuit '{circuit}'")
    return overview


# ── Compound endpoints ─────────────────────────────────────────────────

@router.get("/compounds/{compound}/rankings")
def get_compound_rankings(
    compound: str,
    metric: str = Query(default="degradation_slope"),
    top_n: int = Query(default=10, ge=1, le=40),
):
    """Rank drivers by a metric on a specific tyre compound."""
    docs = _get_profiler().get_compound_rankings(compound, metric, top_n)
    if not docs:
        raise HTTPException(404, f"No data for compound '{compound}'")
    return {"compound": compound, "metric": metric, "drivers": docs}


# ── Standalone mode ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    app = FastAPI(title="F1 Opponent Profiles API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "f1-opponents"}

    port = int(os.getenv("OPPONENTS_PORT", "8102"))
    logger.info("Starting F1 Opponents API on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
