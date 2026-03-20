"""Team Radio Intelligence APIRouter.

Endpoints:
    GET  /api/radio/sessions            — list available sessions (year/meeting/session)
    GET  /api/radio/timeline            — radio timeline for a session
    GET  /api/radio/audio/{file_key}    — serve MP3 audio file
    GET  /api/radio/driver-profile      — driver communication profile
    GET  /api/radio/balance/{circuit}   — car balance timeline for a circuit
    GET  /api/radio/reliability         — reliability precursor events
    GET  /api/radio/strategy            — strategy comms for a session
    GET  /api/radio/stats               — signal distribution stats
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/radio", tags=["radio"])

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "team_radio"


def _get_db():
    from pipeline.updater._db import get_db
    return get_db()


# ── Sessions ─────────────────────────────────────────────────────────────


@router.get("/sessions")
async def list_sessions(
    year: int | None = Query(None),
    driver: str | None = Query(None),
):
    """List available sessions with radio data."""
    db = _get_db()
    pipeline = [
        {"$match": {"signal_tags": {"$exists": True}}},
    ]
    if year:
        pipeline[0]["$match"]["year"] = year
    if driver:
        pipeline[0]["$match"]["driver_name"] = {"$regex": driver, "$options": "i"}

    pipeline.extend([
        {"$group": {
            "_id": {"year": "$year", "meeting": "$meeting", "session": "$session"},
            "count": {"$sum": 1},
            "drivers": {"$addToSet": "$driver_name"},
        }},
        {"$sort": {"_id.year": -1, "_id.meeting": 1, "_id.session": 1}},
    ])

    results = list(db["team_radio_transcripts"].aggregate(pipeline))
    sessions = []
    for r in results:
        sessions.append({
            "year": r["_id"]["year"],
            "meeting": r["_id"]["meeting"],
            "session": r["_id"]["session"],
            "message_count": r["count"],
            "drivers": sorted(r["drivers"]),
        })
    return sessions


# ── Timeline ─────────────────────────────────────────────────────────────


@router.get("/timeline")
async def radio_timeline(
    year: int = Query(...),
    meeting: str = Query(...),
    session: str = Query(...),
    driver: str | None = Query(None),
    signal_type: str | None = Query(None),
    min_urgency: int = Query(0),
):
    """Get chronological radio timeline for a session."""
    db = _get_db()
    query = {
        "year": year,
        "meeting": meeting,
        "session": session,
        "signal_tags": {"$exists": True},
    }
    if driver:
        query["driver_name"] = {"$regex": driver, "$options": "i"}
    if signal_type:
        query["signal_tags"] = signal_type
    if min_urgency > 0:
        query["urgency_level"] = {"$gte": min_urgency}

    docs = list(db["team_radio_transcripts"].find(
        query,
        {"_id": 0, "file_key": 1, "transcript": 1, "driver_name": 1,
         "driver_number": 1, "time": 1, "date": 1, "signal_tags": 1,
         "sentiment_score": 1, "urgency_level": 1, "speaker": 1,
         "mentioned_components": 1, "mentioned_corners": 1,
         "tyre_compound_mentioned": 1, "confidence": 1, "sequence": 1},
    ).sort("time", 1))

    return {
        "year": year,
        "meeting": meeting,
        "session": session,
        "messages": docs,
        "count": len(docs),
    }


# ── Audio ────────────────────────────────────────────────────────────────


@router.get("/audio/{file_key:path}")
async def serve_audio(file_key: str):
    """Serve an MP3 file by file_key (e.g. 2024/British_Grand_Prix/Race/LANNOR01_4_20240707_154214.mp3)."""
    mp3_path = _DATA_DIR / file_key
    if not mp3_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {file_key}")
    # Security: ensure path is within data dir
    if not mp3_path.resolve().is_relative_to(_DATA_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Forbidden")
    return FileResponse(mp3_path, media_type="audio/mpeg")


# ── Driver Profile ───────────────────────────────────────────────────────


@router.get("/driver-profile")
async def driver_profile(
    driver: str = Query(...),
    season: int | None = Query(None),
):
    """Get driver communication profile."""
    db = _get_db()
    query: dict = {"driver_name": {"$regex": driver, "$options": "i"}}
    if season:
        query["season"] = season

    docs = list(db["driver_communication_profiles"].find(query, {"_id": 0}))
    if not docs:
        raise HTTPException(status_code=404, detail=f"No profile for {driver}")
    return docs


# ── Car Balance ──────────────────────────────────────────────────────────


@router.get("/balance/{circuit}")
async def car_balance(
    circuit: str,
    season: int | None = Query(None),
):
    """Get car balance timeline for a circuit."""
    db = _get_db()
    query: dict = {"circuit": {"$regex": circuit, "$options": "i"}}
    if season:
        query["season"] = season

    docs = list(db["car_balance_timeline"].find(query, {"_id": 0, "updated_at": 0}))
    if not docs:
        raise HTTPException(status_code=404, detail=f"No balance data for {circuit}")
    return docs


# ── Reliability ──────────────────────────────────────────────────────────


@router.get("/reliability")
async def reliability_precursors(
    year: int | None = Query(None),
    meeting: str | None = Query(None),
    driver: str | None = Query(None),
    min_urgency: int = Query(0),
):
    """Get reliability precursor events."""
    db = _get_db()
    query: dict = {}
    if year:
        query["year"] = year
    if meeting:
        query["meeting"] = {"$regex": meeting, "$options": "i"}
    if driver:
        query["driver"] = {"$regex": driver, "$options": "i"}
    if min_urgency > 0:
        query["max_urgency"] = {"$gte": min_urgency}

    docs = list(db["reliability_precursors"].find(
        query, {"_id": 0, "updated_at": 0}
    ).sort([("year", -1), ("meeting", 1)]))
    return docs


# ── Strategy ─────────────────────────────────────────────────────────────


@router.get("/strategy")
async def strategy_comms(
    year: int = Query(...),
    meeting: str = Query(...),
    session: str = Query("Race"),
):
    """Get strategy communications for a session."""
    db = _get_db()
    doc = db["strategy_comms_log"].find_one(
        {"year": year, "meeting": meeting, "session": session},
        {"_id": 0, "updated_at": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="No strategy data for this session")
    return doc


# ── Stats ────────────────────────────────────────────────────────────────


@router.get("/stats")
async def radio_stats(
    year: int | None = Query(None),
):
    """Get signal distribution stats."""
    db = _get_db()
    match = {"signal_tags": {"$exists": True}}
    if year:
        match["year"] = year

    pipeline = [
        {"$match": match},
        {"$unwind": "$signal_tags"},
        {"$group": {"_id": "$signal_tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    tag_dist = {r["_id"]: r["count"] for r in db["team_radio_transcripts"].aggregate(pipeline)}

    # Urgency distribution
    urgency_pipeline = [
        {"$match": match},
        {"$group": {"_id": "$urgency_level", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    urgency_dist = {r["_id"]: r["count"] for r in db["team_radio_transcripts"].aggregate(urgency_pipeline)}

    total = db["team_radio_transcripts"].count_documents(match)

    return {
        "total_tagged": total,
        "signal_distribution": tag_dist,
        "urgency_distribution": urgency_dist,
        "year": year,
    }
