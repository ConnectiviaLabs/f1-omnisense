"""AiM RaceStudio 3 API Router.

Endpoints:
    POST /api/aim/upload              — upload .xrk/.gpk/.rrk/.drk files
    GET  /api/aim/sessions            — list sessions
    GET  /api/aim/session/{id}        — session metadata + laps + summary
    GET  /api/aim/telemetry/{id}      — telemetry data (downsampled)
    GET  /api/aim/track/{id}          — track coordinates for map
    GET  /api/aim/laps/{id}           — lap times + per-lap summaries
    GET  /api/aim/health/{id}         — 7-system health assessment
    GET  /api/aim/anomaly/{id}        — per-lap per-system anomaly scores
    GET  /api/aim/compare             — session-over-session comparison
    DELETE /api/aim/session/{id}      — remove session + all data
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

import math

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/aim", tags=["AiM RaceStudio"])


def _get_db():
    from pipeline.updater._db import get_db
    return get_db()


def _scrub_nan(doc: dict) -> dict:
    """Remove NaN/Inf values from a MongoDB doc — they aren't valid JSON."""
    return {
        k: v for k, v in doc.items()
        if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    }


# ── Upload ───────────────────────────────────────────────────────────────


@router.post("/upload")
async def upload_aim_files(
    xrk: UploadFile = File(...),
    gpk: UploadFile | None = File(None),
    rrk: UploadFile | None = File(None),
    drk: UploadFile | None = File(None),
):
    """Upload AiM binary files, convert and ingest."""
    from pipeline.aim_ingest import ingest_xrk_session

    MAX_UPLOAD_BYTES = 500_000_000  # 500 MB

    with tempfile.TemporaryDirectory(prefix="aim_upload_") as tmpdir:
        tmp = Path(tmpdir)

        # Sanitize filenames to prevent path traversal
        safe_name = Path(xrk.filename or "upload.xrk").name
        data = await xrk.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 500 MB)")
        xrk_path = tmp / safe_name
        xrk_path.write_bytes(data)

        gpk_path = None
        if gpk and gpk.filename:
            gpk_path = tmp / Path(gpk.filename).name
            gpk_path.write_bytes(await gpk.read())

        rrk_path = None
        if rrk and rrk.filename:
            rrk_path = tmp / Path(rrk.filename).name
            rrk_path.write_bytes(await rrk.read())

        drk_path = None
        if drk and drk.filename:
            drk_path = tmp / Path(drk.filename).name
            drk_path.write_bytes(await drk.read())

        try:
            session_id = ingest_xrk_session(xrk_path, gpk_path, rrk_path, drk_path)
        except Exception as e:
            logger.error("Upload ingest failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    return {"session_id": session_id, "status": "ingested"}


# ── Sessions ─────────────────────────────────────────────────────────────


@router.get("/sessions")
async def list_sessions(
    driver: str | None = Query(None),
    track: str | None = Query(None),
):
    """List all AiM sessions, optionally filtered."""
    db = _get_db()
    query: dict = {}
    if driver:
        query["driver"] = {"$regex": re.escape(driver), "$options": "i"}
    if track:
        query["track"] = {"$regex": re.escape(track), "$options": "i"}

    docs = list(db["aim_sessions"].find(
        query,
        {"_id": 0, "session_id": 1, "driver": 1, "track": 1, "date": 1,
         "time": 1, "vehicle": 1, "lap_count": 1, "duration_s": 1,
         "uploaded_at": 1, "summary.best_lap_time_s": 1},
    ).sort("uploaded_at", -1))
    return docs


# ── Session detail ───────────────────────────────────────────────────────


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Full session metadata + laps + summary."""
    db = _get_db()
    doc = db["aim_sessions"].find_one({"session_id": session_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return doc


# ── Telemetry ────────────────────────────────────────────────────────────


@router.get("/telemetry/{session_id}")
async def get_telemetry(
    session_id: str,
    lap: int | None = Query(None),
    channels: str | None = Query(None, description="Comma-separated channel names"),
    limit: int = Query(5000, ge=100, le=50000),
):
    """Telemetry data, downsampled to `limit` points. Supports channel filtering."""
    db = _get_db()

    query: dict = {"session_id": session_id}
    if lap is not None:
        query["lap"] = lap

    # Projection
    projection: dict = {"_id": 0}
    if channels:
        channel_list = [c.strip() for c in channels.split(",")]
        for ch in channel_list:
            projection[ch] = 1
        # Always include these for context
        projection["session_id"] = 1
        projection["timestamp_ms"] = 1
        projection["time_s"] = 1
        projection["lap"] = 1

    total = db["aim_telemetry"].count_documents(query)
    if total == 0:
        raise HTTPException(status_code=404, detail=f"No telemetry for {session_id}")

    # Downsample: skip every Nth row
    skip_n = max(1, total // limit)

    cursor = db["aim_telemetry"].find(query, projection).sort("timestamp_ms", 1)
    docs = []
    for i, doc in enumerate(cursor):
        if i % skip_n == 0:
            docs.append(_scrub_nan(doc))
        if len(docs) >= limit:
            break

    return {
        "session_id": session_id,
        "total_samples": total,
        "returned": len(docs),
        "downsample_factor": skip_n,
        "data": docs,
    }


# ── Track ────────────────────────────────────────────────────────────────


@router.get("/track/{session_id}")
async def get_track(
    session_id: str,
    lap: int | None = Query(None),
):
    """Track coordinates for map visualization."""
    db = _get_db()

    query: dict = {"session_id": session_id}
    if lap is not None:
        query["lap"] = lap

    # Get GPS raw data (has lat/lon)
    gps_docs = [_scrub_nan(d) for d in db["aim_gps_raw"].find(
        query,
        {"_id": 0, "GPS_Lat": 1, "GPS_Lon": 1, "GPS_Speed_kmh": 1,
         "GPS_Altitude_m": 1, "time_s": 1},
    ).sort("time_s", 1)]

    # Also get track XY data
    tel_query: dict = {"session_id": session_id}
    if lap is not None:
        tel_query["lap"] = lap

    track_docs = [_scrub_nan(d) for d in db["aim_telemetry"].find(
        tel_query,
        {"_id": 0, "X_m": 1, "Y_m": 1, "Z_m": 1, "GPS_Speed_kmh": 1,
         "GPS_Lat": 1, "GPS_Lon": 1, "time_s": 1, "lap": 1, "Distance_m": 1,
         "FrontBrake_bar": 1, "Throttle_pct": 1, "LateralAcc_g": 1, "Gear": 1},
    ).sort("time_s", 1)]

    # Downsample track to ~2K points
    max_pts = 2000
    if len(track_docs) > max_pts:
        step = len(track_docs) // max_pts
        track_docs = track_docs[::step]

    if not gps_docs and not track_docs:
        raise HTTPException(status_code=404, detail=f"No track data for {session_id}")

    return {
        "session_id": session_id,
        "gps": gps_docs,
        "track_xy": track_docs,
    }


# ── Laps ─────────────────────────────────────────────────────────────────


@router.get("/laps/{session_id}")
async def get_laps(session_id: str):
    """Lap times + per-lap summaries."""
    db = _get_db()

    session = db["aim_sessions"].find_one({"session_id": session_id}, {"_id": 0, "laps": 1})
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    summaries = [_scrub_nan(d) for d in db["aim_lap_summary"].find(
        {"session_id": session_id},
        {"_id": 0},
    ).sort("lap_number", 1)]

    return {
        "session_id": session_id,
        "laps": session.get("laps", []),
        "summaries": summaries,
    }


# ── Health ───────────────────────────────────────────────────────────────


@router.get("/health/{session_id}")
async def get_health(session_id: str):
    """7-system health assessment."""
    from pipeline.aim_anomaly import run_aim_anomaly
    try:
        result = run_aim_anomaly(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Health assessment failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "session_id": result["session_id"],
        "driver": result["driver"],
        "track": result["track"],
        "overall_health": result["overall_health"],
        "severity": result["severity"],
        "systems": [
            {
                "system": s["system"],
                "health_pct": s["health_pct"],
                "severity": s["severity"],
                "action": s["action"],
            }
            for s in result["systems"]
        ],
    }


# ── Anomaly (detailed) ──────────────────────────────────────────────────


@router.get("/anomaly/{session_id}")
async def get_anomaly(session_id: str):
    """Per-lap per-system anomaly scores with SHAP features."""
    from pipeline.aim_anomaly import run_aim_anomaly
    try:
        return run_aim_anomaly(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Anomaly scoring failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Track Anomalies (intelligence layer) ──────────────────────────────


@router.get("/track-anomalies/{session_id}")
async def get_track_anomalies(session_id: str):
    """Track-level intelligence: anomaly zones, point events, degradation, lap deltas."""
    from pipeline.aim_anomaly import run_track_anomalies
    try:
        return run_track_anomalies(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Track anomaly analysis failed: %s", e)
        raise HTTPException(status_code=500, detail="Analysis failed")


# ── Compare ──────────────────────────────────────────────────────────────


@router.get("/compare")
async def compare_sessions(
    sessions: str = Query(..., description="Comma-separated session IDs"),
):
    """Lap-by-lap comparison of two or more sessions."""
    db = _get_db()
    session_ids = [s.strip() for s in sessions.split(",")]

    if len(session_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 session IDs")

    comparison = []
    for sid in session_ids:
        session = db["aim_sessions"].find_one({"session_id": sid}, {"_id": 0})
        if not session:
            continue

        summaries = list(db["aim_lap_summary"].find(
            {"session_id": sid},
            {"_id": 0},
        ).sort("lap_number", 1))

        comparison.append({
            "session_id": sid,
            "driver": session.get("driver", ""),
            "track": session.get("track", ""),
            "date": session.get("date", ""),
            "vehicle": session.get("vehicle", ""),
            "laps": session.get("laps", []),
            "summary": session.get("summary", {}),
            "lap_summaries": summaries,
        })

    return {"sessions": comparison}


# ── Delete ───────────────────────────────────────────────────────────────


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Remove a session and all associated data."""
    db = _get_db()

    session = db["aim_sessions"].find_one({"session_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    counts = {
        "aim_sessions": db["aim_sessions"].delete_many({"session_id": session_id}).deleted_count,
        "aim_telemetry": db["aim_telemetry"].delete_many({"session_id": session_id}).deleted_count,
        "aim_gps_raw": db["aim_gps_raw"].delete_many({"session_id": session_id}).deleted_count,
        "aim_lap_summary": db["aim_lap_summary"].delete_many({"session_id": session_id}).deleted_count,
    }
    logger.info("Deleted session %s: %s", session_id, counts)
    return {"session_id": session_id, "deleted": counts}
