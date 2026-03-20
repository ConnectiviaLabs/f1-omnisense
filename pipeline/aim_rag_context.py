"""AiM RaceStudio 3 → RAG knowledge base integration.

Generates text chunks from AiM session data and embeds them into the
f1_knowledge collection for the OmniRAG chatbot.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _get_db():
    from pipeline.updater._db import get_db
    return get_db()


def build_aim_session_context(session_id: str) -> list[str]:
    """Generate 3-5 text chunks summarizing an AiM session for RAG."""
    db = _get_db()

    session = db["aim_sessions"].find_one({"session_id": session_id}, {"_id": 0})
    if not session:
        raise ValueError(f"Session not found: {session_id}")

    summary = session.get("summary", {})
    laps = session.get("laps", [])
    chunks = []

    # ── Chunk 1: Session overview ──
    best_lap = summary.get("best_lap_time_s", "N/A")
    best_lap_num = summary.get("best_lap_number", "N/A")
    duration = session.get("duration_s", 0)
    lap_times_str = ", ".join(
        f"Lap {l['lap_number']}: {l['lap_time_s']:.3f}s" for l in laps
    )
    chunks.append(
        f"AiM RaceStudio 3 session recorded on {session.get('date', 'unknown')} "
        f"at {session.get('track', 'unknown')} track. "
        f"Driver: {session.get('driver', 'unknown')}. "
        f"Vehicle: {session.get('vehicle', 'unknown')}. "
        f"ECU: {session.get('ecu', 'unknown')} (firmware {session.get('firmware', 'unknown')}). "
        f"Total laps: {session.get('lap_count', 0)}, session duration: {duration:.1f}s. "
        f"Best lap: Lap {best_lap_num} at {best_lap}s. "
        f"Lap times: {lap_times_str}."
    )

    # ── Chunk 2: Performance summary ──
    perf_lines = []
    speed = summary.get("max_GPS_Speed_kmh")
    if speed:
        perf_lines.append(f"Top speed: {speed:.1f} km/h (avg {summary.get('avg_GPS_Speed_kmh', 0):.1f} km/h)")
    throttle = summary.get("avg_Throttle_pct")
    if throttle:
        perf_lines.append(f"Average throttle: {throttle:.1f}%")
    fb = summary.get("max_FrontBrake_bar")
    if fb:
        perf_lines.append(f"Max front brake pressure: {fb:.1f} bar")
    rb = summary.get("max_RearBrake_bar")
    if rb:
        perf_lines.append(f"Max rear brake pressure: {rb:.1f} bar")
    lat_g = summary.get("max_LateralAcc_g")
    if lat_g:
        perf_lines.append(f"Max lateral G: {lat_g:.2f}g")
    inline_g = summary.get("max_InlineAcc_g")
    if inline_g:
        perf_lines.append(f"Max longitudinal G: {inline_g:.2f}g")
    if perf_lines:
        chunks.append(
            f"Performance summary for {session.get('driver', 'driver')} at {session.get('track', 'track')}: "
            + ". ".join(perf_lines) + "."
        )

    # ── Chunk 3: Thermal / reliability ──
    thermal_lines = []
    oil_t = summary.get("max_OilTemp_C")
    if oil_t:
        thermal_lines.append(f"Max oil temperature: {oil_t:.1f}°C (avg {summary.get('avg_OilTemp_C', 0):.1f}°C)")
    water_t = summary.get("max_WaterTemp_C")
    if water_t:
        thermal_lines.append(f"Max water temperature: {water_t:.1f}°C (avg {summary.get('avg_WaterTemp_C', 0):.1f}°C)")
    oil_p = summary.get("avg_OilPressure_bar")
    if oil_p:
        thermal_lines.append(f"Average oil pressure: {oil_p:.2f} bar")
    alerts = summary.get("alert_count", 0)
    thermal_lines.append(f"Total sensor alerts triggered: {alerts}")
    if thermal_lines:
        chunks.append(
            f"Thermal and reliability data for the session: "
            + ". ".join(thermal_lines) + "."
        )

    # ── Chunk 4: Dynamics ──
    dynamics_lines = []
    yaw = summary.get("max_YawRate_dps")
    if yaw:
        dynamics_lines.append(f"Max yaw rate: {yaw:.1f} deg/s")
    roll = summary.get("max_RollRate_dps")
    if roll:
        dynamics_lines.append(f"Max roll rate: {roll:.1f} deg/s")
    pitch = summary.get("max_PitchRate_dps")
    if pitch:
        dynamics_lines.append(f"Max pitch rate: {pitch:.1f} deg/s")
    vert_g = summary.get("max_VerticalAcc_g")
    if vert_g:
        dynamics_lines.append(f"Max vertical G: {vert_g:.2f}g")
    afr = summary.get("avg_AFR")
    if afr:
        dynamics_lines.append(f"Average AFR: {afr:.2f}")
    lam = summary.get("avg_Lambda")
    if lam:
        dynamics_lines.append(f"Average Lambda: {lam:.3f}")
    if dynamics_lines:
        chunks.append(
            f"Vehicle dynamics and engine data: "
            + ". ".join(dynamics_lines) + "."
        )

    return chunks


def push_aim_to_knowledge_base(session_id: str) -> int:
    """Embed AiM session chunks and insert into f1_knowledge vectorstore.

    Returns number of chunks inserted.
    """
    from langchain_core.documents import Document
    from pipeline.embeddings import NomicEmbedder
    from pipeline.vectorstore import get_vector_store

    chunks = build_aim_session_context(session_id)
    if not chunks:
        logger.warning("No chunks generated for %s", session_id)
        return 0

    session = _get_db()["aim_sessions"].find_one({"session_id": session_id}, {"_id": 0})

    # Build LangChain Document objects
    documents = []
    for i, text in enumerate(chunks):
        documents.append(Document(
            page_content=text,
            metadata={
                "category": "aim_racestudio3",
                "data_type": "telemetry_session",
                "_source": f"aim:{session_id}",
                "session_id": session_id,
                "driver": session.get("driver", "") if session else "",
                "track": session.get("track", "") if session else "",
                "date": session.get("date", "") if session else "",
                "chunk": i + 1,
                "total_chunks": len(chunks),
            },
        ))

    # Embed
    embedder = NomicEmbedder()
    texts = [doc.page_content for doc in documents]
    embeddings = embedder.embed_texts(texts)

    # Upsert to vectorstore
    vs = get_vector_store()
    count = vs.upsert_documents(documents, embeddings)
    logger.info("Pushed %d AiM chunks for %s to knowledge base", count, session_id)
    return count
