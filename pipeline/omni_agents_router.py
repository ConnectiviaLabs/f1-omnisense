"""FastAPI router for the F1 Agent System.

Endpoints:
  POST /api/omni/agents/run/{agent_name}   — trigger a single agent
  POST /api/omni/agents/run-all            — trigger full agent chain
  GET  /api/omni/agents/status             — all agent states
  GET  /api/omni/agents/events             — event history from MongoDB
  GET  /api/omni/agents/stream             — SSE real-time event feed
  GET  /api/omni/agents/{name}/output/{sk} — stored analysis output
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/omni/agents", tags=["agents"])

# ── Lazy init ───────────────────────────────────────────────────────────────

_registry = None
_bus = None
_db = None


def _ensure_init():
    """Lazily initialize agents, bus, and registry on first use."""
    global _registry, _bus, _db

    if _registry is not None:
        return

    from omniagents.bus import event_bus
    from omniagents.registry import AgentRegistry
    from omniagents.agents.telemetry_anomaly import TelemetryAnomalyAgent
    from omniagents.agents.weather_adapt import WeatherAdaptAgent
    from omniagents.agents.pit_window import PitWindowAgent
    from omniagents.agents.knowledge_conv import KnowledgeConvergenceAgent
    from omniagents.agents.visual_inspect import VisualInspectionAgent
    from omniagents.agents.predictive_maintenance import PredictiveMaintenanceAgent
    from omniagents.agents.trident_insight import TridentInsightAgent

    # Get MongoDB database
    try:
        from pipeline.chat_server import get_data_db
        _db = get_data_db()
    except Exception:
        logger.warning("Could not get MongoDB database — agents will run without persistence")
        _db = None

    event_bus.set_db(_db)
    _bus = event_bus

    _registry = AgentRegistry(bus=_bus, db=_db)

    # Register all 7 agents
    _registry.register(TelemetryAnomalyAgent(bus=_bus, db=_db))
    _registry.register(WeatherAdaptAgent(bus=_bus, db=_db))
    _registry.register(PitWindowAgent(bus=_bus, db=_db))
    _registry.register(KnowledgeConvergenceAgent(bus=_bus, db=_db))
    _registry.register(VisualInspectionAgent(bus=_bus, db=_db))
    _registry.register(PredictiveMaintenanceAgent(bus=_bus, db=_db))
    _registry.register(TridentInsightAgent(bus=_bus, db=_db))

    logger.info("Agent system initialized: %s", _registry.agent_names)


# ── Request models ──────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    session_key: int
    driver_number: Optional[int] = None
    year: Optional[int] = None


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/run/{agent_name}")
async def run_agent(agent_name: str, req: RunRequest):
    """Trigger a single agent's analysis."""
    _ensure_init()

    agent = _registry.get(agent_name)
    if agent is None:
        raise HTTPException(404, f"Agent '{agent_name}' not found. Available: {_registry.agent_names}")

    # Run in background so endpoint returns immediately
    async def _run():
        try:
            await agent.execute(req.session_key, req.driver_number, req.year)
        except Exception:
            logger.exception("Agent %s failed", agent_name)

    asyncio.create_task(_run())

    return {
        "status": "started",
        "agent": agent_name,
        "session_key": req.session_key,
        "driver_number": req.driver_number,
    }


@router.post("/run-all")
async def run_all_agents(req: RunRequest):
    """Trigger the full 5-agent chain in sequence."""
    _ensure_init()

    async def _run():
        try:
            await _registry.run_all(req.session_key, req.driver_number, req.year)
        except Exception:
            logger.exception("Agent chain failed")

    asyncio.create_task(_run())

    return {
        "status": "started",
        "agents": _registry.agent_names,
        "session_key": req.session_key,
        "driver_number": req.driver_number,
    }


@router.get("/status")
async def get_status():
    """Return status of all registered agents."""
    _ensure_init()
    return {
        "agents": _registry.list_agents(),
        "bus": {
            "event_count": _bus.event_count if _bus else 0,
            "subscriber_count": _bus.subscriber_count if _bus else 0,
            "sse_listeners": len(_bus._sse_queues) if _bus else 0,
        },
    }


@router.get("/events")
async def get_events(
    session_key: Optional[int] = Query(None),
    agent: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    """Return event history from MongoDB."""
    _ensure_init()

    if _db is None:
        return {"events": [], "message": "No database connection"}

    query = {}
    if session_key:
        query["session_key"] = session_key
    if agent:
        query["agent"] = agent

    docs = list(
        _db["agent_events"]
        .find(query, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return {"events": docs, "count": len(docs)}


@router.get("/stream")
async def stream_events():
    """SSE endpoint — streams agent events in real-time."""
    _ensure_init()

    from omnirag.streaming import sse_event

    queue = _bus.create_sse_queue()

    async def event_generator():
        try:
            # Send initial connection event
            yield sse_event("connected", {"message": "Agent event stream connected", "agents": _registry.agent_names})

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield sse_event("agent_event", event)
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield sse_event("keepalive", {"ts": __import__("time").time()})
        except asyncio.CancelledError:
            pass
        finally:
            _bus.remove_sse_queue(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{agent_name}/output/{session_key}")
async def get_agent_output(agent_name: str, session_key: int):
    """Return stored output for a specific agent + session."""
    _ensure_init()

    if _db is None:
        raise HTTPException(503, "No database connection")

    # Map agent names to output collections
    collection_map = {
        "telemetry_anomaly": "agent_telemetry_anomalies",
        "weather_adapt": "agent_weather_alerts",
        "pit_window": "agent_pit_windows",
        "knowledge_convergence": "agent_knowledge_fused",
        "visual_inspect": "agent_visual_incidents",
        "predictive_maintenance": "agent_predictive_maintenance",
        "trident_insight": "trident_reports_history",
    }

    coll_name = collection_map.get(agent_name)
    if not coll_name:
        raise HTTPException(404, f"Unknown agent: {agent_name}")

    doc = _db[coll_name].find_one(
        {"session_key": session_key},
        {"_id": 0},
        sort=[("_timestamp", -1)],
    )

    if not doc:
        raise HTTPException(404, f"No output for {agent_name} session {session_key}")

    return doc
