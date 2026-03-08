"""Agent registry — singleton that holds all agent instances."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from omniagents._types import AgentStatus
from omniagents.base import F1Agent
from omniagents.bus import EventBus

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Singleton registry for all F1 agents."""

    def __init__(self, bus: EventBus, db=None):
        self._agents: Dict[str, F1Agent] = {}
        self._bus = bus
        self._db = db

    def register(self, agent: F1Agent):
        """Register an agent instance."""
        self._agents[agent.name] = agent
        logger.info(
            "Registered agent: %s (subs=%s, pubs=%s)",
            agent.name, agent.subscriptions, agent.publications,
        )

    def get(self, name: str) -> Optional[F1Agent]:
        return self._agents.get(name)

    def list_agents(self) -> List[Dict]:
        """Return status summary for all agents."""
        return [
            {
                "name": a.name,
                "description": a.description,
                "status": a.state.status.value,
                "subscriptions": a.subscriptions,
                "publications": a.publications,
                "last_run": a.state.last_run,
                "last_session_key": a.state.last_session_key,
                "events_published": a.state.events_published,
                "events_consumed": a.state.events_consumed,
                "last_error": a.state.last_error,
            }
            for a in self._agents.values()
        ]

    @property
    def agent_names(self) -> List[str]:
        return list(self._agents.keys())

    async def run_all(self, session_key: int, driver_number: Optional[int] = None, year: Optional[int] = None):
        """Run all agents in sequence for a given session.

        Order: telemetry_anomaly → weather_adapt → pit_window → visual_inspect → predictive_maintenance → knowledge_convergence
        This ensures downstream agents receive events from upstream agents.
        """
        execution_order = [
            "telemetry_anomaly",
            "weather_adapt",
            "pit_window",
            "visual_inspect",
            "predictive_maintenance",
            "knowledge_convergence",
        ]
        results = {}
        for name in execution_order:
            agent = self._agents.get(name)
            if agent is None:
                continue
            try:
                results[name] = await agent.execute(session_key, driver_number, year)
            except Exception as e:
                logger.exception("Agent %s failed", name)
                results[name] = {"error": str(e)}
        return results
