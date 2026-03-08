"""OmniAgents — Autonomous F1 race intelligence agents.

5-agent system with in-process event bus, LLM reasoning, and SSE streaming.
"""

from omniagents._types import AgentEvent, AgentState, AgentStatus, EventSeverity
from omniagents.base import F1Agent
from omniagents.bus import EventBus, event_bus
from omniagents.registry import AgentRegistry

__all__ = [
    "AgentEvent",
    "AgentState",
    "AgentStatus",
    "EventSeverity",
    "F1Agent",
    "EventBus",
    "event_bus",
    "AgentRegistry",
]
