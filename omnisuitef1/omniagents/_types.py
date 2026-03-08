"""Shared type definitions for the omniagents framework."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class EventSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentEvent:
    """A single event published to the event bus."""
    topic: str
    agent: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    session_key: Optional[int] = None
    driver_number: Optional[int] = None
    severity: EventSeverity = EventSeverity.INFO

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class AgentState:
    """Persistent state for a single agent."""
    agent_id: str
    name: str
    status: AgentStatus = AgentStatus.IDLE
    last_run: Optional[float] = None
    last_session_key: Optional[int] = None
    events_published: int = 0
    events_consumed: int = 0
    last_error: Optional[str] = None
    last_output: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d
