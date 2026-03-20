"""In-process async event bus for agent pub/sub communication.

No Redis dependency. Uses asyncio queues for subscriber delivery
and SSE fan-out. Persists every event to MongoDB for replay.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Optional

from omniagents._types import AgentEvent

logger = logging.getLogger(__name__)


class EventBus:
    """Async pub/sub event bus with SSE fan-out and MongoDB persistence."""

    def __init__(self, db=None):
        self._subscribers: Dict[str, List[Callable[..., Coroutine]]] = defaultdict(list)
        self._sse_queues: List[asyncio.Queue] = []
        self._db = db
        self._event_count = 0

    def set_db(self, db):
        """Set MongoDB database reference (deferred init)."""
        self._db = db

    def subscribe(self, topic: str, callback: Callable[..., Coroutine]):
        """Register async callback for a topic."""
        self._subscribers[topic].append(callback)
        logger.info("Subscribed %s to topic %s", getattr(callback, '__qualname__', callback), topic)

    async def publish(self, event: AgentEvent):
        """Publish event to all subscribers, SSE queues, and MongoDB."""
        self._event_count += 1
        event_dict = event.to_dict()

        # Fan out to topic subscribers
        for cb in self._subscribers.get(event.topic, []):
            try:
                asyncio.create_task(cb(event.topic, event_dict))
            except Exception:
                logger.exception("Subscriber callback failed for topic %s", event.topic)

        # Fan out to SSE listeners (all topics)
        dead_queues = []
        for q in self._sse_queues:
            try:
                q.put_nowait(event_dict)
            except asyncio.QueueFull:
                dead_queues.append(q)
        for q in dead_queues:
            self._sse_queues.remove(q)

        # Persist to MongoDB (fire-and-forget)
        if self._db is not None:
            try:
                self._db["agent_events"].insert_one(event_dict)
            except Exception:
                logger.exception("Failed to persist event to MongoDB")

    def create_sse_queue(self, maxsize: int = 500) -> asyncio.Queue:
        """Create a new SSE listener queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._sse_queues.append(q)
        return q

    def remove_sse_queue(self, q: asyncio.Queue):
        """Remove an SSE listener queue on disconnect."""
        try:
            self._sse_queues.remove(q)
        except ValueError:
            pass

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def subscriber_count(self) -> int:
        return sum(len(cbs) for cbs in self._subscribers.values())


# Module-level singleton
event_bus = EventBus()
