"""F1Agent base class — foundation for all autonomous race agents.

Each agent inherits this and implements:
  - run_analysis(session_key, driver_number) — main analysis entrypoint
  - on_event(topic, event) — react to events from other agents

The base class provides:
  - publish() — emit events to the bus
  - reason() — call Groq LLM for narrative synthesis
  - state management — persist to MongoDB
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from omniagents._types import AgentEvent, AgentState, AgentStatus, EventSeverity
from omniagents.bus import EventBus

logger = logging.getLogger(__name__)

# ── Groq client (lazy init) ────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
    _HAS_GROQ = True
except ImportError:
    _HAS_GROQ = False

_GROQ_MODEL = os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile")

# ── RAG deep search (lazy singleton) ─────────────────────────────────────

_retriever = None
_retriever_init_failed = False


def _get_retriever():
    """Lazy singleton: BGE embedder + vectorstore → RAGRetriever."""
    global _retriever, _retriever_init_failed
    if _retriever is not None:
        return _retriever
    if _retriever_init_failed:
        return None
    try:
        from omnidoc.embedder import get_embedder
        from omnirag.vectorstore import get_vectorstore
        from omnirag.retriever import RAGRetriever

        embedder = get_embedder(enable_bge=True, enable_clip=False)
        store = get_vectorstore()
        _retriever = RAGRetriever(store, embedder.embed_query)
        logger.info("RAG deep search retriever initialized")
        return _retriever
    except Exception:
        _retriever_init_failed = True
        logger.warning("RAG deep search unavailable — missing dependencies or vectorstore")
        return None


class F1Agent(ABC):
    """Base class for all F1 autonomous agents."""

    name: str = "base_agent"
    description: str = ""
    subscriptions: List[str] = []   # topics this agent listens to
    publications: List[str] = []    # topics this agent publishes to

    def __init__(self, bus: EventBus, db=None):
        self._bus = bus
        self._db = db
        self._groq = Groq() if _HAS_GROQ else None
        self._state = AgentState(agent_id=self.name, name=self.name)
        self._deep_search_override = False

        # Wire subscriptions
        for topic in self.subscriptions:
            self._bus.subscribe(topic, self._handle_event)

    # ── Abstract methods ────────────────────────────────────────────────────

    @abstractmethod
    async def run_analysis(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Main analysis entrypoint. Load data, process, publish findings."""
        ...

    async def on_event(self, topic: str, event: Dict[str, Any]):
        """React to an event from another agent. Override in subclass."""
        pass

    # ── Event handling ──────────────────────────────────────────────────────

    async def _handle_event(self, topic: str, event: Dict[str, Any]):
        """Internal handler that wraps on_event with state tracking."""
        self._state.events_consumed += 1
        try:
            await self.on_event(topic, event)
        except Exception as e:
            logger.exception("[%s] Error handling event on topic %s", self.name, topic)

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        severity: EventSeverity = EventSeverity.INFO,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None,
    ):
        """Publish an event to the bus."""
        event = AgentEvent(
            topic=topic,
            agent=self.name,
            payload=payload,
            session_key=session_key,
            driver_number=driver_number,
            severity=severity,
        )
        self._state.events_published += 1
        await self._bus.publish(event)

    # ── LLM reasoning ──────────────────────────────────────────────────────

    async def reason(self, prompt: str, data_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Call Groq LLM for narrative synthesis.

        Returns LLM response text, or None if Groq is unavailable.
        Agents should degrade gracefully when this returns None.
        """
        if not self._groq:
            logger.warning("[%s] Groq unavailable — skipping LLM reasoning", self.name)
            return None

        system_msg = (
            f"You are {self.name}, an autonomous F1 race intelligence agent "
            f"in the McLaren Intelligence Platform. {self.description} "
            "Provide concise, technical analysis in the voice of a race engineer. "
            "Use specific numbers and data points. No speculation — only evidence-based insights."
        )

        user_msg = prompt
        if data_context:
            user_msg += f"\n\nData:\n```json\n{_truncate_json(data_context)}\n```"

        try:
            response = await asyncio.to_thread(
                self._groq.chat.completions.create,
                model=_GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception:
            logger.exception("[%s] Groq reasoning call failed", self.name)
            return None

    # ── RAG deep search ─────────────────────────────────────────────────

    async def deep_search(self, query: str, k: int = 5) -> list:
        """Vector search against f1_knowledge. Returns List[SearchResult] or []."""
        retriever = _get_retriever()
        if retriever is None:
            return []
        try:
            return await asyncio.to_thread(retriever.search_enhanced, query, k=k)
        except Exception:
            logger.exception("[%s] deep_search failed for query: %s", self.name, query[:80])
            return []

    async def deep_search_context(self, query: str, k: int = 5) -> str:
        """Vector search returning formatted markdown for prompt injection."""
        retriever = _get_retriever()
        if retriever is None:
            return "No relevant context found."
        try:
            return await asyncio.to_thread(retriever.get_relevant_context, query, k=k)
        except Exception:
            logger.exception("[%s] deep_search_context failed for query: %s", self.name, query[:80])
            return "No relevant context found."

    # ── State management ───────────────────────────────────────────────────

    def set_status(self, status: AgentStatus, error: Optional[str] = None):
        self._state.status = status
        if error:
            self._state.last_error = error
        if status == AgentStatus.RUNNING:
            self._state.last_run = time.time()

    def set_output(self, output: Dict[str, Any], session_key: Optional[int] = None):
        self._state.last_output = output
        if session_key:
            self._state.last_session_key = session_key

    async def save_state(self):
        """Persist agent state to MongoDB."""
        if self._db is not None:
            try:
                self._db["agent_state"].update_one(
                    {"agent_id": self.name},
                    {"$set": self._state.to_dict()},
                    upsert=True,
                )
            except Exception:
                logger.exception("[%s] Failed to save state", self.name)

    async def save_output(self, collection: str, doc: Dict[str, Any]):
        """Persist analysis output to a specific MongoDB collection."""
        if self._db is not None:
            try:
                doc["_agent"] = self.name
                doc["_timestamp"] = time.time()
                self._db[collection].insert_one(doc)
            except Exception:
                logger.exception("[%s] Failed to save output to %s", self.name, collection)

    @property
    def state(self) -> AgentState:
        return self._state

    # ── Run wrapper ────────────────────────────────────────────────────────

    async def execute(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Wrapper around run_analysis with state management."""
        self.set_status(AgentStatus.RUNNING)
        await self.save_state()

        # Publish start event
        await self.publish(
            topic=f"agent:{self.name}:status",
            payload={"status": "running", "session_key": session_key, "driver_number": driver_number},
            session_key=session_key,
            driver_number=driver_number,
        )

        try:
            result = await self.run_analysis(session_key, driver_number, year)
            self.set_status(AgentStatus.COMPLETED)
            self.set_output(result, session_key)
            await self.save_state()

            await self.publish(
                topic=f"agent:{self.name}:status",
                payload={"status": "completed", "session_key": session_key, "summary": _summary(result)},
                session_key=session_key,
                driver_number=driver_number,
            )
            return result

        except Exception as e:
            self.set_status(AgentStatus.ERROR, error=str(e))
            await self.save_state()

            await self.publish(
                topic=f"agent:{self.name}:status",
                payload={"status": "error", "error": str(e)},
                severity=EventSeverity.HIGH,
                session_key=session_key,
                driver_number=driver_number,
            )
            raise


# ── Helpers ─────────────────────────────────────────────────────────────────

def _truncate_json(data: Any, max_len: int = 3000) -> str:
    import json
    text = json.dumps(data, default=str, indent=2)
    if len(text) > max_len:
        return text[:max_len] + "\n... [truncated]"
    return text


def _summary(result: Dict[str, Any]) -> str:
    """Extract a short summary string from agent output."""
    if "summary" in result:
        return str(result["summary"])[:200]
    if "insight" in result:
        return str(result["insight"])[:200]
    return f"{len(result)} keys"
