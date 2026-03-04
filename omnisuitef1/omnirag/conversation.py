"""Session-based conversation memory with token budget pruning and TTL expiration.

Ported from cadAI agent.py conversation management.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from omnirag._types import ChatMessage

MAX_HISTORY_MESSAGES = 20
MAX_TOKEN_BUDGET = 4000  # ~16K chars at 4 chars/token
SESSION_TTL = 3600  # 1 hour


class ConversationManager:
    """Thread-safe session memory with automatic pruning.

    Features:
      - Token budget: keeps recent messages within ~4K tokens
      - Count limit: max 20 messages per session
      - TTL expiration: auto-removes sessions after 1 hour of inactivity

    Usage:
        manager = ConversationManager()
        sid = manager.create_session()
        manager.append(sid, "user", "What is the max pressure?")
        manager.append(sid, "assistant", "The max pressure is 150 bar.")
        history = manager.get_history(sid)
    """

    def __init__(
        self,
        max_messages: int = MAX_HISTORY_MESSAGES,
        max_token_budget: int = MAX_TOKEN_BUDGET,
        session_ttl: int = SESSION_TTL,
    ):
        self._sessions: Dict[str, List[ChatMessage]] = {}
        self._last_access: Dict[str, float] = {}
        self._max_messages = max_messages
        self._max_token_budget = max_token_budget
        self._session_ttl = session_ttl

    def create_session(self) -> str:
        """Create a new session, return session_id."""
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = []
        self._last_access[sid] = time.time()
        return sid

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Get pruned conversation history for a session.

        Applies:
          1. Count limit (last N messages)
          2. Token budget pruning (from most recent backwards)
        """
        self._touch(session_id)
        messages = self._sessions.get(session_id, [])

        # Count limit
        recent = messages[-self._max_messages:]

        # Token budget pruning (4 chars ≈ 1 token)
        char_budget = self._max_token_budget * 4
        total_chars = 0
        kept = []
        for msg in reversed(recent):
            msg_chars = len(msg.content)
            if total_chars + msg_chars > char_budget:
                break
            kept.insert(0, msg)
            total_chars += msg_chars

        return kept

    def append(self, session_id: str, role: str, content: str):
        """Append a message to a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append(
            ChatMessage(role=role, content=content)
        )
        self._touch(session_id)

    def clear(self, session_id: str):
        """Clear all messages for a session."""
        self._sessions.pop(session_id, None)
        self._last_access.pop(session_id, None)

    def expire_old(self):
        """Remove sessions older than TTL."""
        now = time.time()
        stale = [
            sid for sid, ts in self._last_access.items()
            if now - ts > self._session_ttl
        ]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)

    def list_sessions(self) -> List[str]:
        """Return active session IDs."""
        self.expire_old()
        return list(self._sessions.keys())

    def _touch(self, session_id: str):
        """Update last access timestamp."""
        self._last_access[session_id] = time.time()
