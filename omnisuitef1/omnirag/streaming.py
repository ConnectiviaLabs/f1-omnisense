"""Server-Sent Events (SSE) streaming for agent and RAG responses.

Adapted from cadAI agent_stream.py. Events emitted:
  - thinking:     {"message": "..."}
  - tool_start:   {"tool_name": "...", "arguments": {...}}
  - tool_result:  {"tool_name": "...", "result": {...}}
  - token:        {"text": "partial text"}
  - sources:      [{...}]
  - done:         {"session_id": "..."}
  - error:        {"message": "..."}

Usage:
    from omnirag.streaming import stream_agent_response

    @app.get("/agent/chat/stream")
    async def stream(message: str):
        return StreamingResponse(
            stream_agent_response(agent, message),
            media_type="text/event-stream",
        )
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


# ── SSE helpers ──────────────────────────────────────────────────────────────

def sse_event(event: str, data: Any) -> str:
    """Format a single SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# ── Streaming agent ─────────────────────────────────────────────────────────

def stream_agent_response(
    agent,
    message: str,
    session_id: Optional[str] = None,
    **kwargs,
) -> Generator[str, None, None]:
    """Generate SSE events for an agent chat response.

    The tool selection call is non-streaming.
    The final response call IS streaming (token-by-token).
    Falls back to non-streaming RAG if tools/Groq unavailable.

    Args:
        agent: RAGAgent instance.
        message: User message.
        session_id: Optional session ID for conversation continuity.
    """
    # Ensure session
    if session_id is None and agent._conversations:
        session_id = agent._conversations.create_session()
    session_id = session_id or ""

    # No tools or no Groq → fallback
    if not agent.has_tools:
        yield from _stream_fallback_rag(agent, message, session_id, **kwargs)
        return

    try:
        yield sse_event("thinking", {"message": "Analyzing your question..."})

        # Build messages
        messages = [{"role": "system", "content": agent._system_prompt}]
        if agent._conversations and session_id:
            history = agent._conversations.get_history(session_id)
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": message})

        # First call: tool selection (non-streaming)
        tools_schema = agent._registry.to_openai_schema()
        response = agent._client.chat.completions.create(
            model=agent._model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=2048,
        )

        choice = response.choices[0]

        if not choice.message.tool_calls:
            # No tools — stream the direct response
            stream = agent._client.chat.completions.create(
                model=agent._model,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                stream=True,
            )
            full_text = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_text += delta.content
                    yield sse_event("token", {"text": delta.content})

            agent._store_conversation(session_id, message, full_text)
            yield sse_event("done", {"session_id": session_id})
            return

        # Tool calls detected — dispatch
        tool_results = []
        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            yield sse_event("tool_start", {
                "tool_name": fn_name,
                "arguments": fn_args,
            })

            result = agent._registry.dispatch(fn_name, fn_args)
            tool_results.append({
                "tool_name": fn_name,
                "arguments": fn_args,
                "result": result,
            })

            yield sse_event("tool_result", {
                "tool_name": fn_name,
                "arguments": fn_args,
                "result": result,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

        # Stream final response after tools
        stream = agent._client.chat.completions.create(
            model=agent._model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
            stream=True,
        )

        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content
                yield sse_event("token", {"text": delta.content})

        # Emit sources from search-type tools
        sources = []
        for tr in tool_results:
            if isinstance(tr["result"], dict):
                sources.extend(tr["result"].get("sources", []))
        if sources:
            yield sse_event("sources", sources)

        agent._store_conversation(session_id, message, full_text)
        yield sse_event("done", {"session_id": session_id})

    except Exception as e:
        logger.error("Streaming agent error: %s", e)
        yield sse_event("error", {"message": str(e)})


def _stream_fallback_rag(
    agent,
    message: str,
    session_id: str,
    **kwargs,
) -> Generator[str, None, None]:
    """Fallback to non-streaming RAG when Groq is unavailable."""
    yield sse_event("thinking", {"message": "Searching knowledge base..."})

    try:
        if not agent._chain:
            yield sse_event("error", {"message": "No RAG chain configured"})
            return

        k = kwargs.get("k", 5)
        category = kwargs.get("category")
        response = agent._chain.ask(message, k=k, category=category)

        # Emit answer as single token (no streaming from LangChain)
        yield sse_event("token", {"text": response.answer})

        if response.sources:
            yield sse_event("sources", response.sources)

        agent._store_conversation(session_id, message, response.answer)
        yield sse_event("done", {"session_id": session_id})

    except Exception as e:
        yield sse_event("error", {"message": f"RAG fallback failed: {e}"})
