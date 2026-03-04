"""Domain-agnostic tool-calling agent for omniRAG.

Adapted from cadAI EngineeringAgent (backend/rag/agent.py) with all
domain-specific tools, system prompt, and route resolution removed.
Tools are registered at runtime via ToolRegistry, not hardcoded.

Architecture:
  - Tools are registered via ToolRegistry.register()
  - Two-call pattern: 1st LLM call selects tools, dispatch, 2nd call synthesizes
  - Falls back to plain RAG (qa_chain) when tools unavailable or fail
  - Uses Groq SDK directly for function calling (not LangChain)

Usage:
    from omnirag.agent import RAGAgent, ToolRegistry, ToolDefinition

    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="search_inventory",
        description="Search product inventory",
        parameters={"type": "object", "properties": {...}},
        handler=my_search_fn,
    ))

    agent = RAGAgent(chain=rag_chain, tool_registry=registry)
    response = agent.process_message("Find all widgets in stock")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Optional Groq import ────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
    HAS_GROQ_SDK = True
except ImportError:
    HAS_GROQ_SDK = False


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """A tool that can be called by the agent."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for parameters
    handler: Callable[..., Dict[str, Any]]

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI/Groq function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCallResult:
    """Result of a single tool execution."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
        }


@dataclass
class AgentResponse:
    """Response from the agent."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    model_used: str = ""
    tool_results: Optional[List[Dict[str, Any]]] = None
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "answer": self.answer,
            "sources": self.sources,
            "model_used": self.model_used,
            "session_id": self.session_id,
        }
        if self.tool_results is not None:
            d["tool_results"] = self.tool_results
        return d


# ── Tool Registry ────────────────────────────────────────────────────────────

class ToolRegistry:
    """Registry of tools available to the agent."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        """Register a tool definition."""
        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Dict[str, Any]],
    ):
        """Register a tool from individual arguments."""
        self.register(ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        ))

    def unregister(self, name: str):
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Return registered tool names."""
        return list(self._tools.keys())

    def to_openai_schema(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI/Groq function-calling schema."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if not tool:
            return {"status": "error", "message": f"Unknown tool: {name}"}
        try:
            return tool.handler(**arguments)
        except Exception as e:
            return {"status": "error", "message": str(e)}


# ── Agent ────────────────────────────────────────────────────────────────────

DEFAULT_AGENT_PROMPT = """You are a knowledgeable assistant with access to tools.
Use the provided tools when the user's question requires looking up data, performing actions, or searching documents.
If no tools are needed, answer directly from your knowledge."""


class RAGAgent:
    """Tool-calling agent with RAG fallback.

    Uses Groq SDK directly for function calling (the OpenAI-compatible
    tool_calls API). Falls back to plain RAG via RAGChain when Groq is
    unavailable or when no tools are registered.
    """

    def __init__(
        self,
        chain=None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        conversation_manager=None,
        llm_provider: str = "auto",
    ):
        """
        Args:
            chain: RAGChain for fallback Q&A.
            tool_registry: Registered tools. None or empty = RAG-only.
            system_prompt: Custom system prompt for the agent.
            conversation_manager: ConversationManager for session history.
            llm_provider: LLM provider for RAG fallback.
        """
        self._chain = chain
        self._registry = tool_registry or ToolRegistry()
        self._system_prompt = system_prompt or DEFAULT_AGENT_PROMPT
        self._conversations = conversation_manager
        self._llm_provider = llm_provider

        # Init Groq client
        self._client = None
        self._model = os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile")
        if HAS_GROQ_SDK and os.getenv("GROQ_API_KEY"):
            try:
                self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            except Exception:
                pass

    @property
    def has_tools(self) -> bool:
        """Whether tools are available (Groq + registered tools)."""
        return bool(self._client and self._registry.list_tools())

    def process_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Process a user message, dispatching tools as needed.

        Falls back to RAG when:
          - No tools registered
          - Groq SDK not available
          - Groq API call fails
        """
        # Create session if needed
        if session_id is None and self._conversations:
            session_id = self._conversations.create_session()
        session_id = session_id or ""

        # No tools available → straight to RAG
        if not self.has_tools:
            return self._fallback_rag(message, session_id, **kwargs)

        try:
            return self._run_with_tools(message, session_id, **kwargs)
        except Exception as e:
            logger.warning("Agent tool calling failed, falling back to RAG: %s", e)
            return self._fallback_rag(message, session_id, error=str(e), **kwargs)

    def _run_with_tools(self, message: str, session_id: str, **kwargs) -> AgentResponse:
        """Run message through Groq with function calling."""
        # Build messages
        messages = [{"role": "system", "content": self._system_prompt}]

        # Add conversation history
        if self._conversations and session_id:
            history = self._conversations.get_history(session_id)
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": message})

        # First call: tool selection (non-streaming)
        tools_schema = self._registry.to_openai_schema()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            if "tool_use_failed" in str(e):
                return self._run_without_tools(messages, session_id)
            raise

        choice = response.choices[0]

        # No tool calls → direct response
        if not choice.message.tool_calls:
            text = choice.message.content or ""
            self._store_conversation(session_id, message, text)
            return AgentResponse(
                answer=text,
                model_used=self._model,
                session_id=session_id,
            )

        # Dispatch tools
        tool_results = []
        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            result = self._registry.dispatch(fn_name, fn_args)
            tool_results.append(ToolCallResult(
                tool_name=fn_name,
                arguments=fn_args,
                result=result,
            ))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

        # Second call: synthesize final response
        final = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )
        final_text = final.choices[0].message.content or ""

        self._store_conversation(session_id, message, final_text)

        # Collect sources from any search-type tool results
        sources = []
        for tr in tool_results:
            if isinstance(tr.result, dict):
                sources.extend(tr.result.get("sources", []))

        return AgentResponse(
            answer=final_text,
            sources=sources,
            model_used=self._model,
            tool_results=[tr.to_dict() for tr in tool_results],
            session_id=session_id,
        )

    def _run_without_tools(self, messages: list, session_id: str) -> AgentResponse:
        """Retry as plain completion when tool calling fails."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )
        text = response.choices[0].message.content or ""
        user_msg = next(
            (m["content"] for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
            "",
        )
        self._store_conversation(session_id, user_msg, text)
        return AgentResponse(
            answer=text,
            model_used=self._model,
            session_id=session_id,
        )

    def _fallback_rag(
        self,
        message: str,
        session_id: str,
        error: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Fall back to plain RAG when tools are unavailable."""
        if not self._chain:
            answer = "No RAG chain configured and tool calling unavailable."
            if error:
                answer = f"Agent error: {error}. {answer}"
            return AgentResponse(answer=answer, session_id=session_id)

        try:
            k = kwargs.get("k", 5)
            category = kwargs.get("category")
            response = self._chain.ask(message, k=k, category=category)
            answer = response.answer
            if error:
                answer = f"[Using RAG fallback: {error}]\n\n{answer}"

            self._store_conversation(session_id, message, answer)

            return AgentResponse(
                answer=answer,
                sources=response.sources,
                model_used=response.model_used,
                session_id=session_id,
            )
        except Exception as e2:
            return AgentResponse(
                answer=f"Both agent and RAG failed. Agent: {error}. RAG: {e2}",
                session_id=session_id,
            )

    def _store_conversation(self, session_id: str, user_msg: str, assistant_msg: str):
        """Store user + assistant messages in conversation manager."""
        if self._conversations and session_id and user_msg:
            self._conversations.append(session_id, "user", user_msg)
            self._conversations.append(session_id, "assistant", assistant_msg)
