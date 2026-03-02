"""Multi-LLM provider auto-selection: Ollama → Groq → OpenAI → Anthropic.

Ported from cadAI qa_chain.py with provider priority chain.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Optional imports ─────────────────────────────────────────────────

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_anthropic import ChatAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class LLMConfig:
    provider: str = "auto"
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048


def _ollama_reachable(base_url: str = "") -> bool:
    """Check if Ollama is running locally."""
    url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def get_llm(provider: str = "auto", config: Optional[LLMConfig] = None):
    """Get the best available LLM.

    Priority: Ollama (local) → Groq → OpenAI → Anthropic.
    Returns a LangChain chat model instance.
    """
    cfg = config or LLMConfig(provider=provider)
    provider = cfg.provider

    if provider == "auto":
        # Try each in priority order
        if HAS_OLLAMA and _ollama_reachable():
            provider = "ollama"
        elif HAS_GROQ and os.getenv("GROQ_API_KEY"):
            provider = "groq"
        elif HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            raise RuntimeError(
                "No LLM provider available. Install one of: "
                "langchain-ollama, langchain-groq, langchain-openai, langchain-anthropic"
            )

    if provider == "ollama":
        if not HAS_OLLAMA:
            raise ImportError("langchain-ollama not installed")
        model = cfg.model or os.getenv("OLLAMA_MODEL", "gemma3:4b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info("Using Ollama: %s at %s", model, base_url)
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=cfg.temperature,
        )

    if provider == "groq":
        if not HAS_GROQ:
            raise ImportError("langchain-groq not installed")
        model = cfg.model or os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile")
        logger.info("Using Groq: %s", model)
        return ChatGroq(
            model=model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if provider == "openai":
        if not HAS_OPENAI:
            raise ImportError("langchain-openai not installed")
        model = cfg.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("Using OpenAI: %s", model)
        return ChatOpenAI(
            model=model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if provider == "anthropic":
        if not HAS_ANTHROPIC:
            raise ImportError("langchain-anthropic not installed")
        model = cfg.model or os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        logger.info("Using Anthropic: %s", model)
        return ChatAnthropic(
            model=model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


def generate(
    messages: List[Dict[str, str]],
    provider: str = "auto",
    config: Optional[LLMConfig] = None,
) -> tuple:
    """Generate a response from the best available LLM.

    Args:
        messages: List of {"role": str, "content": str} dicts.
        provider: "auto", "ollama", "groq", "openai", "anthropic".
        config: Optional LLMConfig overrides.

    Returns:
        (response_text, model_name) tuple.
    """
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    llm = get_llm(provider, config)

    lc_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    result = llm.invoke(lc_messages)
    model_name = getattr(llm, "model", getattr(llm, "model_name", str(provider)))

    return result.content, str(model_name)
