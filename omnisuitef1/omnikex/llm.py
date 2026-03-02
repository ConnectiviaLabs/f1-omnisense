"""Multi-LLM routing: Groq → Ollama → OpenAI → Anthropic.

Ported from DataSense KeXnStore.generate_text_unified and omnirag/llm.py.
Pure function — no MongoDB, Redis, or Flask dependencies.
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from omnikex._types import (
    KexLLMConfig,
    LLMProvider,
    TASK_TEMPERATURES,
    PERSONA_TEMPERATURES,
    DEFAULT_TEMPERATURE,
)
from omnikex.wise import LLM_GUARDRAIL

logger = logging.getLogger(__name__)

# ── Optional imports ─────────────────────────────────────────────────────────

try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

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


# ── Configuration ────────────────────────────────────────────────────────────

# Default models per provider and task type
_GROQ_MODELS = {
    "realtime": os.getenv("GROQ_KEX_MODEL", "qwen/qwen3-32b"),
    "anomaly": os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile"),
    "forecast": os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile"),
}
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
_OPENAI_MODEL = os.getenv("OPENAI_KEX_MODEL", "gpt-4.1")
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_KEX_MODEL", "claude-sonnet-4-20250514")


# ── Provider resolution ──────────────────────────────────────────────────────

def _ollama_reachable() -> bool:
    """Check if Ollama is running locally."""
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def resolve_provider(provider: LLMProvider = LLMProvider.AUTO) -> LLMProvider:
    """Resolve AUTO to the best available provider.

    Priority: Groq → Ollama → OpenAI → Anthropic.
    """
    if provider != LLMProvider.AUTO:
        return provider

    if HAS_GROQ and os.getenv("GROQ_API_KEY"):
        return LLMProvider.GROQ
    if HAS_OLLAMA and _ollama_reachable():
        return LLMProvider.OLLAMA
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        return LLMProvider.OPENAI
    if HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
        return LLMProvider.ANTHROPIC

    raise RuntimeError(
        "No LLM provider available. Install one of: "
        "langchain-groq, langchain-ollama, langchain-openai, langchain-anthropic "
        "and set the appropriate API key environment variable."
    )


# ── Text generation ──────────────────────────────────────────────────────────

def _strip_thinking_tokens(text: Optional[str]) -> Optional[str]:
    """Remove <think>...</think> reasoning blocks emitted by some models."""
    if not text:
        return text
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    return cleaned or text


def _resolve_temperature(config: KexLLMConfig) -> float:
    """Resolve temperature from config, task type, and persona."""
    if config.temperature is not None:
        return config.temperature
    # Persona-based temperature
    if config.persona:
        ctx_lower = config.persona.lower()
        for key, temp in PERSONA_TEMPERATURES.items():
            if key in ctx_lower:
                return temp
    # Task-based temperature
    return TASK_TEMPERATURES.get(config.task_type, DEFAULT_TEMPERATURE)


def generate(
    prompt: str,
    config: Optional[KexLLMConfig] = None,
) -> Tuple[str, str, str]:
    """Generate text from the best available LLM.

    Args:
        prompt: The prompt string.
        config: Optional LLM configuration.

    Returns:
        (response_text, model_name, provider_name) tuple.
    """
    cfg = config or KexLLMConfig()
    provider = resolve_provider(cfg.provider)
    temperature = _resolve_temperature(cfg)

    # Prepend guardrail
    full_prompt = LLM_GUARDRAIL + prompt

    start = time.time()

    if provider == LLMProvider.GROQ:
        return _generate_groq(full_prompt, cfg, temperature)
    elif provider == LLMProvider.OLLAMA:
        return _generate_ollama(full_prompt, cfg, temperature)
    elif provider == LLMProvider.OPENAI:
        return _generate_openai(full_prompt, cfg, temperature)
    elif provider == LLMProvider.ANTHROPIC:
        return _generate_anthropic(full_prompt, cfg, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _generate_groq(prompt: str, cfg: KexLLMConfig, temperature: float) -> Tuple[str, str, str]:
    """Generate via Groq API."""
    if not HAS_GROQ:
        raise ImportError("langchain-groq not installed")

    model = cfg.model or _GROQ_MODELS.get(cfg.task_type, _GROQ_MODELS["realtime"])
    logger.info("Using Groq: %s (task=%s, temp=%.2f)", model, cfg.task_type, temperature)

    llm = ChatGroq(model=model, temperature=temperature, max_tokens=cfg.max_tokens)

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=LLM_GUARDRAIL.strip()),
        HumanMessage(content=prompt),
    ]
    result = llm.invoke(messages)
    text = _strip_thinking_tokens(result.content)
    return text, model, "groq"


def _generate_ollama(prompt: str, cfg: KexLLMConfig, temperature: float) -> Tuple[str, str, str]:
    """Generate via local Ollama."""
    if not HAS_OLLAMA:
        raise ImportError("langchain-ollama not installed")

    model = cfg.model or _OLLAMA_MODEL
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    logger.info("Using Ollama: %s at %s (temp=%.2f)", model, base_url, temperature)

    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=LLM_GUARDRAIL.strip()),
        HumanMessage(content=prompt),
    ]
    result = llm.invoke(messages)
    text = _strip_thinking_tokens(result.content)
    return text, model, "ollama"


def _generate_openai(prompt: str, cfg: KexLLMConfig, temperature: float) -> Tuple[str, str, str]:
    """Generate via OpenAI API."""
    if not HAS_OPENAI:
        raise ImportError("langchain-openai not installed")

    model = cfg.model or _OPENAI_MODEL
    logger.info("Using OpenAI: %s (temp=%.2f)", model, temperature)

    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=cfg.max_tokens)

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=LLM_GUARDRAIL.strip()),
        HumanMessage(content=prompt),
    ]
    result = llm.invoke(messages)
    text = _strip_thinking_tokens(result.content)
    return text, model, "openai"


def _generate_anthropic(prompt: str, cfg: KexLLMConfig, temperature: float) -> Tuple[str, str, str]:
    """Generate via Anthropic API."""
    if not HAS_ANTHROPIC:
        raise ImportError("langchain-anthropic not installed")

    model = cfg.model or _ANTHROPIC_MODEL
    logger.info("Using Anthropic: %s (temp=%.2f)", model, temperature)

    llm = ChatAnthropic(model=model, temperature=temperature, max_tokens=cfg.max_tokens)

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=LLM_GUARDRAIL.strip()),
        HumanMessage(content=prompt),
    ]
    result = llm.invoke(messages)
    text = _strip_thinking_tokens(result.content)
    return text, model, "anthropic"


def list_available_providers() -> List[str]:
    """List LLM providers that are installed and configured."""
    available = []
    if HAS_GROQ and os.getenv("GROQ_API_KEY"):
        available.append("groq")
    if HAS_OLLAMA and _ollama_reachable():
        available.append("ollama")
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        available.append("openai")
    if HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    return available
