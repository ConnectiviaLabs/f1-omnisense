"""Multimodal vision clients: Groq (cloud) + Ollama (edge).

Unified interface for sending images + prompts to vision models,
with retry logic, timing, cost tracking, and JSON extraction.

Adapted from cadAI pdf_pipeline/models.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omnidoc.renderer import img_to_b64

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    model: str
    raw_text: str
    parsed: Optional[dict]
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float


# ── JSON extraction ───────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """Extract JSON object from model response text.

    Handles pure JSON, ```json blocks, embedded JSON, and truncated JSON.
    """
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: i + 1])
                    except json.JSONDecodeError:
                        break

        # Truncated — try adding closing braces
        fragment = text[start:]
        for _ in range(10):
            fragment += "}"
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                continue
        fragment = text[start:]
        for _ in range(10):
            fragment += "]}"
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                continue

    return None


# ── Groq Client (Cloud) ──────────────────────────────────────────────────

class GroqVisionClient:
    """Groq cloud vision client — Llama 4 Maverick."""

    MAX_IMAGES = 5
    MAX_RETRIES = 3
    COST_PER_M_INPUT = 0.50
    COST_PER_M_OUTPUT = 0.77
    DEFAULT_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

    def __init__(self, model_tag: str | None = None, name: str = "groq"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to .env or environment.\n"
                "Get a key at: https://console.groq.com/keys"
            )

        self.MODEL = model_tag or os.environ.get("GROQ_VISION_MODEL", self.DEFAULT_MODEL)
        self.name = name

        import groq
        self._client = groq.Groq(api_key=api_key)
        logger.info("Groq: %s (cloud)", self.MODEL)

    def analyze(
        self, images: list[Path], prompt: str,
        system: str = "", max_tokens: int = 4096,
    ) -> ModelResponse:
        image_b64_list = [img_to_b64(p) for p in images[:self.MAX_IMAGES]]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        content_parts = [{"type": "text", "text": prompt}]
        for b64 in image_b64_list:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        messages.append({"role": "user", "content": content_parts})

        for attempt in range(self.MAX_RETRIES):
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(
                    model=self.MODEL, messages=messages,
                    temperature=0.05, max_tokens=max_tokens,
                )
                latency = time.time() - t0
                raw = resp.choices[0].message.content or ""
                tokens_in = resp.usage.prompt_tokens if resp.usage else 0
                tokens_out = resp.usage.completion_tokens if resp.usage else 0
                cost = (
                    tokens_in * self.COST_PER_M_INPUT / 1_000_000
                    + tokens_out * self.COST_PER_M_OUTPUT / 1_000_000
                )
                return ModelResponse(
                    model=self.MODEL, raw_text=raw, parsed=extract_json(raw),
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    latency_s=round(latency, 2), cost_usd=round(cost, 6),
                )
            except Exception as e:
                latency = time.time() - t0
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    wait = 10 * (2 ** attempt)
                    logger.warning("[%s] Rate limited, waiting %ds...", self.name, wait)
                    time.sleep(wait)
                else:
                    logger.warning("[%s] Error: %s", self.name, err)
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(2)
                    else:
                        return ModelResponse(
                            model=self.MODEL, raw_text=f"ERROR: {err}",
                            parsed=None, tokens_in=0, tokens_out=0,
                            latency_s=round(latency, 2), cost_usd=0,
                        )

        return ModelResponse(
            model=self.MODEL, raw_text="ERROR: max retries exceeded",
            parsed=None, tokens_in=0, tokens_out=0, latency_s=0, cost_usd=0,
        )


# ── Ollama Client (Edge) ─────────────────────────────────────────────────

class OllamaClient:
    """Ollama local multimodal client."""

    MAX_IMAGES = 10
    MAX_RETRIES = 2
    OLLAMA_URL = "http://localhost:11434"

    def __init__(self, model_tag: str, name: str):
        self.MODEL = model_tag
        self.name = name

        import requests
        try:
            resp = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            if model_tag not in available:
                matches = [m for m in available if model_tag.split(":")[0] in m]
                if not matches:
                    raise ValueError(
                        f"Model '{model_tag}' not found in Ollama. "
                        f"Available: {available}. Run: ollama pull {model_tag}"
                    )
        except Exception as e:
            if "Connection" in str(e):
                raise ValueError("Ollama not running. Start with: ollama serve") from e
            raise

    def analyze(
        self, images: list[Path], prompt: str,
        system: str = "", max_tokens: int = 8192,
    ) -> ModelResponse:
        import requests

        image_b64_list = [img_to_b64(p) for p in images[:self.MAX_IMAGES]]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user_msg = {"role": "user", "content": prompt}
        if image_b64_list:
            user_msg["images"] = image_b64_list
        messages.append(user_msg)

        payload = {
            "model": self.MODEL, "messages": messages, "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.time()
                resp = requests.post(
                    f"{self.OLLAMA_URL}/api/chat", json=payload, timeout=300,
                )
                latency = time.time() - t0

                if resp.status_code != 200:
                    err = resp.text[:200]
                    logger.warning("[%s] HTTP %d: %s", self.name, resp.status_code, err)
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(2)
                        continue
                    return ModelResponse(
                        model=self.MODEL,
                        raw_text=f"ERROR: HTTP {resp.status_code}: {err}",
                        parsed=None, tokens_in=0, tokens_out=0,
                        latency_s=round(latency, 2), cost_usd=0,
                    )

                data = resp.json()
                raw = data.get("message", {}).get("content", "")
                tokens_in = data.get("prompt_eval_count", 0)
                tokens_out = data.get("eval_count", 0)

                return ModelResponse(
                    model=self.MODEL, raw_text=raw, parsed=extract_json(raw),
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    latency_s=round(latency, 2), cost_usd=0,
                )

            except Exception as e:
                logger.warning("[%s] Error: %s", self.name, e)
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    return ModelResponse(
                        model=self.MODEL, raw_text=f"ERROR: {e}",
                        parsed=None, tokens_in=0, tokens_out=0,
                        latency_s=0, cost_usd=0,
                    )

        return ModelResponse(
            model=self.MODEL, raw_text="ERROR: max retries exceeded",
            parsed=None, tokens_in=0, tokens_out=0, latency_s=0, cost_usd=0,
        )


# ── Pre-configured models ─────────────────────────────────────────────────

class GemmaClient(OllamaClient):
    def __init__(self):
        tag = os.environ.get("GEMMA_MODEL_TAG", "gemma3:4b")
        super().__init__(tag, "gemma")


class QwenVLClient(OllamaClient):
    def __init__(self):
        tag = os.environ.get("QWEN_MODEL_TAG", "qwen3-vl:8b")
        super().__init__(tag, "qwen")
