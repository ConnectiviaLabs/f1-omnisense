"""Multimodal vision clients: Ollama (edge) + Groq (cloud).

Provides a unified interface for sending images + prompts to vision models,
with retry logic, timing, cost tracking, and JSON extraction.

Edge mode:  gemma3 + qwen3-vl via local Ollama (free, consensus)
Cloud mode: Llama 4 Maverick via Groq API (paid, higher quality)
"""

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .renderer import img_to_b64

# ── Response dataclass ───────────────────────────────────────────────────

@dataclass
class ModelResponse:
    model: str
    raw_text: str
    parsed: dict | None
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float  # 0 for local Ollama, computed for Groq


# ── JSON extraction ──────────────────────────────────────────────────────

def extract_json(text: str) -> dict | None:
    """Extract JSON object from model response text.

    Handles:
      - Pure JSON
      - JSON in ```json ... ``` code blocks
      - JSON embedded in surrounding text
      - Truncated JSON (best-effort repair)
    """
    if not text:
        return None

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try code block extraction
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object boundaries
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
                        return json.loads(text[start : i + 1])
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
        # Try closing brackets + braces
        fragment = text[start:]
        for _ in range(10):
            fragment += "]}"
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                continue

    return None


# ── Ollama Client (Edge) ────────────────────────────────────────────────

class OllamaClient:
    """Generic Ollama multimodal client.

    Works with any Ollama model that supports vision (images parameter).
    """

    MAX_IMAGES = 10  # Ollama handles fewer images than cloud APIs
    MAX_RETRIES = 2
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    def __init__(self, model_tag: str, name: str):
        """
        Args:
            model_tag: Ollama model tag (e.g. 'gemma3:12b', 'qwen2.5-vl:7b')
            name: Short name for logging and file naming (e.g. 'gemma', 'qwen')
        """
        self.MODEL = model_tag
        self.name = name

        # Verify model is available
        import requests
        try:
            resp = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            if model_tag not in available:
                # Try partial match (e.g. 'gemma3:12b' matches 'gemma3:12b')
                matches = [m for m in available if model_tag.split(":")[0] in m]
                if not matches:
                    raise ValueError(
                        f"Model '{model_tag}' not found in Ollama. "
                        f"Available: {available}. Run: ollama pull {model_tag}"
                    )
        except Exception as e:
            if "Connection" in str(e):
                raise ValueError(
                    "Ollama not running. Start it with: ollama serve"
                ) from e
            raise

    def analyze(
        self,
        images: list[Path],
        prompt: str,
        system: str = "",
        max_tokens: int = 8192,
    ) -> ModelResponse:
        """Send images + prompt to Ollama and return structured response."""
        import requests

        # Encode images as base64
        image_b64_list = []
        for img_path in images[: self.MAX_IMAGES]:
            b64 = img_to_b64(img_path)
            image_b64_list.append(b64)

        # Build Ollama /api/chat request
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        user_msg = {"role": "user", "content": prompt}
        if image_b64_list:
            user_msg["images"] = image_b64_list
        messages.append(user_msg)

        payload = {
            "model": self.MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
            },
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.time()
                resp = requests.post(
                    f"{self.OLLAMA_URL}/api/chat",
                    json=payload,
                    timeout=300,  # 5 min timeout for vision models
                )
                latency = time.time() - t0

                if resp.status_code != 200:
                    err = resp.text[:200]
                    print(f"    [{self.name}] HTTP {resp.status_code}: {err}")
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

                # Token counts from Ollama response
                tokens_in = data.get("prompt_eval_count", 0)
                tokens_out = data.get("eval_count", 0)

                return ModelResponse(
                    model=self.MODEL,
                    raw_text=raw,
                    parsed=extract_json(raw),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_s=round(latency, 2),
                    cost_usd=0,  # Local = free
                )

            except Exception as e:
                err = str(e)
                print(f"    [{self.name}] Error: {err}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    return ModelResponse(
                        model=self.MODEL,
                        raw_text=f"ERROR: {err}",
                        parsed=None, tokens_in=0, tokens_out=0,
                        latency_s=0, cost_usd=0,
                    )

        return ModelResponse(
            model=self.MODEL, raw_text="ERROR: max retries exceeded",
            parsed=None, tokens_in=0, tokens_out=0, latency_s=0, cost_usd=0,
        )


# ── Groq Client (Cloud) ────────────────────────────────────────────────

class GroqVisionClient:
    """Groq cloud vision client — Llama 4 Maverick (128 experts).

    Matches the same analyze() interface as OllamaClient, returning
    ModelResponse with actual cost_usd computed from token counts.
    """

    MAX_IMAGES = 10
    MAX_RETRIES = 3

    # Groq Llama 4 Maverick pricing (per million tokens)
    COST_PER_M_INPUT = 0.50
    COST_PER_M_OUTPUT = 0.77

    DEFAULT_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

    def __init__(self, model_tag: str | None = None, name: str = "groq"):
        """
        Args:
            model_tag: Groq model ID. Defaults to GROQ_VISION_MODEL env var
                       or Llama 4 Maverick.
            name: Short name for logging and file naming.
        """
        # Load .env if available
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            else:
                load_dotenv()  # Try default locations
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

        print(f"  Groq: {self.MODEL} (cloud)")

    def analyze(
        self,
        images: list[Path],
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Send images + prompt to Groq vision API and return structured response."""
        # Encode images as base64
        image_b64_list = []
        for img_path in images[: self.MAX_IMAGES]:
            b64 = img_to_b64(img_path)
            image_b64_list.append(b64)

        # Build Groq messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        # User message with text + image_url content parts
        content_parts = [{"type": "text", "text": prompt}]
        for b64 in image_b64_list:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        messages.append({"role": "user", "content": content_parts})

        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.time()
                resp = self._client.chat.completions.create(
                    model=self.MODEL,
                    messages=messages,
                    temperature=0.05,
                    max_tokens=max_tokens,
                )
                latency = time.time() - t0

                raw = resp.choices[0].message.content or ""
                tokens_in = resp.usage.prompt_tokens if resp.usage else 0
                tokens_out = resp.usage.completion_tokens if resp.usage else 0

                # Compute cost
                cost = (
                    tokens_in * self.COST_PER_M_INPUT / 1_000_000
                    + tokens_out * self.COST_PER_M_OUTPUT / 1_000_000
                )

                return ModelResponse(
                    model=self.MODEL,
                    raw_text=raw,
                    parsed=extract_json(raw),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_s=round(latency, 2),
                    cost_usd=round(cost, 6),
                )

            except Exception as e:
                latency = time.time() - t0 if 't0' in dir() else 0
                err = str(e)

                # Rate limit handling with exponential backoff
                if "429" in err or "rate" in err.lower():
                    wait = 10 * (2 ** attempt)  # 10s, 20s, 40s
                    print(f"    [{self.name}] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    [{self.name}] Error: {err}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(2)
                    else:
                        return ModelResponse(
                            model=self.MODEL,
                            raw_text=f"ERROR: {err}",
                            parsed=None, tokens_in=0, tokens_out=0,
                            latency_s=round(latency, 2), cost_usd=0,
                        )

        return ModelResponse(
            model=self.MODEL, raw_text="ERROR: max retries exceeded",
            parsed=None, tokens_in=0, tokens_out=0, latency_s=0, cost_usd=0,
        )


# ── Pre-configured model classes ─────────────────────────────────────────

class GemmaClient(OllamaClient):
    """Alias for backward compat — now uses Qwen 3.5."""
    def __init__(self):
        tag = os.environ.get("OLLAMA_MODEL_TAG", "qwen3.5:9b")
        super().__init__(tag, "qwen3.5")


class QwenVLClient(OllamaClient):
    """Qwen 3.5 — unified vision + reasoning model."""
    def __init__(self):
        tag = os.environ.get("OLLAMA_MODEL_TAG", "qwen3.5:9b")
        super().__init__(tag, "qwen3.5")


# ── Multi-model analyzer ────────────────────────────────────────────────

class MultiModelAnalyzer:
    """Send the same prompt + images to multiple local models."""

    def __init__(self, model_names: list[str] | None = None):
        self.clients = {}
        names = model_names or ["gemma", "qwen"]

        for name in names:
            try:
                if name == "gemma":
                    self.clients["gemma"] = GemmaClient()
                elif name == "qwen":
                    self.clients["qwen"] = QwenVLClient()
                else:
                    raise ValueError(f"Unknown model: {name}")
            except ValueError as e:
                print(f"  [warn] Skipping {name}: {e}")

        if not self.clients:
            raise ValueError(
                "No Ollama models available. Ensure ollama is running and models are pulled:\n"
                "  ollama pull qwen3.5:9b"
            )

    def analyze(
        self,
        images: list[Path],
        prompt: str,
        system: str = "",
        delay_between: float = 1.0,
    ) -> dict[str, ModelResponse]:
        """Send to all configured models, return {name: response}."""
        results = {}
        for i, (name, client) in enumerate(self.clients.items()):
            if i > 0 and delay_between > 0:
                time.sleep(delay_between)
            results[name] = client.analyze(images, prompt, system)
        return results
