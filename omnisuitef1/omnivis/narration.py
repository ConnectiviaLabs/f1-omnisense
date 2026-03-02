"""LLM scene narration via Ollama (MiniCPM-V / Gemma).

Usage:
    from omnisee.narration import narrate, narrate_with_detections

    result = narrate(frame)
    result = narrate_with_detections(frame, detections)
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from omnivis._types import Detection, NarrationResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "minicpm-v:4.5"
FALLBACK_MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434"


def narrate(
    frame: np.ndarray,
    *,
    context: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 256,
    ollama_url: str = OLLAMA_URL,
) -> NarrationResult:
    """Generate a text description of a video frame using a vision LLM.

    Args:
        frame: BGR numpy array.
        context: Additional context to include in the prompt.
        model: Ollama model name (default: minicpm-v:4.5).
        temperature: LLM temperature (lower = more deterministic).
        max_tokens: Maximum tokens to generate.
        ollama_url: Ollama server URL.

    Returns:
        NarrationResult with the generated text.
    """
    b64 = _frame_to_b64(frame)
    prompt = "Describe what you see in this image in 2-3 sentences. Focus on the key objects, actions, and scene context."
    if context:
        prompt = f"{context}\n\n{prompt}"

    t0 = time.time()
    text = _call_ollama(model, prompt, b64, temperature, max_tokens, ollama_url)

    if text is None:
        # Try fallback model
        logger.warning("Primary model %s failed, trying %s", model, FALLBACK_MODEL)
        text = _call_ollama(FALLBACK_MODEL, prompt, b64, temperature, max_tokens, ollama_url)
        if text is not None:
            model = FALLBACK_MODEL

    latency = time.time() - t0

    if text is None:
        return NarrationResult(text="", model=model, latency_s=latency, success=False)

    return NarrationResult(text=text, model=model, latency_s=round(latency, 2), success=True)


def narrate_with_detections(
    frame: np.ndarray,
    detections: List[Detection],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 256,
    ollama_url: str = OLLAMA_URL,
) -> NarrationResult:
    """Narrate a frame with YOLO detection context for richer descriptions."""
    if not detections:
        return narrate(frame, model=model, temperature=temperature,
                       max_tokens=max_tokens, ollama_url=ollama_url)

    # Format detection context
    counts: Dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
    summary = ", ".join(f"{count}x {name}" for name, count in sorted(counts.items(), key=lambda x: -x[1]))
    context = f"Objects detected: {summary}. Describe the scene in detail."

    return narrate(frame, context=context, model=model, temperature=temperature,
                   max_tokens=max_tokens, ollama_url=ollama_url)


# ── Internal ──────────────────────────────────────────────────────────────

def _frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _call_ollama(
    model: str, prompt: str, image_b64: str,
    temperature: float, max_tokens: int, ollama_url: str,
) -> Optional[str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    try:
        resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.ConnectionError:
        logger.error("Cannot connect to Ollama at %s", ollama_url)
        return None
    except Exception as e:
        logger.error("Ollama request failed (%s): %s", model, e)
        return None
