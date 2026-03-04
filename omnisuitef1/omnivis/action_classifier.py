"""Video action classification — TimeSformer + VideoMAE (Kinetics-400).

Usage:
    from omnisee.action_classifier import classify_actions

    actions = classify_actions(frames, model="timesformer", top_k=5)
    actions = classify_actions(frames, model="videomae", top_k=10)
"""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from omnivis._types import ActionClassification
from omnivis.model_manager import get_model_manager

logger = logging.getLogger(__name__)

TIMESFORMER_ID = "facebook/timesformer-base-finetuned-k400"
VIDEOMAE_ID = "MCG-NJU/videomae-base-finetuned-kinetics"

# Required frame counts
TIMESFORMER_FRAMES = 8
VIDEOMAE_FRAMES = 16


def classify_actions(
    frames: List[np.ndarray],
    *,
    model: str = "timesformer",
    top_k: int = 5,
) -> List[ActionClassification]:
    """Classify human actions in a video clip.

    Args:
        frames: List of BGR numpy arrays (a video clip).
        model: "timesformer" (8 frames) or "videomae" (16 frames).
        top_k: Number of top predictions to return.

    Returns:
        List of ActionClassification sorted by confidence descending.
    """
    if model == "timesformer":
        return _classify_timesformer(frames, top_k)
    elif model == "videomae":
        return _classify_videomae(frames, top_k)
    else:
        raise ValueError(f"Unknown model: {model}. Use 'timesformer' or 'videomae'.")


def _classify_timesformer(frames: List[np.ndarray], top_k: int) -> List[ActionClassification]:
    import torch

    manager = get_model_manager()
    model, processor = manager.load_hf_model(TIMESFORMER_ID)

    sampled = _sample_frames(frames, TIMESFORMER_FRAMES)
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled]

    inputs = processor(rgb_frames, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    top_indices = probs.topk(top_k).indices.cpu().tolist()
    results = []
    for idx in top_indices:
        results.append(ActionClassification(
            label=model.config.id2label[idx],
            confidence=round(float(probs[idx]), 4),
            model="timesformer",
        ))
    return results


def _classify_videomae(frames: List[np.ndarray], top_k: int) -> List[ActionClassification]:
    import torch

    manager = get_model_manager()
    model, processor = manager.load_hf_model(VIDEOMAE_ID)

    sampled = _sample_frames(frames, VIDEOMAE_FRAMES)
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled]

    inputs = processor(rgb_frames, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    top_indices = probs.topk(top_k).indices.cpu().tolist()
    results = []
    for idx in top_indices:
        results.append(ActionClassification(
            label=model.config.id2label[idx],
            confidence=round(float(probs[idx]), 4),
            model="videomae",
        ))
    return results


def _sample_frames(frames: List[np.ndarray], n_target: int) -> List[np.ndarray]:
    """Uniformly sample n_target frames from a longer clip."""
    if len(frames) <= n_target:
        # Pad by repeating last frame
        while len(frames) < n_target:
            frames = frames + [frames[-1]]
        return frames[:n_target]

    indices = np.linspace(0, len(frames) - 1, n_target, dtype=int)
    return [frames[i] for i in indices]
