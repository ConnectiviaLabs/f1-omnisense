"""Shared type definitions for the omnidoc vision service."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class ModelType(str, Enum):
    DETECTION = "Detection"
    SEGMENTATION = "Segmentation"
    POSE = "Pose"
    CLASSIFICATION = "Classification"
    TRACK = "Track"
    OBB = "OBB"
    THREAT = "Threat"
    OPEN_VOCAB = "OpenVocab"
    DRONE = "Drone"
    DRONE_VISION = "DroneVision"
    FISH = "Fish"
    MARITIME = "Maritime"
    ENVIRONMENT = "Environment"


@dataclass
class Detection:
    """A single object detection result."""

    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    class_id: int = -1
    track_id: Optional[int] = None
    mask: Optional[Any] = None  # np.ndarray when present
    keypoints: Optional[Any] = None  # np.ndarray when present (pose)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "class_id": self.class_id,
        }
        if self.track_id is not None:
            d["track_id"] = self.track_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class SegmentationResult:
    """Instance segmentation output."""

    detections: List[Detection] = field(default_factory=list)
    masks: List[Any] = field(default_factory=list)  # list[np.ndarray]
    scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "mask_count": len(self.masks),
            "scores": self.scores,
        }


@dataclass
class ActionClassification:
    """A single action classification prediction."""

    label: str
    confidence: float
    model: str  # "timesformer" or "videomae"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NarrationResult:
    """LLM scene narration output."""

    text: str
    model: str
    latency_s: float = 0.0
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CLIPEmbedding:
    """A single CLIP embedding vector."""

    vector: List[float]  # 512-dim
    source: str  # "image" or "text"

    def to_dict(self) -> Dict[str, Any]:
        return {"vector": self.vector, "source": self.source, "dim": len(self.vector)}


@dataclass
class DroneDetection:
    """Unified drone detection from multi-modal fusion."""

    source: str  # "video", "audio", "fused"
    confidence: float
    priority: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    bbox: Optional[List[float]] = None
    audio_class: Optional[str] = None
    detected_class: Optional[str] = None
    is_drone_model: bool = True
    timestamp: float = field(default_factory=time.time)
    frame_number: Optional[int] = None
    heuristic_signatures: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StreamState:
    """Per-stream state for real-time processing."""

    stream_id: str
    frame_count: int = 0
    detection_history: deque = field(default_factory=lambda: deque(maxlen=3000))
    stop_requested: bool = False
    last_access: float = field(default_factory=time.time)


# ── Priority helpers ──────────────────────────────────────────────────────

CRITICAL_THRESHOLD = 0.80
HIGH_THRESHOLD = 0.60
MEDIUM_THRESHOLD = 0.40


def get_priority(confidence: float, *, require_fused_for_critical: bool = False, is_fused: bool = False) -> str:
    if confidence >= CRITICAL_THRESHOLD:
        if require_fused_for_critical and not is_fused:
            return "HIGH"
        return "CRITICAL"
    elif confidence >= HIGH_THRESHOLD:
        return "HIGH"
    elif confidence >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"
