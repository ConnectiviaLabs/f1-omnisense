"""OmniSee — unified computer vision service.

Combines real-time YOLO detection/tracking (MediaSense) with zero-shot
grounding, segmentation, video classification, and scene narration (F1).

Quick start:
    from omnisee import detect, segment, track, zero_shot_detect

    detections = detect(frame)
    result = segment(frame, model="sam2")
    tracked = track(frame, tracker="bytetrack")
    grounded = zero_shot_detect(frame, "person . car . dog")
"""

# Core detection
from omnivis.detectors import detect, detect_batch, detect_threat

# Segmentation
from omnivis.segmentation import segment, segment_with_boxes, overlay_masks

# Zero-shot grounding
from omnivis.grounding import zero_shot_detect

# Multi-object tracking
from omnivis.tracking import track, track_video, reset_tracker

# Action classification
from omnivis.action_classifier import classify_actions

# Scene narration
from omnivis.narration import narrate, narrate_with_detections

# CLIP embeddings
from omnivis.clip_embeddings import embed_clip, embed_text, auto_tag, search_by_text

# Drone pipeline
from omnivis.drone_pipeline import detect_drone

# Streaming
from omnivis.streaming import generate_detection_stream, get_stream_registry

# Preprocessing
from omnivis.preprocessing import preprocess_frame, upscale_frame, slice_frame

# Model management
from omnivis.model_manager import get_model_manager

# Types
from omnivis._types import (
    Detection,
    SegmentationResult,
    ActionClassification,
    NarrationResult,
    CLIPEmbedding,
    DroneDetection,
    StreamState,
    ModelType,
)

__all__ = [
    # Detection
    "detect",
    "detect_batch",
    "detect_threat",
    # Segmentation
    "segment",
    "segment_with_boxes",
    "overlay_masks",
    # Grounding
    "zero_shot_detect",
    # Tracking
    "track",
    "track_video",
    "reset_tracker",
    # Action classification
    "classify_actions",
    # Narration
    "narrate",
    "narrate_with_detections",
    # CLIP
    "embed_clip",
    "embed_text",
    "auto_tag",
    "search_by_text",
    # Drone
    "detect_drone",
    # Streaming
    "generate_detection_stream",
    "get_stream_registry",
    # Preprocessing
    "preprocess_frame",
    "upscale_frame",
    "slice_frame",
    # Model management
    "get_model_manager",
    # Types
    "Detection",
    "SegmentationResult",
    "ActionClassification",
    "NarrationResult",
    "CLIPEmbedding",
    "DroneDetection",
    "StreamState",
    "ModelType",
]
