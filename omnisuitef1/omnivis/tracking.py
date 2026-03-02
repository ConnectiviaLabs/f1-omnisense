"""Multi-object tracking via YOLO model.track() — ByteTrack / BoT-SORT.

Usage:
    from omnisee.tracking import track, track_video

    dets = track(frame, tracker="bytetrack")      # single frame, persistent IDs
    all_dets = track_video(frames, tracker="botsort")  # batch with maintained IDs
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from omnivis._types import Detection
from omnivis.detectors import _yolo_to_detections
from omnivis.model_manager import get_model_manager

logger = logging.getLogger(__name__)

TRACKERS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
    "botsort_reid": "botsort.yaml",  # ultralytics handles reid config
}


def track(
    frame: np.ndarray,
    *,
    tracker: str = "bytetrack",
    confidence: float = 0.25,
    imgsz: int = 768,
    persist: bool = True,
    model_type: str = "Track",
) -> List[Detection]:
    """Run detection + tracking on a single frame.

    Call repeatedly with persist=True to maintain track IDs across frames.

    Args:
        frame: BGR numpy array.
        tracker: "bytetrack" or "botsort".
        confidence: Minimum confidence threshold.
        imgsz: YOLO input resolution.
        persist: Maintain track state across calls.
        model_type: YOLO model to use (default "Track" = yolo26l).

    Returns:
        List of Detection objects with track_id populated.
    """
    manager = get_model_manager()
    model = manager.load_model(model_type)
    if model is None:
        return []

    tracker_yaml = TRACKERS.get(tracker.lower(), "bytetrack.yaml")

    results = model.track(
        frame, verbose=False, conf=confidence, imgsz=imgsz,
        persist=persist, tracker=tracker_yaml,
    )
    return _yolo_to_detections(results)


def track_video(
    frames: List[np.ndarray],
    *,
    tracker: str = "bytetrack",
    confidence: float = 0.25,
    imgsz: int = 768,
    model_type: str = "Track",
) -> List[List[Detection]]:
    """Run tracking across a sequence of frames.

    Maintains track state through the entire sequence.

    Returns:
        List of detection lists (one per frame).
    """
    manager = get_model_manager()
    model = manager.load_model(model_type)
    if model is None:
        return [[] for _ in frames]

    tracker_yaml = TRACKERS.get(tracker.lower(), "bytetrack.yaml")
    all_dets: List[List[Detection]] = []

    for i, frame in enumerate(frames):
        results = model.track(
            frame, verbose=False, conf=confidence, imgsz=imgsz,
            persist=True, tracker=tracker_yaml,
        )
        all_dets.append(_yolo_to_detections(results))

    return all_dets


def reset_tracker(model_type: str = "Track"):
    """Reset tracker state (call when switching to a new video/stream)."""
    manager = get_model_manager()
    model = manager.load_model(model_type)
    if model is not None and hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "trackers"):
            model.predictor.trackers = []
            logger.info("Tracker state reset for %s", model_type)
