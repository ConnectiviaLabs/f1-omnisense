"""Object detection via YOLO26 + RF-DETR threat detection.

Usage:
    from omnisee.detectors import detect, detect_threat

    dets = detect(frame)
    dets = detect(frame, model_type="Drone", sahi=True)
    threats = detect_threat(frame)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from omnivis._types import Detection, ModelType
from omnivis.model_manager import get_model_manager, THREAT_CONFIDENCE

logger = logging.getLogger(__name__)


def detect(
    frame: np.ndarray,
    *,
    model_type: str = "Detection",
    confidence: float = 0.25,
    classes: Optional[List[int]] = None,
    imgsz: int = 768,
    sahi: bool = False,
    sahi_tile_size: int = 512,
    sahi_overlap: float = 0.2,
) -> List[Detection]:
    """Run YOLO detection on a single frame.

    Args:
        frame: BGR numpy array.
        model_type: One of ModelType values (Detection, Drone, Maritime, etc.).
        confidence: Minimum confidence threshold.
        classes: Filter to specific class IDs.
        imgsz: Input resolution for YOLO.
        sahi: Use SAHI sliced inference for small objects.
        sahi_tile_size: Tile size in pixels for SAHI.
        sahi_overlap: Overlap ratio between tiles.

    Returns:
        List of Detection objects.
    """
    if sahi:
        return _detect_sahi(frame, model_type=model_type, confidence=confidence,
                            tile_size=sahi_tile_size, overlap=sahi_overlap)

    manager = get_model_manager()
    model = manager.load_model(model_type)
    if model is None:
        logger.error("Model %s not available", model_type)
        return []

    kwargs: Dict[str, Any] = {"verbose": False, "conf": confidence, "imgsz": imgsz}
    if classes:
        kwargs["classes"] = classes

    results = model(frame, **kwargs)
    return _yolo_to_detections(results)


def detect_batch(
    frames: List[np.ndarray],
    *,
    model_type: str = "Detection",
    confidence: float = 0.25,
    imgsz: int = 768,
) -> List[List[Detection]]:
    """Run detection on multiple frames."""
    manager = get_model_manager()
    model = manager.load_model(model_type)
    if model is None:
        return [[] for _ in frames]

    all_results = model(frames, verbose=False, conf=confidence, imgsz=imgsz)
    return [_yolo_to_detections([r]) for r in all_results]


def detect_threat(
    frame: np.ndarray,
    *,
    confidence: float = THREAT_CONFIDENCE,
    max_area_ratio: float = 0.25,
) -> List[Detection]:
    """Run RF-DETR threat detection (weapons, explosives).

    Args:
        frame: BGR numpy array.
        confidence: Minimum confidence.
        max_area_ratio: Max detection area as fraction of frame (filter oversized FPs).
    """
    manager = get_model_manager()
    model = manager.load_model("Threat")
    if model is None:
        return []

    try:
        results = model.predict(frame, threshold=confidence)
        h, w = frame.shape[:2]
        frame_area = h * w
        detections: List[Detection] = []

        for det in results:
            bbox = det["bbox"]  # xyxy
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area / frame_area > max_area_ratio:
                continue

            detections.append(Detection(
                bbox=[float(c) for c in bbox],
                confidence=float(det["confidence"]),
                class_name=det.get("class", "threat"),
                class_id=det.get("class_id", -1),
            ))
        return detections
    except Exception as e:
        logger.error("Threat detection failed: %s", e)
        return []


# ── SAHI sliced inference ─────────────────────────────────────────────────

def _detect_sahi(
    frame: np.ndarray,
    *,
    model_type: str,
    confidence: float,
    tile_size: int,
    overlap: float,
) -> List[Detection]:
    """Run detection using SAHI tiling for small objects."""
    from omnivis.preprocessing import slice_frame, merge_tile_detections

    manager = get_model_manager()
    model = manager.load_model(model_type)
    if model is None:
        return []

    tiles = slice_frame(frame, tile_size=tile_size, overlap=overlap)
    tile_results = []

    for t in tiles:
        results = model(t["tile"], verbose=False, conf=confidence)
        dets = _yolo_to_detections(results)
        tile_results.append((dets, t["offset"]))

    return merge_tile_detections(tile_results, iou_threshold=0.5)


# ── YOLO result conversion ───────────────────────────────────────────────

def _yolo_to_detections(results) -> List[Detection]:
    """Convert ultralytics Results to Detection dataclass list."""
    detections: List[Detection] = []

    for r in results:
        if r.boxes is None:
            continue

        names = r.names or {}

        for i, box in enumerate(r.boxes):
            xyxy = box.xyxy[0].cpu().tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))

            d = Detection(
                bbox=xyxy,
                confidence=conf,
                class_name=cls_name,
                class_id=cls_id,
            )

            # Track ID if available
            if box.id is not None:
                d.track_id = int(box.id[0])

            # Mask if available (segmentation)
            if r.masks is not None and i < len(r.masks):
                d.mask = r.masks[i].data.cpu().numpy()

            # Keypoints if available (pose)
            if r.keypoints is not None and i < len(r.keypoints):
                d.keypoints = r.keypoints[i].data.cpu().numpy()

            detections.append(d)

    return detections
