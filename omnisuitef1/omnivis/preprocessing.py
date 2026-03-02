"""Frame preprocessing — CLAHE, bilateral filter, upscaling, SAHI tiling, bbox utils.

Adapted from MediaSense frame_preprocessor.py.

Usage:
    from omnisee.preprocessing import preprocess_frame, upscale_frame, slice_frame

    enhanced = preprocess_frame(frame)
    upscaled, scale = upscale_frame(frame, scale_factor=2.0)
    tiles = slice_frame(frame, tile_size=512, overlap=0.2)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

from omnivis._types import Detection

logger = logging.getLogger(__name__)

INTERPOLATION_METHODS = {
    "LANCZOS": cv2.INTER_LANCZOS4,
    "CUBIC": cv2.INTER_CUBIC,
    "LINEAR": cv2.INTER_LINEAR,
    "NEAREST": cv2.INTER_NEAREST,
}


def preprocess_frame(
    frame: np.ndarray,
    *,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    bilateral_d: int = 9,
    bilateral_sigma: float = 75.0,
) -> np.ndarray:
    """Apply CLAHE on LAB L-channel + bilateral filter.

    Improves detection of small dark objects against bright backgrounds.
    """
    try:
        denoised = cv2.bilateralFilter(frame, bilateral_d, bilateral_sigma, bilateral_sigma)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        l_enhanced = clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception as e:
        logger.warning("Preprocessing failed, using original: %s", e)
        return frame


def upscale_frame(
    frame: np.ndarray,
    scale_factor: float = 2.0,
    method: str = "LANCZOS",
) -> Tuple[np.ndarray, float]:
    """Upscale a frame for better small-object detection.

    Returns (upscaled_frame, scale_factor_used).
    """
    if scale_factor <= 1.0:
        return frame, 1.0
    h, w = frame.shape[:2]
    interp = INTERPOLATION_METHODS.get(method.upper(), cv2.INTER_LANCZOS4)
    upscaled = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)), interpolation=interp)
    return upscaled, scale_factor


def scale_bbox(bbox: List[float], scale_factor: float) -> List[float]:
    """Scale bbox coordinates back to original frame size."""
    if scale_factor <= 1.0:
        return bbox
    return [c / scale_factor for c in bbox]


def scale_bboxes(bboxes: List[List[float]], scale_factor: float) -> List[List[float]]:
    if scale_factor <= 1.0:
        return bboxes
    return [scale_bbox(b, scale_factor) for b in bboxes]


def validate_bbox(bbox: List[float], width: int, height: int) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height


def clamp_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    if not bbox or len(bbox) != 4:
        return bbox
    x1, y1, x2, y2 = bbox
    return [max(0, min(x1, width)), max(0, min(y1, height)),
            max(0, min(x2, width)), max(0, min(y2, height))]


def bbox_area(bbox: List[float]) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_iou(a: List[float], b: List[float]) -> float:
    """Intersection over Union for two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── SAHI-style tiling ─────────────────────────────────────────────────────

def slice_frame(
    frame: np.ndarray,
    tile_size: int = 512,
    overlap: float = 0.2,
) -> List[Dict]:
    """Split frame into overlapping tiles for sliced inference.

    Returns list of {"tile": np.ndarray, "offset": (x, y), "size": (w, h)}.
    """
    h, w = frame.shape[:2]
    step = int(tile_size * (1 - overlap))
    tiles = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            tile = frame[y:y2, x:x2]
            tiles.append({
                "tile": tile,
                "offset": (x, y),
                "size": (x2 - x, y2 - y),
            })
    return tiles


def merge_tile_detections(
    tile_results: List[Tuple[List[Detection], Tuple[int, int]]],
    iou_threshold: float = 0.5,
) -> List[Detection]:
    """Merge detections from multiple tiles, applying NMS.

    Args:
        tile_results: List of (detections, (offset_x, offset_y)) pairs.
        iou_threshold: IoU threshold for NMS deduplication.
    """
    all_dets: List[Detection] = []

    for dets, (ox, oy) in tile_results:
        for d in dets:
            shifted = Detection(
                bbox=[d.bbox[0] + ox, d.bbox[1] + oy, d.bbox[2] + ox, d.bbox[3] + oy],
                confidence=d.confidence,
                class_name=d.class_name,
                class_id=d.class_id,
                metadata=d.metadata,
            )
            all_dets.append(shifted)

    if not all_dets:
        return []

    # NMS per class
    by_class: Dict[str, List[Detection]] = {}
    for d in all_dets:
        by_class.setdefault(d.class_name, []).append(d)

    result: List[Detection] = []
    for cls_dets in by_class.values():
        cls_dets.sort(key=lambda d: d.confidence, reverse=True)
        keep: List[Detection] = []
        for d in cls_dets:
            if all(bbox_iou(d.bbox, k.bbox) < iou_threshold for k in keep):
                keep.append(d)
        result.extend(keep)

    return result
