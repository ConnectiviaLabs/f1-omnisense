"""Instance segmentation — SAM2 (high quality) + YOLO-seg (fast).

Usage:
    from omnisee.segmentation import segment, segment_with_boxes, overlay_masks

    result = segment(frame, model="sam2")
    result = segment_with_boxes(frame, boxes=[[100, 100, 200, 200]])
    annotated = overlay_masks(frame, result)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from omnivis._types import Detection, SegmentationResult
from omnivis.model_manager import get_model_manager

logger = logging.getLogger(__name__)

MASK_ALPHA = 0.35
MASK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]


def segment(
    frame: np.ndarray,
    *,
    model: str = "yolo",
    confidence: float = 0.25,
) -> SegmentationResult:
    """Run instance segmentation on a frame.

    Args:
        frame: BGR numpy array.
        model: "yolo" for fast YOLO26-seg, "sam2" for high-quality SAM2.
        confidence: Minimum confidence threshold.
    """
    if model == "yolo":
        return _segment_yolo(frame, confidence)
    elif model == "sam2":
        # Detect first, then segment with SAM2
        from omnivis.detectors import detect
        dets = detect(frame, confidence=confidence)
        if not dets:
            return SegmentationResult()
        boxes = [d.bbox for d in dets]
        result = segment_with_boxes(frame, boxes)
        result.detections = dets
        return result
    else:
        raise ValueError(f"Unknown segmentation model: {model}. Use 'yolo' or 'sam2'.")


def segment_with_boxes(
    frame: np.ndarray,
    boxes: List[List[float]],
) -> SegmentationResult:
    """Run SAM2 segmentation from provided bounding boxes.

    Args:
        frame: BGR numpy array.
        boxes: List of [x1, y1, x2, y2] bounding boxes.
    """
    manager = get_model_manager()

    try:
        predictor = manager.load_hf_model("sam2")
    except Exception as e:
        logger.error("SAM2 not available: %s", e)
        return SegmentationResult()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    masks_list = []
    scores_list = []

    try:
        # sam2 SAM2ImagePredictor API
        if hasattr(predictor, "set_image"):
            predictor.set_image(rgb)
            for box in boxes:
                input_box = np.array(box)
                masks, scores, _ = predictor.predict(box=input_box, multimask_output=False)
                masks_list.append(masks[0])
                scores_list.append(float(scores[0]))
        else:
            # transformers SamModel fallback
            model, processor = predictor
            for box in boxes:
                inputs = processor(rgb, input_boxes=[[[int(c) for c in box]]], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                import torch
                with torch.no_grad():
                    outputs = model(**inputs)
                mask = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )[0][0][0].numpy()
                masks_list.append(mask)
                scores_list.append(1.0)

    except Exception as e:
        logger.error("SAM2 prediction failed: %s", e)

    return SegmentationResult(masks=masks_list, scores=scores_list)


def overlay_masks(
    frame: np.ndarray,
    result: SegmentationResult,
    alpha: float = MASK_ALPHA,
) -> np.ndarray:
    """Draw semi-transparent mask overlays on a frame."""
    output = frame.copy()

    for i, mask in enumerate(result.masks):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        if mask.dtype == bool:
            mask_uint8 = mask.astype(np.uint8)
        else:
            mask_uint8 = (mask > 0.5).astype(np.uint8)

        # Resize mask to frame if needed
        if mask_uint8.shape[:2] != frame.shape[:2]:
            mask_uint8 = cv2.resize(mask_uint8, (frame.shape[1], frame.shape[0]))

        colored = np.zeros_like(frame)
        colored[:] = color
        mask_3ch = np.stack([mask_uint8] * 3, axis=-1)
        output = np.where(mask_3ch, cv2.addWeighted(output, 1 - alpha, colored, alpha, 0), output)

    return output


def _segment_yolo(frame: np.ndarray, confidence: float) -> SegmentationResult:
    """Fast segmentation via YOLO26-seg."""
    manager = get_model_manager()
    model = manager.load_model("Segmentation")
    if model is None:
        return SegmentationResult()

    results = model(frame, verbose=False, conf=confidence)
    from omnivis.detectors import _yolo_to_detections
    dets = _yolo_to_detections(results)

    masks = [d.mask for d in dets if d.mask is not None]
    scores = [d.confidence for d in dets]

    return SegmentationResult(detections=dets, masks=masks, scores=scores)
