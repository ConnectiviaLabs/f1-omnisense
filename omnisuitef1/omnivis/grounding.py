"""GroundingDINO zero-shot object detection — detect anything by text prompt.

Usage:
    from omnisee.grounding import zero_shot_detect

    dets = zero_shot_detect(frame, "person . car . dog")
    dets = zero_shot_detect(frame, "formula one car . tire . pit crew")
"""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from omnivis._types import Detection
from omnivis.model_manager import get_model_manager

logger = logging.getLogger(__name__)

MODEL_ID = "IDEA-Research/grounding-dino-base"


def zero_shot_detect(
    frame: np.ndarray,
    text_prompt: str,
    *,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    max_area_ratio: float = 0.45,
) -> List[Detection]:
    """Detect objects in a frame using text-prompted zero-shot detection.

    Args:
        frame: BGR numpy array.
        text_prompt: Dot-separated categories, e.g. "person . car . dog".
        box_threshold: Minimum box confidence.
        text_threshold: Minimum text-grounding confidence.
        max_area_ratio: Filter detections covering more than this fraction of frame.

    Returns:
        List of Detection objects.
    """
    manager = get_model_manager()

    try:
        model, processor = manager.load_hf_model(MODEL_ID)
    except Exception as e:
        logger.error("GroundingDINO not available: %s", e)
        return []

    from PIL import Image
    import torch

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(frame.shape[0], frame.shape[1])],
    )[0]

    h, w = frame.shape[:2]
    frame_area = h * w
    detections: List[Detection] = []

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results.get("labels", results.get("text", []))
    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    for bbox, score, label in zip(boxes, scores, labels):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area / frame_area > max_area_ratio:
            continue

        detections.append(Detection(
            bbox=[float(c) for c in bbox],
            confidence=float(score),
            class_name=str(label).strip(),
        ))

    return detections
