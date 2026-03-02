"""OmniVis APIRouter — computer vision: detection, segmentation, tracking, CLIP, narration.

Wraps the omnivis (OmniSee) module as a thin gateway router.
Heavy CV endpoints are proxied rather than re-implemented.

Endpoints:
    POST /api/omni/vis/detect                — object detection on uploaded image
    POST /api/omni/vis/detect/threat          — threat detection
    POST /api/omni/vis/segment                — instance segmentation
    POST /api/omni/vis/zero-shot              — zero-shot text-prompted detection
    POST /api/omni/vis/embed/image            — CLIP image embedding (512-dim)
    POST /api/omni/vis/embed/text             — CLIP text embedding (512-dim)
    POST /api/omni/vis/auto-tag               — auto-tag image against categories
    POST /api/omni/vis/narrate                — scene narration via vision LLM
    GET  /api/omni/vis/models                 — list loaded models
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/omni/vis", tags=["OmniVis"])


def _read_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to BGR numpy array."""
    import cv2
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")
    return frame


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/detect")
async def detect(
    image: UploadFile = File(...),
    model_type: str = Form("Detection"),
    confidence: float = Form(0.25),
    imgsz: int = Form(768),
    sahi: bool = Form(False),
):
    """Run object detection on an uploaded image."""
    from omnivis import detect as _detect

    frame = _read_image(await image.read())
    detections = _detect(frame, model_type=model_type, confidence=confidence, imgsz=imgsz, sahi=sahi)
    return {
        "detections": [d.to_dict() for d in detections],
        "count": len(detections),
    }


@router.post("/detect/threat")
async def detect_threat(
    image: UploadFile = File(...),
    confidence: float = Form(0.25),
):
    """Run threat detection (weapons, explosives) on an uploaded image."""
    from omnivis import detect_threat as _detect_threat

    frame = _read_image(await image.read())
    detections = _detect_threat(frame, confidence=confidence)
    return {
        "detections": [d.to_dict() for d in detections],
        "count": len(detections),
    }


@router.post("/segment")
async def segment(
    image: UploadFile = File(...),
    model: str = Form("yolo"),
    confidence: float = Form(0.25),
):
    """Run instance segmentation."""
    from omnivis import segment as _segment

    frame = _read_image(await image.read())
    result = _segment(frame, model=model, confidence=confidence)
    return {
        "detections": [d.to_dict() for d in result.detections],
        "scores": result.scores,
        "count": len(result.detections),
    }


@router.post("/zero-shot")
async def zero_shot_detect(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: float = Form(0.25),
    text_threshold: float = Form(0.20),
):
    """Zero-shot text-prompted detection via GroundingDINO."""
    from omnivis import zero_shot_detect as _zsd

    frame = _read_image(await image.read())
    detections = _zsd(frame, text_prompt=text_prompt, box_threshold=box_threshold, text_threshold=text_threshold)
    return {
        "detections": [d.to_dict() for d in detections],
        "count": len(detections),
        "prompt": text_prompt,
    }


@router.post("/embed/image")
async def embed_image(image: UploadFile = File(...)):
    """Generate 512-dim CLIP embedding for an image."""
    from omnivis import embed_clip

    frame = _read_image(await image.read())
    embedding = embed_clip(frame)
    return {
        "vector": embedding,
        "dimensions": len(embedding),
        "source": "image",
    }


class TextEmbedRequest(BaseModel):
    text: str


@router.post("/embed/text")
def embed_text(req: TextEmbedRequest):
    """Generate 512-dim CLIP embedding for text."""
    from omnivis import embed_text as _embed_text

    embedding = _embed_text(req.text)
    return {
        "vector": embedding,
        "dimensions": len(embedding),
        "source": "text",
        "text": req.text,
    }


@router.post("/auto-tag")
async def auto_tag(
    image: UploadFile = File(...),
    categories: str = Form("car,person,tire,helmet,wing,engine,pit stop"),
    top_k: int = Form(5),
):
    """Auto-tag image by matching against category list via CLIP."""
    from omnivis import auto_tag as _auto_tag

    frame = _read_image(await image.read())
    cat_list = [c.strip() for c in categories.split(",") if c.strip()]
    tags = _auto_tag(frame, cat_list, top_k=top_k)
    return {"tags": tags}


@router.post("/narrate")
async def narrate(
    image: UploadFile = File(...),
    context: str = Form(""),
    model: str = Form("minicpm-v:4.5"),
):
    """Narrate a scene using a vision LLM via Ollama."""
    from omnivis import narrate as _narrate

    frame = _read_image(await image.read())
    result = _narrate(frame, context=context, model=model)
    return {
        "text": result.text,
        "model": result.model,
        "latency_s": result.latency_s,
        "success": result.success,
    }


@router.post("/analyze-video")
async def analyze_video(
    video: UploadFile = File(...),
    tasks: str = Form("detect,classify"),
    confidence: float = Form(0.25),
    max_frames: int = Form(30),
):
    """Upload a video for CV analysis. Runs detection and/or classification."""
    import cv2
    import os
    import tempfile
    from datetime import datetime, timezone

    ext = os.path.splitext(video.filename or "video.mp4")[1].lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(400, f"Unsupported video format: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        # Sample evenly spaced frames
        step = max(1, total_frames // max_frames)
        idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()

        if not frames:
            raise HTTPException(400, "No frames extracted from video")

        task_list = [t.strip() for t in tasks.split(",")]
        results: dict = {
            "filename": video.filename,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "sampled_frames": len(frames),
        }

        # Object detection on sampled frames
        if "detect" in task_list:
            from omnivis import detect as _detect
            detections = []
            for i, frame in enumerate(frames):
                dets = _detect(frame, confidence=confidence)
                detections.append({
                    "frame_index": i * step,
                    "detections": [d.to_dict() for d in dets],
                })
            results["gdino"] = detections

        # Video classification
        if "classify" in task_list:
            from omnivis import classify_actions
            actions = classify_actions(frames)
            results["classification"] = [a.__dict__ if hasattr(a, "__dict__") else a for a in actions]

        # Store results in MongoDB for pipeline display
        try:
            from pymongo import MongoClient
            uri = os.environ.get("MONGODB_URI", "")
            db_name = os.environ.get("MONGODB_DB", "marip_f1")
            if uri:
                client = MongoClient(uri)
                db = client[db_name]
                ts = datetime.now(timezone.utc)
                if "gdino" in results:
                    db["pipeline_gdino_results"].insert_one({
                        "filename": video.filename,
                        "timestamp": ts,
                        "frames": results["gdino"],
                    })
                if "classification" in results:
                    db["pipeline_videomae_results"].insert_one({
                        "filename": video.filename,
                        "timestamp": ts,
                        "total_frames": total_frames,
                        "fps": round(fps, 2),
                        "top_predictions": results["classification"],
                    })
        except Exception as e:
            logger.warning("Failed to store results in MongoDB: %s", e)

        return results

    finally:
        os.unlink(tmp_path)


@router.get("/models")
def list_models():
    """List currently loaded models and their status."""
    try:
        from omnivis import get_model_manager
        manager = get_model_manager()
        return {"models": manager.list_loaded()}
    except Exception:
        return {"models": [], "note": "Model manager not initialized"}
