"""FastAPI server exposing all OmniSee vision capabilities.

Usage:
    uvicorn omnisee.server:app --host 0.0.0.0 --port 8000

Environment variables:
    OMNIDOC_MODELS_DIR   — Model cache directory (default ~/.cache/omnidoc/models/)
    OMNIDOC_OLLAMA_URL   — Ollama base URL (default http://localhost:11434)
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OmniSee",
    description="Unified computer vision service — detection, segmentation, tracking, narration, and more.",
    version="0.1.0",
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _read_image(file: UploadFile) -> np.ndarray:
    """Read an uploaded image into a BGR numpy array."""
    data = file.file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _read_frames(files: List[UploadFile]) -> List[np.ndarray]:
    """Read multiple uploaded images as video frames."""
    return [_read_image(f) for f in files]


# ── Health & Info ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/models")
async def list_models():
    from omnivis.model_manager import get_model_manager
    manager = get_model_manager()
    return {
        "loaded": list(manager._models.keys()),
        "models_dir": str(manager.models_dir),
    }


# ── Detection ────────────────────────────────────────────────────────────

@app.post("/detect")
async def detect_endpoint(
    image: UploadFile = File(...),
    model_type: str = Form("Detection"),
    confidence: float = Form(0.25),
    imgsz: int = Form(768),
    sahi: bool = Form(False),
):
    from omnivis.detectors import detect

    frame = _read_image(image)
    dets = detect(frame, model_type=model_type, confidence=confidence, imgsz=imgsz, sahi=sahi)
    return {"detections": [d.to_dict() for d in dets], "count": len(dets)}


@app.post("/detect/batch")
async def detect_batch_endpoint(
    images: List[UploadFile] = File(...),
    model_type: str = Form("Detection"),
    confidence: float = Form(0.25),
    imgsz: int = Form(768),
):
    from omnivis.detectors import detect_batch

    frames = _read_frames(images)
    results = detect_batch(frames, model_type=model_type, confidence=confidence, imgsz=imgsz)
    return {
        "results": [
            {"detections": [d.to_dict() for d in dets], "count": len(dets)}
            for dets in results
        ]
    }


@app.post("/detect/threat")
async def detect_threat_endpoint(
    image: UploadFile = File(...),
    confidence: float = Form(0.25),
):
    from omnivis.detectors import detect_threat

    frame = _read_image(image)
    dets = detect_threat(frame, confidence=confidence)
    return {"detections": [d.to_dict() for d in dets], "count": len(dets)}


# ── Segmentation ─────────────────────────────────────────────────────────

@app.post("/segment")
async def segment_endpoint(
    image: UploadFile = File(...),
    model: str = Form("yolo"),
    confidence: float = Form(0.25),
    imgsz: int = Form(768),
):
    from omnivis.segmentation import segment

    frame = _read_image(image)
    result = segment(frame, model=model, confidence=confidence, imgsz=imgsz)
    return result.to_dict()


# ── Zero-shot Detection ─────────────────────────────────────────────────

@app.post("/zero-shot")
async def zero_shot_endpoint(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: float = Form(0.25),
    text_threshold: float = Form(0.20),
):
    from omnivis.grounding import zero_shot_detect

    frame = _read_image(image)
    dets = zero_shot_detect(frame, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold)
    return {"detections": [d.to_dict() for d in dets], "count": len(dets)}


# ── Tracking ─────────────────────────────────────────────────────────────

@app.post("/track")
async def track_endpoint(
    image: UploadFile = File(...),
    tracker: str = Form("bytetrack"),
    confidence: float = Form(0.25),
    imgsz: int = Form(768),
):
    from omnivis.tracking import track

    frame = _read_image(image)
    dets = track(frame, tracker=tracker, confidence=confidence, imgsz=imgsz)
    return {"detections": [d.to_dict() for d in dets], "count": len(dets)}


# ── Action Classification ───────────────────────────────────────────────

@app.post("/classify-actions")
async def classify_actions_endpoint(
    frames: List[UploadFile] = File(...),
    model: str = Form("timesformer"),
    top_k: int = Form(5),
):
    from omnivis.action_classifier import classify_actions

    video_frames = _read_frames(frames)
    actions = classify_actions(video_frames, model=model, top_k=top_k)
    return {"actions": [a.to_dict() for a in actions]}


# ── Narration ────────────────────────────────────────────────────────────

@app.post("/narrate")
async def narrate_endpoint(
    image: UploadFile = File(...),
    context: str = Form(""),
    model: str = Form("minicpm-v:4.5"),
):
    from omnivis.narration import narrate

    frame = _read_image(image)
    result = narrate(frame, context=context if context else None, model=model)
    return result.to_dict()


# ── CLIP Embeddings ──────────────────────────────────────────────────────

@app.post("/embed/image")
async def embed_image_endpoint(image: UploadFile = File(...)):
    from omnivis.clip_embeddings import embed_clip

    frame = _read_image(image)
    emb = embed_clip(frame)
    return emb.to_dict()


@app.post("/embed/text")
async def embed_text_endpoint(text: str = Form(...)):
    from omnivis.clip_embeddings import embed_text

    emb = embed_text(text)
    return emb.to_dict()


@app.post("/embed/search")
async def embed_search_endpoint(
    query: str = Form(...),
    embeddings: str = Form(...),
    top_k: int = Form(5),
):
    """Search pre-computed image embeddings by text query.

    Expects `embeddings` as a JSON string: list of {"id": ..., "vector": [...]}.
    """
    import json
    from omnivis.clip_embeddings import search_by_text

    parsed = json.loads(embeddings)
    vectors = {item["id"]: item["vector"] for item in parsed}
    results = search_by_text(query, vectors, top_k=top_k)
    return {"results": results}


@app.post("/auto-tag")
async def auto_tag_endpoint(
    image: UploadFile = File(...),
    categories: str = Form(...),
    top_k: int = Form(5),
):
    from omnivis.clip_embeddings import auto_tag

    frame = _read_image(image)
    cat_list = [c.strip() for c in categories.split(",") if c.strip()]
    tags = auto_tag(frame, cat_list, top_k=top_k)
    return {"tags": tags}


# ── Drone Detection ─────────────────────────────────────────────────────

@app.post("/drone/detect")
async def drone_detect_endpoint(
    image: UploadFile = File(...),
    confidence: float = Form(0.25),
    imgsz: int = Form(768),
    enable_heuristics: bool = Form(True),
):
    from omnivis.drone_pipeline import detect_drone

    frame = _read_image(image)
    result = detect_drone(frame, confidence=confidence, imgsz=imgsz, enable_heuristics=enable_heuristics)
    return result.to_dict() if result else {"detected": False}


# ── Streaming ────────────────────────────────────────────────────────────

@app.get("/stream/{source:path}")
async def stream_endpoint(
    source: str,
    model_type: str = Query("Detection"),
    confidence: float = Query(0.25),
    tracker: Optional[str] = Query(None),
    imgsz: int = Query(768),
    skip_frames: int = Query(0),
    jpeg_quality: int = Query(85),
):
    from omnivis.streaming import generate_detection_stream

    return StreamingResponse(
        generate_detection_stream(
            source,
            model_type=model_type,
            confidence=confidence,
            tracker=tracker,
            imgsz=imgsz,
            skip_frames=skip_frames,
            jpeg_quality=jpeg_quality,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/streams")
async def list_streams():
    from omnivis.streaming import get_stream_registry

    registry = get_stream_registry()
    return {"active_streams": registry.active_streams()}


@app.delete("/stream/{source:path}")
async def stop_stream(source: str):
    from omnivis.streaming import get_stream_registry

    registry = get_stream_registry()
    state = registry.get(source)
    if state:
        state.stop_requested = True
        return {"stopped": source}
    return JSONResponse(status_code=404, content={"error": f"Stream '{source}' not found"})
