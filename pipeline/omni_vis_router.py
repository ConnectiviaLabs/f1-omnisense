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


class AnalyzeByNameRequest(BaseModel):
    filename: str
    tasks: str = "detect,classify"
    confidence: float = 0.25
    max_frames: int = 30


@router.post("/analyze-video-by-name")
async def analyze_video_by_name(req: AnalyzeByNameRequest):
    """Analyze a video already on the server (f1data/McMedia/) by filename."""
    import cv2
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    media_dir = Path(__file__).resolve().parent.parent / "f1data" / "McMedia"
    video_path = media_dir / req.filename
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(404, f"Video not found: {req.filename}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    step = max(1, total_frames // req.max_frames)
    idx = 0
    while cap.isOpened() and len(frames) < req.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        raise HTTPException(400, "No frames extracted from video")

    task_list = [t.strip() for t in req.tasks.split(",")]
    results: dict = {
        "filename": req.filename,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "sampled_frames": len(frames),
    }

    if "detect" in task_list:
        from omnivis import detect as _detect
        detections = []
        for i, frame in enumerate(frames):
            dets = _detect(frame, confidence=req.confidence)
            detections.append({
                "frame_index": i * step,
                "detections": [d.to_dict() for d in dets],
            })
        results["gdino"] = detections

    if "classify" in task_list:
        from omnivis import classify_actions
        actions = classify_actions(frames)
        results["classification"] = [a.__dict__ if hasattr(a, "__dict__") else a for a in actions]

    # Store results in MongoDB
    try:
        from pymongo import MongoClient
        uri = os.environ.get("MONGODB_URI", "")
        db_name = os.environ.get("MONGODB_DB", "marip_f1")
        if uri:
            client = MongoClient(uri)
            db = client[db_name]
            ts = datetime.now(timezone.utc)
            if "gdino" in results:
                db["pipeline_gdino_results"].update_one(
                    {"filename": req.filename},
                    {"$set": {"filename": req.filename, "timestamp": ts, "frames": results["gdino"]}},
                    upsert=True,
                )
            if "classification" in results:
                db["pipeline_videomae_results"].update_one(
                    {"filename": req.filename},
                    {"$set": {
                        "filename": req.filename, "timestamp": ts,
                        "total_frames": total_frames, "fps": round(fps, 2),
                        "top_predictions": results["classification"],
                    }},
                    upsert=True,
                )
            # Mark video as analyzed
            db["media_videos"].update_one(
                {"filename": req.filename},
                {"$set": {"analyzed": True, "last_analyzed": ts}},
            )
    except Exception as e:
        logger.warning("Failed to store results in MongoDB: %s", e)

    return results


@router.get("/models")
def list_models():
    """List currently loaded models and their status."""
    try:
        from omnivis import get_model_manager
        manager = get_model_manager()
        return {"models": manager.list_loaded()}
    except Exception:
        return {"models": [], "note": "Model manager not initialized"}


# ── Media Intelligence endpoints ─────────────────────────────────────────

@router.get("/videos")
def list_videos():
    """List all videos from media_videos, sorted by added descending."""
    from pipeline.chat_server import get_data_db

    db = get_data_db()
    docs = list(
        db["media_videos"]
        .find({}, {"_id": 0})
        .sort("added", -1)
    )
    return {"videos": docs}


@router.get("/results/{filename}")
def get_results(filename: str):
    """Get all analysis results for a specific video across all pipeline collections."""
    from pipeline.chat_server import get_data_db

    db = get_data_db()

    # GDino frames
    gdino_docs = list(
        db["pipeline_gdino_results"]
        .find({"filename": filename}, {"_id": 0})
    )
    # Flatten frames from all matching docs
    gdino_frames = []
    for doc in gdino_docs:
        gdino_frames.extend(doc.get("frames", []))

    # VideoMAE classification
    videomae_doc = db["pipeline_videomae_results"].find_one(
        {"filename": filename}, {"_id": 0}
    )

    # TimeSformer classification
    timesformer_doc = db["pipeline_timesformer_results"].find_one(
        {"filename": filename}, {"_id": 0}
    )

    # MiniCPM narrations
    minicpm_doc = db["pipeline_minicpm_results"].find_one(
        {"filename": filename}, {"_id": 0}
    )

    # VLM analysis
    vlm_doc = db["media_vlm_analyses"].find_one(
        {"filename": filename}, {"_id": 0}
    )

    return {
        "filename": filename,
        "gdino_frames": gdino_frames,
        "videomae": videomae_doc,
        "timesformer": timesformer_doc,
        "minicpm_narrations": minicpm_doc.get("frames", []) if minicpm_doc else [],
        "vlm_analysis": vlm_doc,
    }


# ── VLM, Upload, CLIP endpoints ─────────────────────────────────────────

class VLMAnalyzeRequest(BaseModel):
    filename: str
    n_frames: int = 8


def _det_label(det: dict) -> str:
    """Extract label from either GDino format (category/class_name)."""
    return det.get("category") or det.get("class_name") or det.get("label") or "object"


def _det_score(det: dict) -> float:
    """Extract score from either GDino format (score/confidence)."""
    return det.get("score") or det.get("confidence") or 0.0


def _draw_annotations(frame, detections: list[dict]):
    """Draw GDino bounding boxes and labels onto a raw video frame.

    Handles both detection formats:
      Format A: {category, score, bbox} — absolute pixel coords
      Format B: {class_name, confidence, bbox, class_id} — absolute pixel coords
    """
    import cv2

    h, w = frame.shape[:2]
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        label = _det_label(det)
        score = _det_score(det)

        x1, y1, x2, y2 = [float(v) for v in bbox]
        # Normalised coords (all values 0-1) → scale to pixels
        if max(x1, y1, x2, y2) <= 1.0:
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = (0, 165, 255)  # orange BGR
        thickness = max(2, min(h, w) // 300)  # scale with resolution
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        text = f"{label} {score:.0%}"
        font_scale = max(0.4, min(h, w) / 1200)
        (tw, th_text), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # Label background — above the box, or below if near top edge
        label_y = y1 - 4 if y1 > th_text + 8 else y2 + th_text + 4
        bg_y1 = label_y - th_text - 4
        bg_y2 = label_y + 4
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 6, bg_y2), color, -1)
        cv2.putText(frame, text, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


@router.post("/vlm-analyze")
async def vlm_analyze(req: VLMAnalyzeRequest):
    """Extract N frames with GDino annotations drawn on, send to Groq vision.

    Strategy: if GDino data exists, sample N frames from the GDino-analysed
    frame indices (so annotations are guaranteed to match). Otherwise fall back
    to evenly-spaced raw frames from the video.
    """
    import base64
    import os
    import re
    import json
    import time
    from pathlib import Path
    from collections import Counter

    import cv2

    from pipeline.chat_server import get_data_db

    media_dir = Path(__file__).resolve().parent.parent / "f1data" / "McMedia"
    video_path = media_dir / req.filename
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(404, f"Video not found: {req.filename}")

    n = max(1, min(req.n_frames, 16))  # clamp 1-16

    # Load GDino detections keyed by frame_index
    db = get_data_db()
    gdino_doc = db["pipeline_gdino_results"].find_one(
        {"filename": req.filename}, {"_id": 0}
    )
    gdino_frames: list[dict] = []
    det_by_frame: dict[int, list[dict]] = {}
    if gdino_doc and gdino_doc.get("frames"):
        gdino_frames = gdino_doc["frames"]
        for fr in gdino_frames:
            fidx = fr.get("frame_index")
            if fidx is not None:
                det_by_frame[fidx] = fr.get("detections", [])

    # Decide which frame indices to extract
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video file")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if det_by_frame:
        # Sample N from the GDino-analysed frame indices (evenly spaced)
        sorted_gdino_idxs = sorted(det_by_frame.keys())
        if len(sorted_gdino_idxs) <= n:
            target_indices = sorted_gdino_idxs
        else:
            step = len(sorted_gdino_idxs) / n
            target_indices = [sorted_gdino_idxs[int(i * step)] for i in range(n)]
    else:
        # No GDino data — evenly space through the video
        target_indices = [int(i * (total_frames - 1) / max(n - 1, 1)) for i in range(n)]

    # Extract frames and draw annotations
    frames_b64: list[str] = []
    annotation_count = 0
    for target_idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw annotations if this frame has GDino detections
        dets = det_by_frame.get(target_idx, [])
        if dets:
            frame = _draw_annotations(frame, dets)
            annotation_count += len(dets)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames_b64.append(base64.b64encode(buf.tobytes()).decode())
    cap.release()

    if not frames_b64:
        raise HTTPException(400, "No frames extracted from video")

    # Build detection summary for text context
    detection_summary = ""
    if det_by_frame:
        cat_counts: Counter = Counter()
        cat_best: dict[str, float] = {}
        for dets in det_by_frame.values():
            for det in dets:
                label = _det_label(det)
                score = _det_score(det)
                cat_counts[label] += 1
                if score > cat_best.get(label, 0.0):
                    cat_best[label] = score
        lines = [f"  {lab}: {cnt}x (best {cat_best[lab]:.0%})" for lab, cnt in cat_counts.most_common(15)]
        detection_summary = "GDino detections (bounding boxes drawn on frames):\n" + "\n".join(lines)

    # Build prompt
    has_annotations = annotation_count > 0
    prompt = (
        f"You are an F1 video analyst at McLaren. I'm showing you {len(frames_b64)} frames "
        f"sampled from: {req.filename}\n\n"
    )
    if has_annotations:
        prompt += (
            f"These frames have GroundingDINO object detection annotations drawn on them — "
            f"orange bounding boxes with labels and confidence scores. "
            f"{annotation_count} detections total across the frames shown.\n\n"
        )
    if detection_summary:
        prompt += f"{detection_summary}\n\n"
    prompt += (
        "Analyze what you see in these frames. Return ONLY valid JSON (no markdown fences):\n"
        '{"scene":"2-3 sentence description of the video content and what is happening",'
        '"key_objects":["important objects/elements visible in the frames"],'
        '"track_conditions":"dry/wet/mixed, surface condition, visibility",'
        '"strategic_notes":"any tactical or strategic implications for race engineering",'
        '"incidents":["any notable events, incidents, or anomalies observed"]}\n'
    )

    # Build Groq vision request
    content_parts: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in frames_b64:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    from groq import Groq

    t0 = time.time()
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": content_parts}],
        temperature=0.3,
        max_completion_tokens=600,
    )
    elapsed = round(time.time() - t0, 2)

    raw_text = resp.choices[0].message.content or ""
    tokens = getattr(resp.usage, "total_tokens", None)

    # Parse JSON from response
    structured = None
    try:
        structured = json.loads(raw_text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw_text)
        if m:
            try:
                structured = json.loads(m.group())
            except json.JSONDecodeError:
                pass

    # Upsert into MongoDB
    from datetime import datetime, timezone

    result = {
        "filename": req.filename,
        "analysis": raw_text,
        "structured": structured,
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "frames_analyzed": len(frames_b64),
        "annotations_drawn": annotation_count,
        "tokens": tokens,
        "time_s": elapsed,
    }

    db["media_vlm_analyses"].update_one(
        {"filename": req.filename},
        {"$set": {**result, "updated": datetime.now(timezone.utc)}},
        upsert=True,
    )

    return result


@router.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    """Upload a video file to McMedia and register in MongoDB."""
    import os
    import re as _re
    from datetime import datetime, timezone
    from pathlib import Path

    from pipeline.chat_server import get_data_db

    original = video.filename or "upload.mp4"
    ext = os.path.splitext(original)[1].lower()
    if ext not in (".mp4", ".webm", ".mov"):
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: .mp4, .webm, .mov")

    # Sanitize filename
    basename = os.path.splitext(original)[0]
    safe_base = _re.sub(r"[^\w\-.]", "_", basename)
    safe_filename = f"{safe_base}{ext}"

    media_dir = Path(__file__).resolve().parent.parent / "f1data" / "McMedia"
    media_dir.mkdir(parents=True, exist_ok=True)
    dest = media_dir / safe_filename

    content = await video.read()
    with open(dest, "wb") as f:
        f.write(content)

    size_mb = round(len(content) / (1024 * 1024), 2)
    now = datetime.now(timezone.utc)

    db = get_data_db()
    db["media_videos"].update_one(
        {"filename": safe_filename},
        {"$set": {
            "filename": safe_filename,
            "size_mb": size_mb,
            "added": now,
            "analyzed": False,
        }},
        upsert=True,
    )

    return {
        "filename": safe_filename,
        "size_mb": size_mb,
        "added": now.isoformat(),
        "analyzed": False,
    }


@router.get("/clip/search")
def clip_search(q: str, k: int = 12):
    """CLIP text-to-image search across indexed frames."""
    from pipeline.chat_server import get_clip_index, get_clip_embedder

    try:
        index = get_clip_index()
    except FileNotFoundError:
        raise HTTPException(404, "CLIP index not found")

    embedder = get_clip_embedder()
    query_vec = np.array(embedder.embed_text(q), dtype=np.float32)

    images = index.get("images", [])
    if not images:
        return {"query": q, "results": []}

    # Build matrix of embeddings
    emb_matrix = np.array(
        [img["embedding"] for img in images], dtype=np.float32
    )
    # Cosine similarity
    norms_q = np.linalg.norm(query_vec) + 1e-9
    norms_db = np.linalg.norm(emb_matrix, axis=1) + 1e-9
    scores = emb_matrix @ query_vec / (norms_db * norms_q)

    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        img = images[idx]
        results.append({
            "path": img.get("path", ""),
            "score": round(float(scores[idx]), 4),
            "auto_tags": img.get("auto_tags", []),
            "source_video": img.get("source_video", ""),
            "frame_index": img.get("frame_index", None),
        })

    return {"query": q, "results": results}


@router.get("/clip/tags")
def clip_tags():
    """Return all indexed images with tags and a tag summary."""
    from pipeline.chat_server import get_clip_index

    try:
        index = get_clip_index()
    except FileNotFoundError:
        return {"images": [], "tags": []}

    images = index.get("images", [])

    # Build tag summary: unique labels with max scores
    tag_max: dict[str, float] = {}
    for img in images:
        for tag in img.get("auto_tags", []):
            label = tag.get("label", "") if isinstance(tag, dict) else str(tag)
            score = tag.get("score", 0.0) if isinstance(tag, dict) else 0.0
            if score > tag_max.get(label, 0.0):
                tag_max[label] = score

    tags_sorted = sorted(
        [{"label": lab, "max_score": round(sc, 4)} for lab, sc in tag_max.items()],
        key=lambda t: t["max_score"],
        reverse=True,
    )

    return {"images": images, "tags": tags_sorted}
