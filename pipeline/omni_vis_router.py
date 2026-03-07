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


@router.post("/vlm-analyze")
async def vlm_analyze(req: VLMAnalyzeRequest):
    """Analyze a video with Groq vision LLM (Llama 4 Scout)."""
    import base64
    import os
    import re
    import json
    import time
    from pathlib import Path

    import cv2

    from pipeline.chat_server import get_data_db

    media_dir = Path(__file__).resolve().parent.parent / "f1data" / "McMedia"
    video_path = media_dir / req.filename
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(404, f"Video not found: {req.filename}")

    # Extract 8 evenly-spaced frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video file")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * (total_frames - 1) / 7) for i in range(8)]
    frames_b64: list[str] = []
    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames_b64.append(base64.b64encode(buf.tobytes()).decode())
    cap.release()

    if not frames_b64:
        raise HTTPException(400, "No frames extracted from video")

    # Load GDino detection context
    db = get_data_db()
    gdino_doc = db["pipeline_gdino_results"].find_one(
        {"filename": req.filename}, {"_id": 0}
    )
    detection_summary = ""
    if gdino_doc and gdino_doc.get("frames"):
        from collections import Counter
        cat_counts: Counter = Counter()
        cat_best_score: dict[str, float] = {}
        for fr in gdino_doc["frames"]:
            for det in fr.get("detections", []):
                label = det.get("label") or det.get("class_name", "unknown")
                score = det.get("confidence") or det.get("score", 0.0)
                cat_counts[label] += 1
                if score > cat_best_score.get(label, 0.0):
                    cat_best_score[label] = score
        top_cats = cat_counts.most_common(15)
        lines = [f"  {lab}: {cnt}x (best {cat_best_score[lab]:.2f})" for lab, cnt in top_cats]
        detection_summary = "GDino detections:\n" + "\n".join(lines)

    # Build prompt
    prompt = (
        "You are an F1 video analyst. Analyze these frames from a McLaren F1 video.\n"
    )
    if detection_summary:
        prompt += f"\nObject detection context:\n{detection_summary}\n"
    prompt += (
        "\nReturn ONLY valid JSON with this schema:\n"
        '{"scene":"...","key_objects":[...],"track_conditions":"...",'
        '"strategic_notes":"...","incidents":[...]}\n'
    )

    # Build Groq vision request content parts
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
        model="llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": content_parts}],
        temperature=0.3,
        max_completion_tokens=512,
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

    db["media_vlm_analyses"].update_one(
        {"filename": req.filename},
        {"$set": {
            "filename": req.filename,
            "analysis": raw_text,
            "structured": structured,
            "model": "llama-4-scout-17b-16e-instruct",
            "frames_analyzed": len(frames_b64),
            "tokens": tokens,
            "time_s": elapsed,
            "updated": datetime.now(timezone.utc),
        }},
        upsert=True,
    )

    return {
        "filename": req.filename,
        "analysis": raw_text,
        "structured": structured,
        "model": "llama-4-scout-17b-16e-instruct",
        "frames_analyzed": len(frames_b64),
        "tokens": tokens,
        "time_s": elapsed,
    }


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
