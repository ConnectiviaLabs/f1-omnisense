# Media Intelligence Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewire Media Intelligence to use MongoDB-backed backend endpoints and Groq vision API, removing all Ollama/local-file/Vite-middleware dependencies.

**Architecture:** Add 6 endpoints to the existing `omni_vis_router.py`. Frontend fetches from these endpoints instead of local JSON files. VLM analysis uses Groq `llama-4-scout-17b-16e-instruct` instead of Ollama. CLIP search reuses the existing `get_clip_index()` / `get_clip_embedder()` from `chat_server.py`.

**Tech Stack:** FastAPI, MongoDB (marip_f1), Groq API, React/TypeScript, cv2 for frame extraction

---

### Task 1: Add video list + results endpoints to backend

**Files:**
- Modify: `pipeline/omni_vis_router.py` (append after line 396)

**Step 1: Add `GET /videos` endpoint**

Add to `pipeline/omni_vis_router.py` after the `list_models` endpoint:

```python
@router.get("/videos")
def list_videos():
    """List all videos from media_videos collection."""
    from pipeline.chat_server import get_data_db
    db = get_data_db()
    docs = list(db["media_videos"].find({}, {"_id": 0}).sort("added", -1))
    return {"videos": docs}
```

**Step 2: Add `GET /results/{filename}` endpoint**

```python
@router.get("/results/{filename:path}")
def get_video_results(filename: str):
    """Get all analysis results for a specific video."""
    from pipeline.chat_server import get_data_db
    db = get_data_db()

    gdino_doc = db["pipeline_gdino_results"].find_one(
        {"filename": filename}, {"_id": 0}
    )
    videomae_doc = db["pipeline_videomae_results"].find_one(
        {"filename": filename}, {"_id": 0}
    )
    timesformer_doc = db["pipeline_timesformer_results"].find_one(
        {"filename": filename}, {"_id": 0}
    )
    minicpm_doc = db["pipeline_minicpm_results"].find_one(
        {"filename": filename}, {"_id": 0}
    )
    vlm_doc = db["media_vlm_analyses"].find_one(
        {"filename": filename}, {"_id": 0}
    )

    return {
        "filename": filename,
        "gdino_frames": gdino_doc.get("frames", []) if gdino_doc else [],
        "videomae": {
            "total_frames": videomae_doc.get("total_frames"),
            "fps": videomae_doc.get("fps"),
            "top_predictions": videomae_doc.get("top_predictions", []),
        } if videomae_doc else None,
        "timesformer": {
            "total_frames": timesformer_doc.get("total_frames"),
            "fps": timesformer_doc.get("fps"),
            "top_predictions": timesformer_doc.get("top_predictions", []),
        } if timesformer_doc else None,
        "minicpm_narrations": minicpm_doc.get("frames", []) if minicpm_doc else [],
        "vlm_analysis": vlm_doc if vlm_doc else None,
    }
```

**Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('pipeline/omni_vis_router.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pipeline/omni_vis_router.py
git commit -m "feat(media): add video list and results endpoints to omni_vis_router"
```

---

### Task 2: Add VLM analyze endpoint (Groq vision, replaces Ollama)

**Files:**
- Modify: `pipeline/omni_vis_router.py`

**Step 1: Add VLM analyze endpoint**

Add after the results endpoint:

```python
class VLMAnalyzeRequest(BaseModel):
    filename: str


@router.post("/vlm-analyze")
async def vlm_analyze(req: VLMAnalyzeRequest):
    """Analyze video frames using Groq vision model (replaces Ollama)."""
    import base64
    import cv2
    import os
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from pipeline.chat_server import get_data_db

    media_dir = Path(__file__).resolve().parent.parent / "f1data" / "McMedia"
    video_path = media_dir / req.filename
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {req.filename}")

    # Extract 8 evenly-spaced frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // 8)
    frame_images: list[str] = []
    idx = 0
    while cap.isOpened() and len(frame_images) < 8:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_images.append(base64.b64encode(buf.tobytes()).decode())
        idx += 1
    cap.release()

    if not frame_images:
        raise HTTPException(400, "No frames extracted")

    # Load detection context from MongoDB
    db = get_data_db()
    gdino_doc = db["pipeline_gdino_results"].find_one(
        {"filename": req.filename}, {"_id": 0}
    )
    det_summary = ""
    if gdino_doc and gdino_doc.get("frames"):
        cats: dict[str, list[float]] = {}
        for f in gdino_doc["frames"]:
            for d in f.get("detections", []):
                cats.setdefault(d.get("category", "?"), []).append(d.get("score", 0))
        det_lines = sorted(cats.items(), key=lambda x: -max(x[1]))[:15]
        det_summary = "\n".join(
            f"  - {cat}: seen {len(scores)}x, best {max(scores)*100:.0f}%"
            for cat, scores in det_lines
        )

    prompt = (
        f"You are an F1 race strategy engineer at McLaren analyzing video: {req.filename}\n\n"
        f"I'm showing you {len(frame_images)} evenly-spaced frames from this video.\n\n"
        f"OBJECT DETECTION RESULTS (GroundingDINO + SAM2):\n{det_summary or 'No objects detected.'}\n\n"
        f"Based on these {len(frame_images)} frames and the model results, respond with ONLY a JSON object (no markdown, no code fences):\n\n"
        '{"scene":"1-2 sentence description","key_objects":["object1","object2"],'
        '"track_conditions":"dry/wet + observations","strategic_notes":"strategic implications",'
        '"incidents":["any incidents observed"]}'
    )

    # Build Groq vision request
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img_b64 in frame_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(500, "GROQ_API_KEY not set")

    from groq import Groq
    client = Groq(api_key=groq_key)

    t0 = time.time()
    chat = client.chat.completions.create(
        model="llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": content}],
        temperature=0.3,
        max_completion_tokens=512,
    )
    elapsed = time.time() - t0

    raw = chat.choices[0].message.content or ""
    tokens = chat.usage.completion_tokens if chat.usage else 0

    # Parse structured JSON
    import json as _json
    structured = None
    try:
        structured = _json.loads(raw)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                structured = _json.loads(m.group(0))
            except Exception:
                pass

    result = {
        "filename": req.filename,
        "analysis": raw,
        "structured": structured,
        "model": "llama-4-scout-17b-16e-instruct",
        "frames_analyzed": len(frame_images),
        "tokens": tokens,
        "time_s": round(elapsed, 2),
    }

    # Store in MongoDB
    db["media_vlm_analyses"].update_one(
        {"filename": req.filename},
        {"$set": {**result, "timestamp": datetime.now(timezone.utc)}},
        upsert=True,
    )

    return result
```

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('pipeline/omni_vis_router.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add pipeline/omni_vis_router.py
git commit -m "feat(media): add Groq vision VLM endpoint replacing Ollama"
```

---

### Task 3: Add upload + CLIP endpoints to backend

**Files:**
- Modify: `pipeline/omni_vis_router.py`

**Step 1: Add upload endpoint**

```python
@router.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    """Upload a video file and register in media_videos collection."""
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    from pipeline.chat_server import get_data_db

    ext = os.path.splitext(video.filename or "video.mp4")[1].lower()
    if ext not in (".mp4", ".webm", ".mov"):
        raise HTTPException(400, f"Only mp4/webm/mov allowed, got {ext}")

    media_dir = Path(__file__).resolve().parent.parent / "f1data" / "McMedia"
    safe_name = (video.filename or "video.mp4").replace("/", "_").replace("\\", "_")
    dest = media_dir / safe_name

    content = await video.read()
    with open(dest, "wb") as f:
        f.write(content)

    size_mb = round(len(content) / (1024 * 1024), 2)
    ts = datetime.now(timezone.utc)

    db = get_data_db()
    db["media_videos"].update_one(
        {"filename": safe_name},
        {"$set": {
            "filename": safe_name,
            "size_mb": size_mb,
            "added": ts,
            "analyzed": False,
        }},
        upsert=True,
    )

    return {
        "filename": safe_name,
        "size_mb": size_mb,
        "added": ts.isoformat(),
        "analyzed": False,
    }
```

**Step 2: Add CLIP search endpoint**

```python
@router.get("/clip/search")
def clip_search(q: str, k: int = 12):
    """CLIP text-to-image search using MongoDB-backed index."""
    from pipeline.chat_server import get_clip_index, get_clip_embedder

    index = get_clip_index()
    clip = get_clip_embedder()

    query_vec = np.array(clip.embed_text(q))
    query_vec = query_vec / np.linalg.norm(query_vec)

    image_vecs = index["_image_vecs"]
    image_norms = image_vecs / np.linalg.norm(image_vecs, axis=1, keepdims=True)
    similarities = query_vec @ image_norms.T

    top_indices = np.argsort(similarities)[::-1][:k]
    results = []
    for idx in top_indices:
        img = index["images"][int(idx)]
        results.append({
            "path": img["path"],
            "score": round(float(similarities[idx]), 4),
            "auto_tags": img["auto_tags"],
            "source_video": img["source_video"],
            "frame_index": img["frame_index"],
        })

    return {"query": q, "results": results}
```

**Step 3: Add CLIP tags endpoint**

```python
@router.get("/clip/tags")
def clip_tags():
    """Return tag summary from CLIP index."""
    try:
        from pipeline.chat_server import get_clip_index
        index = get_clip_index()
    except FileNotFoundError:
        return {"images": [], "tags": []}

    images = [
        {
            "path": img["path"],
            "auto_tags": img["auto_tags"],
            "source_video": img["source_video"],
            "frame_index": img["frame_index"],
        }
        for img in index["images"]
    ]

    tag_summary: dict[str, float] = {}
    for img in index["images"]:
        for tag in img["auto_tags"]:
            label = tag["label"]
            if label not in tag_summary or tag["score"] > tag_summary[label]:
                tag_summary[label] = tag["score"]

    top_tags = sorted(tag_summary.items(), key=lambda x: -x[1])

    return {
        "images": images,
        "tags": [{"label": t[0], "max_score": round(t[1], 4)} for t in top_tags],
    }
```

**Step 4: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('pipeline/omni_vis_router.py').read()); print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add pipeline/omni_vis_router.py
git commit -m "feat(media): add upload and CLIP search/tags endpoints"
```

---

### Task 4: Rewire MediaIntelligence.tsx to use backend endpoints

**Files:**
- Modify: `frontend/src/app/components/MediaIntelligence.tsx`

**Step 1: Remove `import { pipeline } from '../api/local'`**

This import (line 6) is the local-JSON fetcher. Remove it entirely.

**Step 2: Replace the `useEffect` data loading block (lines 392-424)**

Replace the entire `Promise.allSettled(...)` block and the `visual-tags` fetch with:

```typescript
// Load video list on mount
useEffect(() => {
  fetch('/api/omni/vis/videos')
    .then(r => r.ok ? r.json() : { videos: [] })
    .then(data => setVideos(data.videos ?? []))
    .catch(() => {})
    .finally(() => setLoading(false));

  fetch('/api/omni/vis/clip/tags')
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (data) { setAllTags(data.images); setTopTags(data.tags || []); }
    })
    .catch(() => {});
}, []);
```

**Step 3: Add per-video results loading**

Add a new `useEffect` after the video list loader that fetches results when `selectedVideo` changes:

```typescript
// Load results for selected video
useEffect(() => {
  if (!selectedVideo) {
    setGdinoData(null);
    setVideomaeData(null);
    setTimesformerData(null);
    setNarrationData(null);
    setVlmAnalysis(null);
    return;
  }
  fetch(`/api/omni/vis/results/${encodeURIComponent(selectedVideo)}`)
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (!data) return;
      // Set GDino frames keyed by filename for the existing AnnotatedVideoPlayer
      if (data.gdino_frames?.length) {
        setGdinoData({ [selectedVideo]: data.gdino_frames });
      } else {
        setGdinoData(null);
      }
      if (data.videomae) {
        setVideomaeData({ [selectedVideo]: data.videomae });
      } else {
        setVideomaeData(null);
      }
      if (data.timesformer) {
        setTimesformerData({ [selectedVideo]: data.timesformer });
      } else {
        setTimesformerData(null);
      }
      if (data.minicpm_narrations?.length) {
        setNarrationData({ [selectedVideo]: data.minicpm_narrations });
      } else {
        setNarrationData(null);
      }
      if (data.vlm_analysis) {
        setVlmAnalysis(data.vlm_analysis);
      }
    })
    .catch(() => {});
}, [selectedVideo]);
```

**Step 4: Replace `analyzeVideo` function (lines 447-506)**

Replace the GDino-polling + Ollama VLM flow with backend calls:

```typescript
const analyzeVideo = async (filename: string) => {
  setAnalyzeProgress('detecting');
  setVlmAnalysis(null);

  try {
    // Step 1: Run detection + classification via backend
    const detectRes = await fetch('/api/omni/vis/analyze-video-by-name', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, tasks: 'detect,classify' }),
    });

    if (!detectRes.ok) throw new Error(await detectRes.text());
    const detectData = await detectRes.json();

    // Update local state with fresh detection results
    if (detectData.gdino) {
      setGdinoData({ [filename]: detectData.gdino });
    }
    if (detectData.classification) {
      setVideomaeData({
        [filename]: {
          total_frames: detectData.total_frames,
          fps: detectData.fps,
          inference_time_s: 0,
          top_predictions: detectData.classification,
        },
      });
    }

    // Step 2: VLM analysis via Groq
    setAnalyzeProgress('analyzing');
    const vlmRes = await fetch('/api/omni/vis/vlm-analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename }),
    });

    if (vlmRes.ok) {
      const vlmData = await vlmRes.json();
      setVlmAnalysis(vlmData);
    }
    setAnalyzeProgress('done');
  } catch {
    setAnalyzeProgress('error');
  }
};
```

**Step 5: Replace `handleUpload` function (lines 508-518)**

```typescript
const handleUpload = async (file: File) => {
  const formData = new FormData();
  formData.append('video', file);
  const res = await fetch('/api/omni/vis/upload', {
    method: 'POST',
    body: formData,
  });
  if (res.ok) {
    const data = await res.json();
    setVideos(prev => [data, ...prev]);
  }
};
```

**Step 6: Replace `doClipSearch` function (lines 520-528)**

```typescript
const doClipSearch = async (query: string) => {
  if (!query.trim()) { setClipResults(null); return; }
  setClipSearching(true);
  try {
    const res = await fetch(`/api/omni/vis/clip/search?q=${encodeURIComponent(query)}&k=12`);
    if (res.ok) setClipResults((await res.json()).results);
  } catch { /* */ }
  finally { setClipSearching(false); }
};
```

**Step 7: Remove unused state variables**

Remove the `fusedData` state variable (line 361) since fused data is merged into gdino results. Remove any references to `fusedData` in the render section (line 657: change `const resultData = fusedData ?? gdinoData;` to `const resultData = gdinoData;`).

**Step 8: Commit**

```bash
git add frontend/src/app/components/MediaIntelligence.tsx
git commit -m "feat(media): rewire MediaIntelligence to MongoDB-backed backend endpoints"
```

---

### Task 5: Remove Vite media middleware and stale proxies

**Files:**
- Modify: `frontend/vite.config.ts`

**Step 1: Remove Vite middleware blocks**

Remove these middleware blocks from the `localDataPlugin` function in `vite.config.ts`:
- `/api/run-gdino` middleware (lines 272-316)
- `/api/gdino-status` middleware (lines 318-331)
- `/api/upload-video` middleware (lines 334-357)
- `/api/vlm-analyze` middleware (lines 360-467)
- The `activeGdinoJobs` variable (line 274)

Also remove the `execFile` import from line 6 if no longer used.

**Step 2: Remove stale proxy entries**

Remove these proxy entries from the `server.proxy` config:
- `/api/visual-search` (lines 957-961)
- `/api/visual-tags` (lines 962-966)

These are now covered by the existing `/api/omni` proxy.

**Step 3: Remove local JSON pipeline routes for media**

Remove these entries from the `routes` object in `localDataPlugin`:
- `'pipeline/gdino'` (line 57)
- `'pipeline/fused'` (line 58)
- `'pipeline/minicpm'` (line 59)
- `'pipeline/videomae'` (line 60)
- `'pipeline/timesformer'` (line 61)

**Step 4: Verify Vite config compiles**

Run: `cd frontend && npx tsc --noEmit --skipLibCheck 2>&1 | head -20`
Expected: No errors related to vite.config.ts

**Step 5: Commit**

```bash
git add frontend/vite.config.ts
git commit -m "chore(media): remove Vite media middleware and stale proxies"
```

---

### Task 6: Clean up api/local.ts and update data tracker

**Files:**
- Modify: `frontend/src/app/api/local.ts:41-48`
- Modify: `data_tracker.html`

**Step 1: Remove pipeline media entries from local.ts**

Remove or comment out lines 42-47 (gdino, fused, minicpm, videomae, timesformer, videos) from the `pipeline` object in `frontend/src/app/api/local.ts`. Keep the `pipeline` export if other entries remain (intelligence, anomaly).

**Step 2: Update data_tracker.html**

Add `media_vlm_analyses` collection entry to `data_tracker.html` with:
- Category: Media
- Writer: `omni_vis_router.py`
- Reader: `MediaIntelligence.tsx`
- Status: Active

**Step 3: Commit**

```bash
git add frontend/src/app/api/local.ts data_tracker.html
git commit -m "chore(media): clean up local.ts pipeline entries and update data tracker"
```

---

### Task 7: Verify end-to-end

**Step 1: Check Python backend syntax**

Run: `python3 -c "import ast; ast.parse(open('pipeline/omni_vis_router.py').read()); print('OK')"`
Expected: `OK`

**Step 2: Check TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit --skipLibCheck 2>&1 | grep -c 'error'`
Expected: `0`

**Step 3: Verify no remaining local-file references in MediaIntelligence**

Run: `grep -n 'pipeline\.\|api/local\|run-gdino\|gdino-status\|vlm-analyze\|upload-video\|visual-search\|visual-tags' frontend/src/app/components/MediaIntelligence.tsx`
Expected: No matches

**Step 4: Final commit if any fixes needed**

---

Plan complete and saved to `docs/plans/2026-03-06-media-intelligence-redesign.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?
