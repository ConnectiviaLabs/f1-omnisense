"""RAG Chatbot server — F1 Knowledge Base Agent.

Queries MongoDB Atlas vector search for context, then generates
answers via Groq LLM (llama-3.3-70b-versatile).

Usage:
    python pipeline/chat_server.py          # Runs on port 8300
"""

from __future__ import annotations

import logging
import os
import sys
import json
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
# OmniSuite packages
sys.path.insert(0, str(Path(__file__).parent.parent / "omnisuitef1"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from groq import Groq
from pipeline.vectorstore import get_vector_store
from pipeline.embeddings import NomicEmbedder
from pipeline.model_3d_server import router as model_3d_router, mount_3d_static
from pipeline.omni_health_router import router as omni_health_router
from pipeline.omni_analytics_router import router as omni_analytics_router
from pipeline.omni_rag_router import router as omni_rag_router
from pipeline.omni_kex_router import router as omni_kex_router
from pipeline.omni_doc_router import router as omni_doc_router
from pipeline.omni_data_router import router as omni_data_router
from pipeline.omni_bedding_router import router as omni_bedding_router
from pipeline.omni_vis_router import router as omni_vis_router
from pipeline.omni_dapt_router import router as omni_dapt_router
from pipeline.opponents.server import router as opponents_router, init_profiler_with_db
from pipeline.updater.server import router as updater_router

# ── Config ───────────────────────────────────────────────────────────────

GROQ_MODEL = os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile")
PORT = int(os.getenv("PORT", os.getenv("API_PORT", "8300")))
USE_OMNIRAG = os.getenv("USE_OMNIRAG", "").lower() in ("1", "true", "yes")

app = FastAPI(title="F1 OmniSense API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rewrite /api/X → /api/local/X (frontend expects short paths, backend uses /api/local/)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as _Request

class _RewriteMiddleware(BaseHTTPMiddleware):
    _PREFIXES = (
        "/api/openf1/", "/api/jolpica/", "/api/driver_intel/",
        "/api/circuit_intel/", "/api/pipeline/", "/api/constructor_profiles",
        "/api/mclaren/",
    )
    async def dispatch(self, request: _Request, call_next):
        path = request.scope["path"]
        for prefix in self._PREFIXES:
            if path.startswith(prefix):
                request.scope["path"] = path.replace("/api/", "/api/local/", 1)
                break
        return await call_next(request)

app.add_middleware(_RewriteMiddleware)

# Mount 3D model generation routes
app.include_router(model_3d_router)
mount_3d_static(app)

# Mount OmniSuite routers
app.include_router(omni_health_router)
app.include_router(omni_analytics_router)
app.include_router(omni_rag_router)
app.include_router(omni_kex_router)
app.include_router(omni_doc_router)
app.include_router(omni_data_router)
app.include_router(omni_bedding_router)
app.include_router(omni_vis_router)
app.include_router(omni_dapt_router)
app.include_router(opponents_router)
app.include_router(updater_router)

# Lazy-init singletons
_groq: Groq | None = None
_vs = None
_embedder: NomicEmbedder | None = None
_clip_index: dict | None = None
_clip_embedder = None

CLIP_INDEX_PATH = Path(__file__).parent.parent / "f1data" / "McMedia" / "clip_index.json"


def get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq


def get_vs():
    global _vs
    if _vs is None:
        _vs = get_vector_store()
    return _vs


def get_embedder() -> NomicEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = NomicEmbedder()
    return _embedder


def get_clip_index() -> dict:
    """Load pre-built CLIP index from MongoDB (preferred) or disk fallback."""
    global _clip_index
    if _clip_index is None:
        # Try MongoDB first
        try:
            db = get_data_db()
            doc = db["clip_index"].find_one({"_type": "clip_index"}, {"_id": 0, "_type": 0})
            if doc and "images" in doc:
                _clip_index = doc
        except Exception:
            pass
        # Fall back to local file
        if _clip_index is None:
            if not CLIP_INDEX_PATH.exists():
                raise FileNotFoundError(
                    "CLIP index not found in MongoDB or on disk. "
                    "Run: python pipeline/push_clip_index_to_mongo.py"
                )
            with open(CLIP_INDEX_PATH) as f:
                _clip_index = json.load(f)
        # Pre-compute numpy arrays for fast search
        _clip_index["_image_vecs"] = np.array(
            [img["embedding"] for img in _clip_index["images"]]
        )
    return _clip_index


def get_clip_embedder():
    """Lazy-load CLIP embedder for query embedding."""
    global _clip_embedder
    if _clip_embedder is None:
        from pipeline.embeddings import CLIPEmbedder
        _clip_embedder = CLIPEmbedder()
    return _clip_embedder


# ── Models ───────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    data_types: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# ── System Prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the F1 OmniSense Knowledge Agent — an expert on Formula 1 technical regulations, car specifications, equipment, and engineering standards.

You have access to a knowledge base extracted from FIA 2024 Technical Regulations and related engineering documents. When answering questions:

1. Use ONLY the provided context to answer. If the context doesn't contain enough information, say so clearly.
2. Cite specific regulation IDs (e.g., "Article 3.5.2") and page numbers when available.
3. Be precise with numerical values, units, and tolerances.
4. For dimensional specifications, always include the value and unit.
5. When discussing equipment, reference tags and types.
6. Keep answers concise but thorough. Use bullet points for lists.
7. If a question is ambiguous, briefly clarify what you're answering.

You are speaking with F1 engineers and technical staff — use appropriate technical language."""


# ── RAG Pipeline ─────────────────────────────────────────────────────────

def _text_search_fallback(query: str, k: int = 8, data_types: list[str] | None = None) -> list[dict]:
    """Fallback: keyword search on f1_knowledge when vector search is unavailable."""
    vs = get_vs()
    coll = vs.collection
    keywords = [w for w in query.split() if len(w) > 2]
    if not keywords:
        keywords = query.split()
    regex = "|".join(keywords)
    try:
        match_filter: dict = {"page_content": {"$regex": regex, "$options": "i"}}
        if data_types:
            match_filter["metadata.data_type"] = {"$in": data_types}
        results = list(
            coll.find(
                match_filter,
                {"page_content": 1, "metadata": 1, "_id": 0},
            ).limit(k)
        )
    except Exception:
        results = list(coll.find({}, {"page_content": 1, "metadata": 1, "_id": 0}).limit(k))
    sources = []
    for r in results:
        meta = r.get("metadata", {})
        sources.append({
            "content": r.get("page_content", ""),
            "data_type": meta.get("data_type", ""),
            "category": meta.get("category", ""),
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
        })
    return sources


def retrieve_context(query: str, k: int = 8, data_types: list[str] | None = None) -> list[dict]:
    """Retrieve relevant documents using text search (fast) with optional
    vector search upgrade if embedder is already loaded."""
    global _embedder
    vs_filter = {"metadata.data_type": {"$in": data_types}} if data_types else None
    # If embedder is already loaded, use vector search
    if _embedder is not None:
        try:
            vs = get_vs()
            query_vec = _embedder.embed([query])[0]
            docs = vs.similarity_search(query, k=k, query_embedding=query_vec, filter=vs_filter)
            sources = []
            for doc in docs:
                sources.append({
                    "content": doc.page_content,
                    "data_type": doc.metadata.get("data_type", ""),
                    "category": doc.metadata.get("category", ""),
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", 0),
                })
            return sources
        except Exception as e:
            print(f"  Vector search failed ({e}), falling back to text search")
    # Fast text search — no model loading required
    return _text_search_fallback(query, k, data_types=data_types)


def build_rag_prompt(query: str, sources: list[dict], history: list[ChatMessage]) -> list[dict]:
    """Build the full prompt with system, context, history, and user query."""
    context_parts = []
    for i, src in enumerate(sources, 1):
        context_parts.append(
            f"[{i}] ({src['data_type']}/{src['category']}) "
            f"Page {src['page']} — {src['source']}\n{src['content']}"
        )
    context_block = "\n\n".join(context_parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (last 10 messages)
    for msg in history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})

    # User message with context
    user_prompt = f"""CONTEXT FROM KNOWLEDGE BASE:
{context_block}

USER QUESTION:
{query}

Answer based on the context above. Cite regulation IDs and page numbers where applicable."""

    messages.append({"role": "user", "content": user_prompt})
    return messages


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """RAG chat endpoint. Delegates to OmniRAG when USE_OMNIRAG=true."""
    if USE_OMNIRAG:
        from pipeline.omni_rag_router import chat as _omni_chat, ChatRequest as _OmniReq
        result = _omni_chat(_OmniReq(message=req.message))
        return ChatResponse(answer=result["answer"], sources=result.get("sources", []))

    # 1. Retrieve context
    sources = retrieve_context(req.message, k=8, data_types=req.data_types)

    # 2. Build prompt
    messages = build_rag_prompt(req.message, sources, req.history)

    # 3. Generate answer via Groq
    groq = get_groq()
    completion = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )

    answer = completion.choices[0].message.content

    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    vs = get_vs()
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "documents": vs.count(),
        "services": ["knowledge_agent", "3d_model_gen"],
    }


# ── CLIP Visual Search ───────────────────────────────────────────────

@app.get("/visual-search")
def visual_search(q: str, k: int = 8):
    """Search images by text query using CLIP embeddings."""
    index = get_clip_index()
    clip = get_clip_embedder()

    # Embed text query into CLIP space
    query_vec = np.array(clip.embed_text(q))
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Cosine similarity against all image embeddings
    image_vecs = index["_image_vecs"]
    image_norms = image_vecs / np.linalg.norm(image_vecs, axis=1, keepdims=True)
    similarities = query_vec @ image_norms.T

    # Top-k results
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


@app.get("/api/visual-tags")
def visual_tags():
    """Return all images with their auto-tags for gallery filtering."""
    try:
        index = get_clip_index()
    except FileNotFoundError:
        return {"images": [], "tags": [], "stats": {}}

    images = []
    for img in index["images"]:
        images.append({
            "path": img["path"],
            "auto_tags": img["auto_tags"],
            "source_video": img["source_video"],
            "frame_index": img["frame_index"],
        })

    # Collect all unique tag labels with their max scores
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
        "stats": index["stats"],
    }


# ── Document Upload & Ingestion ──────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".json", ".md"}


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    import fitz
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx."""
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_plain(file_path: str) -> str:
    """Read plain text files (txt, csv, json, md)."""
    return Path(file_path).read_text(encoding="utf-8", errors="replace")


EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_plain,
    ".csv": extract_text_from_plain,
    ".json": extract_text_from_plain,
    ".md": extract_text_from_plain,
}


@app.post("/upload")
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document, extract text, embed, and ingest into the knowledge base.

    Tries OmniDoc enhanced processing first (richer metadata, table extraction),
    falls back to legacy extractor if OmniDoc is unavailable.
    """
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    # Try OmniDoc route first (supports more formats, extracts tables/tags)
    try:
        from pipeline.omni_doc_router import upload_and_ingest
        # Reset file position and delegate
        await file.seek(0)
        return await upload_and_ingest(file=file)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("OmniDoc upload failed (%s), falling back to legacy", e)

    # Legacy fallback
    await file.seek(0)

    if ext not in ALLOWED_EXTENSIONS:
        return {"filename": filename, "status": "error",
                "error": f"Unsupported format: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract text
        extractor = EXTRACTORS[ext]
        text = extractor(tmp_path)

        if not text.strip():
            return {"filename": filename, "status": "error", "error": "No text extracted from file"}

        # Chunk text
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        if not chunks:
            return {"filename": filename, "status": "error", "error": "No chunks created"}

        # Build LangChain Documents
        from langchain_core.documents import Document
        docs = []
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "data_type": "uploaded_document",
                    "category": "user_upload",
                    "source": filename,
                    "chunk": i + 1,
                    "total_chunks": len(chunks),
                },
            ))

        # Embed
        embedder = get_embedder()
        texts = [doc.page_content for doc in docs]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            embeddings.extend(embedder.embed(batch))

        # Upsert to Atlas
        vs = get_vs()
        count = vs.upsert_documents(docs, embeddings)

        return {
            "filename": filename,
            "status": "ok",
            "chunks": count,
            "text_length": len(text),
        }

    except Exception as e:
        return {"filename": filename, "status": "error", "error": str(e)}
    finally:
        os.unlink(tmp_path)


# ── Data API — serve MongoDB collections to frontend ─────────────────────

from pymongo import MongoClient as _MongoClient

_data_client: _MongoClient | None = None
_data_db = None

def get_data_db():
    global _data_client, _data_db
    if _data_db is None:
        _data_client = _MongoClient(os.getenv("MONGODB_URI", ""))
        _data_db = _data_client[os.getenv("MONGODB_DB", "marip_f1")]
        # Share this connection with the opponents profiler
        init_profiler_with_db(_data_db)
    return _data_db

import base64 as _b64
from fastapi.responses import Response as _Response

@app.get("/media/{folder}/{filename}")
async def serve_media_frame(folder: str, filename: str):
    """Serve detection frame images from MongoDB media_frames collection."""
    path = f"{folder}/{filename}"
    db = get_data_db()
    doc = db["media_frames"].find_one({"path": path}, {"data_b64": 1, "content_type": 1})
    if doc:
        return _Response(content=_b64.b64decode(doc["data_b64"]), media_type=doc.get("content_type", "image/jpeg"))
    return _Response(status_code=404, content=b"Not found")

@app.get("/api/local/jolpica/race_results")
async def jolpica_race_results():
    db = get_data_db()
    docs = list(db["jolpica_race_results"].find({}, {"_id": 0, "ingested_at": 0}))
    return docs

@app.get("/api/local/jolpica/driver_standings")
async def jolpica_driver_standings():
    db = get_data_db()
    docs = list(db["jolpica_driver_standings"].find({}, {"_id": 0, "ingested_at": 0}))
    return docs

@app.get("/api/local/jolpica/constructor_standings")
async def jolpica_constructor_standings():
    db = get_data_db()
    docs = list(db["jolpica_constructor_standings"].find({}, {"_id": 0, "ingested_at": 0}))
    return docs

@app.get("/api/local/constructor_profiles")
async def constructor_profiles(
    season: int | None = None,
    constructor_id: str | None = None,
):
    """Team performance profiles aggregated from race results, qualifying, pit stops, and telemetry."""
    db = get_data_db()
    filt: dict = {}
    if season:
        filt["season"] = season
    if constructor_id:
        filt["constructor_id"] = constructor_id
    docs = list(db["constructor_profiles"].find(filt, {"_id": 0, "updated_at": 0}))
    docs.sort(key=lambda d: (d.get("season", 0), d.get("championship_position") or 99))
    return docs

@app.get("/api/local/constructor_profiles/{constructor_id}")
async def constructor_profile_detail(constructor_id: str, season: int | None = None):
    """Single constructor profile, optionally filtered by season."""
    db = get_data_db()
    filt: dict = {"constructor_id": constructor_id}
    if season:
        filt["season"] = season
    docs = list(db["constructor_profiles"].find(filt, {"_id": 0, "updated_at": 0}))
    if not docs:
        return {"error": f"No profile found for {constructor_id}"}
    docs.sort(key=lambda d: d.get("season", 0))
    return docs[0] if season else docs

@app.get("/api/local/jolpica/qualifying")
async def jolpica_qualifying(season: int | None = None, driver: str | None = None):
    db = get_data_db()
    filt: dict = {}
    if season:
        filt["season"] = season
    if driver:
        filt["driver_code"] = driver.upper()
    return list(db["jolpica_qualifying"].find(filt, {"_id": 0}).limit(5000))

@app.get("/api/local/jolpica/circuits")
async def jolpica_circuits():
    db = get_data_db()
    pipeline = [
        {"$group": {"_id": "$circuit_id", "race_name": {"$first": "$race_name"}, "races": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    return list(db["jolpica_race_results"].aggregate(pipeline))

@app.get("/api/local/jolpica/pit_stops")
async def jolpica_pit_stops(season: int | None = None, circuit: str | None = None):
    db = get_data_db()
    filt: dict = {}
    if season:
        filt["season"] = season
    if circuit:
        filt["circuit_id"] = circuit
    return list(db["jolpica_pit_stops"].find(filt, {"_id": 0}).limit(6000))

@app.get("/api/local/jolpica/sprint_results")
async def jolpica_sprint_results(season: int | None = None):
    db = get_data_db()
    filt: dict = {}
    if season:
        filt["season"] = season
    return list(db["jolpica_sprint_results"].find(filt, {"_id": 0}))

@app.get("/api/local/jolpica/lap_times")
async def jolpica_lap_times():
    return []

@app.get("/api/local/jolpica/drivers")
async def jolpica_drivers():
    db = get_data_db()
    pipeline = [
        {"$group": {"_id": "$driver_id", "code": {"$first": "$driver_code"}, "constructor": {"$last": "$constructor_id"}, "races": {"$sum": 1}}},
        {"$sort": {"races": -1}},
    ]
    return list(db["jolpica_race_results"].aggregate(pipeline))

@app.get("/api/local/jolpica/seasons")
async def jolpica_seasons():
    db = get_data_db()
    return sorted(db["jolpica_race_results"].distinct("season"))

# ── Driver Intelligence endpoints ─────────────────────────────────────

@app.get("/api/local/driver_intel/performance_markers")
async def driver_intel_performance_markers(driver: str | None = None):
    db = get_data_db()
    filt: dict = {}
    if driver:
        filt["Driver"] = driver.upper()
    return list(db["driver_performance_markers"].find(filt, {"_id": 0}))

@app.get("/api/local/driver_intel/overtake_profiles")
async def driver_intel_overtake_profiles(driver: str | None = None):
    db = get_data_db()
    filt: dict = {}
    if driver:
        filt["driver_code"] = driver.upper()
    return list(db["driver_overtake_profiles"].find(filt, {"_id": 0}))

@app.get("/api/local/driver_intel/telemetry_profiles")
async def driver_intel_telemetry_profiles(driver: str | None = None):
    db = get_data_db()
    filt: dict = {}
    if driver:
        filt["driver_code"] = driver.upper()
    return list(db["driver_telemetry_profiles"].find(filt, {"_id": 0}))

# ── Circuit Intelligence endpoints ────────────────────────────────────

@app.get("/api/local/circuit_intel/circuits")
async def circuit_intel_circuits(circuit: str | None = None):
    db = get_data_db()
    filt: dict = {}
    if circuit:
        filt["circuit_slug"] = circuit
    # Exclude heavy coordinate arrays when listing all circuits
    proj: dict = {"_id": 0}
    if not circuit:
        proj["coordinates"] = 0
        proj["bbox"] = 0
    return list(db["circuit_intelligence"].find(filt, proj))

@app.get("/api/local/circuit_intel/pit_loss")
async def circuit_intel_pit_loss(circuit: str | None = None):
    db = get_data_db()
    filt: dict = {}
    if circuit:
        filt["circuit"] = circuit
    return list(db["circuit_pit_loss_times"].find(filt, {"_id": 0}))

@app.get("/api/local/circuit_intel/air_density")
async def circuit_intel_air_density(circuit: str | None = None, year: int | None = None):
    db = get_data_db()
    filt: dict = {}
    if circuit:
        filt["circuit_slug"] = circuit
    if year:
        filt["year"] = year
    return list(db["race_air_density"].find(filt, {"_id": 0}))

@app.get("/api/local/pipeline/anomaly")
async def pipeline_anomaly():
    db = get_data_db()
    snapshot = db["anomaly_scores_snapshot"].find_one({}, {"_id": 0})
    return snapshot or {}

@app.get("/api/local/pipeline/intelligence")
async def pipeline_intelligence():
    """Serve regulations, equipment, dimensions, materials from f1_knowledge."""
    db = get_data_db()
    rules = []
    equipment = []
    dimensional_data = []
    material_specs = []

    for doc in db["f1_knowledge"].find({}, {"_id": 0, "embedding": 0}):
        meta = doc.get("metadata", {})
        dt = meta.get("data_type", "")
        content = doc.get("page_content", "")

        if dt == "regulation":
            # Parse description from page_content
            desc_lines = content.split("\n")
            description = desc_lines[1] if len(desc_lines) > 1 else content
            rules.append({
                "id": meta.get("rule_id", ""),
                "category": meta.get("category", ""),
                "description": description,
                "value": None,
                "unit": None,
                "condition": None,
                "reference": None,
                "severity": meta.get("severity", "info"),
                "source_standard": meta.get("source", ""),
                "_source": meta.get("source", ""),
                "_page": meta.get("page", 0),
            })
        elif dt == "equipment":
            # Parse equipment fields from page_content
            lines = content.split("\n")
            eq_type = ""
            eq_desc = ""
            eq_location = None
            for line in lines:
                if line.startswith("Type: "):
                    eq_type = line[6:]
                elif line.startswith("Location: "):
                    eq_location = line[10:]
                elif not line.startswith("["):
                    eq_desc = line
            equipment.append({
                "tag": meta.get("tag", ""),
                "type": eq_type or meta.get("category", ""),
                "description": eq_desc,
                "kks": "",
                "specs": {},
                "location_description": eq_location,
                "_source": meta.get("source", ""),
                "_page": meta.get("page", 0),
            })
        elif dt == "dimension":
            # Parse dimensional data from page_content
            lines = content.split("\n")
            dimension_desc = lines[1] if len(lines) > 1 else ""
            value = None
            unit = ""
            for line in lines:
                if line.startswith("Value: "):
                    val_str = line[7:].strip()
                    parts = val_str.split()
                    try:
                        value = float(parts[0])
                        unit = " ".join(parts[1:]) if len(parts) > 1 else ""
                    except (ValueError, IndexError):
                        value = val_str
            dimensional_data.append({
                "component": meta.get("component", ""),
                "dimension": dimension_desc,
                "value": value,
                "unit": unit,
                "_source": meta.get("source", ""),
                "_page": meta.get("page", 0),
            })
        elif dt == "material":
            lines = content.split("\n")
            application = ""
            for line in lines:
                if line.startswith("Application: "):
                    application = line[13:]
            material_specs.append({
                "material": meta.get("material", ""),
                "application": application,
                "properties": {},
                "_source": meta.get("source", ""),
                "_page": meta.get("page", 0),
            })

    return {
        "documents": [],
        "rules": rules,
        "equipment": equipment,
        "dimensional_data": dimensional_data,
        "material_specs": material_specs,
        "stats": {
            "total_pages": 0,
            "total_rules": len(rules),
            "total_equipment": len(equipment),
            "total_dimensions": len(dimensional_data),
            "total_materials": len(material_specs),
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_cost_usd": 0,
            "total_latency_s": 0,
        },
    }

@app.get("/api/local/pipeline/gdino")
async def pipeline_gdino():
    db = get_data_db()
    docs = list(db["pipeline_gdino_results"].find({}, {"_id": 0}).sort("timestamp", -1).limit(100))
    return {"results": docs, "count": len(docs)}

@app.get("/api/local/pipeline/fused")
async def pipeline_fused():
    db = get_data_db()
    docs = list(db["pipeline_fused_results"].find({}, {"_id": 0}).sort("timestamp", -1).limit(100))
    return {"results": docs, "count": len(docs)}

@app.get("/api/local/pipeline/minicpm")
async def pipeline_minicpm():
    db = get_data_db()
    docs = list(db["pipeline_minicpm_results"].find({}, {"_id": 0}).sort("timestamp", -1).limit(100))
    return {"results": docs, "count": len(docs)}

@app.get("/api/local/pipeline/videomae")
async def pipeline_videomae():
    db = get_data_db()
    docs = list(db["pipeline_videomae_results"].find({}, {"_id": 0}).sort("timestamp", -1).limit(100))
    return {"results": docs, "count": len(docs)}

@app.get("/api/local/pipeline/timesformer")
async def pipeline_timesformer():
    db = get_data_db()
    docs = list(db["pipeline_timesformer_results"].find({}, {"_id": 0}).sort("timestamp", -1).limit(100))
    return {"results": docs, "count": len(docs)}

@app.get("/api/local/pipeline/videos")
async def pipeline_videos():
    """List available videos from media_videos collection."""
    db = get_data_db()
    docs = list(db["media_videos"].find({}, {"_id": 0}).sort("added", -1))
    return {"videos": docs, "count": len(docs)}


# ── Strategy Tracker ──

@app.get("/api/local/strategy/tracker")
async def strategy_tracker(session_key: int = None, year: int = None):
    """
    Competitor Strategy Tracker: shows each driver's stint timeline,
    compound choices, and predicted next pit window based on tyre degradation curves.
    """
    db = get_data_db()

    # Find session
    if session_key:
        sess = db["openf1_sessions"].find_one({"session_key": session_key, "session_type": "Race"}, {"_id": 0})
    elif year:
        sess = db["openf1_sessions"].find_one({"session_type": "Race", "year": year}, {"_id": 0}, sort=[("session_key", -1)])
    else:
        sess = db["openf1_sessions"].find_one({"session_type": "Race"}, {"_id": 0}, sort=[("session_key", -1)])

    if not sess:
        return {"error": "No race session found", "drivers": []}

    sk = sess["session_key"]
    circuit = sess.get("circuit_short_name", "")

    # Get stints for this session
    stints = list(db["openf1_stints"].find(
        {"session_key": sk},
        {"_id": 0}
    ).sort([("driver_number", 1), ("stint_number", 1)]))

    # Get driver names
    drivers_raw = list(db["openf1_drivers"].find(
        {"session_key": sk},
        {"_id": 0, "driver_number": 1, "name_acronym": 1, "team_name": 1}
    ))
    drv_map = {}
    for d in drivers_raw:
        drv_map[d["driver_number"]] = {"name": d.get("name_acronym", ""), "team": d.get("team_name", "")}

    # Load tyre degradation curves for this circuit
    deg_curves = list(db["tyre_degradation_curves"].find(
        {"circuit": {"$regex": circuit, "$options": "i"}},
        {"_id": 0, "compound": 1, "cliff_lap": 1, "coefficients": 1, "intercept": 1}
    ))
    cliff_laps = {c["compound"]: c.get("cliff_lap") for c in deg_curves if c.get("cliff_lap")}

    # Get total laps from actual data
    max_lap_doc = db["openf1_laps"].find_one(
        {"session_key": sk}, {"lap_number": 1}, sort=[("lap_number", -1)]
    )
    total_laps = max_lap_doc["lap_number"] if max_lap_doc else 57

    # Build per-driver strategy timeline
    driver_strategies = {}
    for s in stints:
        dn = s["driver_number"]
        if dn not in driver_strategies:
            info = drv_map.get(dn, {"name": str(dn), "team": "Unknown"})
            driver_strategies[dn] = {
                "driver_number": dn,
                "driver": info["name"],
                "team": info["team"],
                "stints": [],
                "total_stops": 0,
            }

        stint_laps = (s.get("lap_end", 0) or 0) - (s.get("lap_start", 0) or 0) + 1
        compound = s.get("compound", "UNKNOWN")
        cliff = cliff_laps.get(compound)

        stint_info = {
            "stint_number": s.get("stint_number", 0),
            "compound": compound,
            "lap_start": s.get("lap_start"),
            "lap_end": s.get("lap_end"),
            "stint_laps": stint_laps,
            "tyre_age_at_start": s.get("tyre_age_at_start", 0),
            "cliff_lap": cliff,
        }

        # Predict pit window: when tyre reaches cliff or degradation exceeds threshold
        if cliff and s.get("lap_start"):
            effective_life = stint_laps + (s.get("tyre_age_at_start", 0) or 0)
            laps_until_cliff = max(0, cliff - effective_life)
            stint_info["predicted_pit_window"] = (s.get("lap_end", 0) or 0) + laps_until_cliff
        else:
            stint_info["predicted_pit_window"] = None

        driver_strategies[dn]["stints"].append(stint_info)

    # Count pit stops (stints - 1)
    for dn, ds in driver_strategies.items():
        ds["total_stops"] = max(0, len(ds["stints"]) - 1)

    drivers_list = sorted(driver_strategies.values(), key=lambda x: x["driver"])

    return {
        "session": {
            "session_key": sk,
            "circuit": circuit,
            "year": sess.get("year"),
            "meeting": f'{sess.get("country_name", "")} GP',
            "total_laps": total_laps,
        },
        "cliff_laps": cliff_laps,
        "drivers": drivers_list,
        "count": len(drivers_list),
    }


@app.get("/api/local/strategy/battle-intel")
async def strategy_battle_intel(session_key: int = None):
    """Per-driver gap evolution insights: undercut threats, closing trends."""
    db = get_data_db()

    # Find session — pick the latest race that has interval data
    if session_key:
        sess = db["openf1_sessions"].find_one({"session_key": session_key, "session_type": "Race"}, {"_id": 0})
    else:
        # Get session keys that actually have intervals
        interval_sks = db["openf1_intervals"].distinct("session_key")
        if interval_sks:
            sess = db["openf1_sessions"].find_one(
                {"session_key": {"$in": interval_sks}, "session_type": "Race"},
                {"_id": 0}, sort=[("session_key", -1)]
            )
        else:
            sess = None
    if not sess:
        return {"battles": [], "session_key": None}

    sk = sess["session_key"]

    # Driver name map
    drivers_raw = list(db["openf1_drivers"].find(
        {"session_key": sk},
        {"_id": 0, "driver_number": 1, "name_acronym": 1, "team_name": 1}
    ))
    drv_map = {d["driver_number"]: {"name": d.get("name_acronym", ""), "team": d.get("team_name", "")} for d in drivers_raw}

    # Get the last ~20 laps of interval data (most recent readings)
    intervals = list(db["openf1_intervals"].find(
        {"session_key": sk},
        {"_id": 0, "driver_number": 1, "gap_to_leader": 1, "interval": 1, "date": 1}
    ).sort("date", -1).limit(2000))

    if not intervals:
        return {"battles": [], "session_key": sk}

    # Group by driver, take last 5 readings per driver
    from collections import defaultdict
    by_driver = defaultdict(list)
    for iv in intervals:
        dn = iv.get("driver_number")
        if dn is not None:
            by_driver[dn].append(iv)

    battles = []
    for dn, readings in by_driver.items():
        readings = sorted(readings, key=lambda x: x.get("date", ""))
        recent = readings[-5:]  # last 5 readings
        info = drv_map.get(dn, {"name": str(dn), "team": "Unknown"})

        # Latest gap
        latest = recent[-1]
        gap = latest.get("gap_to_leader")
        interval_val = latest.get("interval")

        # Parse numeric values
        try:
            gap_num = float(gap) if gap is not None else None
        except (ValueError, TypeError):
            gap_num = None
        try:
            interval_num = float(interval_val) if interval_val is not None else None
        except (ValueError, TypeError):
            interval_num = None

        # Gap trend from last 5 readings
        gap_values = []
        for r in recent:
            try:
                g = float(r.get("gap_to_leader", 0))
                gap_values.append(g)
            except (ValueError, TypeError):
                pass

        trend = 0  # 0 = stable
        if len(gap_values) >= 3:
            delta = gap_values[-1] - gap_values[0]
            if delta < -0.3:
                trend = 1   # closing
            elif delta > 0.3:
                trend = -1  # falling back

        # Undercut threat: within 1.5s of car ahead
        undercut = False
        if interval_num is not None and 0 < abs(interval_num) < 1.5:
            undercut = True

        battles.append({
            "driver": info["name"],
            "team": info["team"],
            "driver_number": dn,
            "gap_to_leader": round(gap_num, 2) if gap_num is not None else None,
            "interval": round(interval_num, 2) if interval_num is not None else None,
            "trend": trend,
            "undercut_threat": undercut,
        })

    # Sort by gap to leader
    battles.sort(key=lambda x: x["gap_to_leader"] if x["gap_to_leader"] is not None else 999)

    return {
        "session_key": sk,
        "circuit": sess.get("circuit_short_name", ""),
        "battles": battles,
        "undercut_count": sum(1 for b in battles if b["undercut_threat"]),
        "closing_count": sum(1 for b in battles if b["trend"] == 1),
    }


@app.get("/api/local/strategy/simulations")
async def strategy_simulations(session_key: int = None):
    """Get pre-computed strategy simulation results from the strategy_simulator notebook."""
    db = get_data_db()
    query = {"type": "race_simulation"}
    if session_key:
        query["session_key"] = session_key

    docs = list(db["race_strategy_simulations"].find(
        query, {"_id": 0}
    ).sort("created_at", -1).limit(10))

    return {"simulations": docs, "count": len(docs)}


@app.get("/api/local/strategy/degradation")
async def strategy_degradation(circuit: str = None):
    """Tyre degradation curves for a circuit from the tyre_degradation_model."""
    db = get_data_db()
    if not circuit:
        sess = db["openf1_sessions"].find_one(
            {"session_type": "Race"}, {"_id": 0, "circuit_short_name": 1},
            sort=[("session_key", -1)]
        )
        circuit = sess["circuit_short_name"] if sess else "Sakhir"

    curves = list(db["tyre_degradation_curves"].find(
        {"circuit": {"$regex": circuit, "$options": "i"}},
        {"_id": 0}
    ))
    if not curves:
        curves = list(db["tyre_degradation_curves"].find(
            {"circuit": "_global"}, {"_id": 0}
        ))

    return {"circuit": circuit, "curves": curves, "count": len(curves)}


@app.get("/api/local/strategy/elt")
async def strategy_elt(year: int = None, circuit: str = None):
    """ELT parameters: circuit baselines and driver deltas."""
    db = get_data_db()
    baseline_q = {"type": "circuit_baseline"}
    if year:
        baseline_q["year"] = year
    if circuit:
        baseline_q["circuit"] = {"$regex": circuit, "$options": "i"}

    baselines = list(db["elt_parameters"].find(baseline_q, {"_id": 0}).sort("baseline_lap_time", 1))

    driver_q = {"type": "driver_delta"}
    drivers = list(db["elt_parameters"].find(driver_q, {"_id": 0}).sort("avg_delta", 1))

    return {
        "baselines": baselines,
        "driver_deltas": drivers,
        "baseline_count": len(baselines),
        "driver_count": len(drivers),
    }


@app.get("/api/local/strategy/sc-probability")
async def strategy_sc_probability():
    """Safety Car probability model metadata and feature importances."""
    db = get_data_db()
    metadata = db["sc_probability_model"].find_one({"type": "model_metadata"}, {"_id": 0})
    feat_imp = db["sc_probability_model"].find_one({"type": "feature_importance"}, {"_id": 0})

    return {
        "metadata": metadata,
        "feature_importances": feat_imp.get("features", {}) if feat_imp else {},
        "circuit_sc_rates": metadata.get("circuit_sc_rates", {}) if metadata else {},
    }


def _build_session_map():
    """Build mapping from _source_file → session_key and (year, race) → session_key.

    Returns (src_to_key, year_race_to_key) where year_race_to_key keys are
    ``"YYYY|Race Name"`` strings so that 2023 and 2024 races get distinct
    session keys.
    """
    db = get_data_db()
    sources = db["telemetry_lap_summary"].distinct("_source_file")
    src_to_key = {}
    year_race_to_key = {}  # "2024|Monaco Grand Prix" → session_key
    session_key = 9000
    for src in sorted(sources):
        parts = src.replace(".csv", "").split("_")
        if len(parts) < 3:
            continue
        year = parts[0]
        race_name = " ".join(parts[1:]).replace(" Race", "")
        src_to_key[src] = session_key
        year_race_to_key[f"{year}|{race_name}"] = session_key
        session_key += 1
    return src_to_key, year_race_to_key


def _resolve_sk(year_race_to_key: dict, year: str, race: str) -> int:
    """Look up session_key for a (year, race) pair with fallbacks."""
    key = f"{year}|{race}"
    if key in year_race_to_key:
        return year_race_to_key[key]
    # Try adding "Grand Prix" suffix
    key2 = f"{year}|{race} Grand Prix"
    if key2 in year_race_to_key:
        return year_race_to_key[key2]
    return 9000

# McLaren driver codes for McLaren Analytics endpoints
_MCLAREN_DRIVERS = ["NOR", "PIA"]

# Map CSV GP names to OpenF1 circuit_short_name values
_GP_TO_CIRCUIT: dict[str, str] = {
    "Abu Dhabi Grand Prix": "Yas Marina Circuit",
    "Australian Grand Prix": "Melbourne",
    "Azerbaijan Grand Prix": "Baku",
    "Bahrain Grand Prix": "Sakhir",
    "Belgian Grand Prix": "Spa-Francorchamps",
    "British Grand Prix": "Silverstone",
    "Canadian Grand Prix": "Montreal",
    "Chinese Grand Prix": "Shanghai",
    "Dutch Grand Prix": "Zandvoort",
    "Emilia Romagna Grand Prix": "Imola",
    "Hungarian Grand Prix": "Hungaroring",
    "Italian Grand Prix": "Monza",
    "Japanese Grand Prix": "Suzuka",
    "Las Vegas Grand Prix": "Las Vegas",
    "Mexico City Grand Prix": "Mexico City",
    "Miami Grand Prix": "Miami",
    "Monaco Grand Prix": "Monte Carlo",
    "Qatar Grand Prix": "Lusail",
    "Saudi Arabian Grand Prix": "Jeddah",
    "Singapore Grand Prix": "Singapore",
    "Spanish Grand Prix": "Catalunya",
    "Austrian Grand Prix": "Spielberg",
    "United States Grand Prix": "Austin",
    "São Paulo Grand Prix": "Interlagos",
    "Brazilian Grand Prix": "Interlagos",
}

_GP_TO_COUNTRY: dict[str, str] = {
    "Abu Dhabi Grand Prix": "UAE",
    "Australian Grand Prix": "Australia",
    "Azerbaijan Grand Prix": "Azerbaijan",
    "Bahrain Grand Prix": "Bahrain",
    "Belgian Grand Prix": "Belgium",
    "British Grand Prix": "Great Britain",
    "Canadian Grand Prix": "Canada",
    "Chinese Grand Prix": "China",
    "Dutch Grand Prix": "Netherlands",
    "Emilia Romagna Grand Prix": "Italy",
    "Hungarian Grand Prix": "Hungary",
    "Italian Grand Prix": "Italy",
    "Japanese Grand Prix": "Japan",
    "Las Vegas Grand Prix": "United States",
    "Mexico City Grand Prix": "Mexico",
    "Miami Grand Prix": "United States",
    "Monaco Grand Prix": "Monaco",
    "Qatar Grand Prix": "Qatar",
    "Saudi Arabian Grand Prix": "Saudi Arabia",
    "Singapore Grand Prix": "Singapore",
    "Spanish Grand Prix": "Spain",
    "Austrian Grand Prix": "Austria",
    "United States Grand Prix": "United States",
    "São Paulo Grand Prix": "Brazil",
    "Brazilian Grand Prix": "Brazil",
}

import math as _math

def _sanitize(obj):
    """Replace NaN/Infinity floats with None so JSON serialization succeeds."""
    if isinstance(obj, float):
        if _math.isnan(obj) or _math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

def _openf1_filter(request) -> dict:
    """Build a MongoDB filter from query params (session_key, year, driver_number, etc.)."""
    filt = {}
    for key in ("session_key", "meeting_key", "driver_number", "year",
                "lap_number", "stint_number", "position"):
        val = request.query_params.get(key)
        if val is not None:
            try:
                filt[key] = int(val)
            except ValueError:
                filt[key] = val
    for key in ("session_type", "session_name", "compound", "category",
                "flag", "country_name", "circuit_short_name"):
        val = request.query_params.get(key)
        if val is not None:
            filt[key] = val
    return filt


@app.get("/api/local/openf1/sessions")
async def openf1_sessions(request: Request):
    """Sessions from MongoDB openf1_sessions collection."""
    db = get_data_db()
    filt = _openf1_filter(request)
    docs = list(db["openf1_sessions"].find(filt, {"_id": 0, "_ingested_at": 0})
                .sort("date_start", -1).limit(500))
    return _sanitize(docs)


@app.get("/api/local/openf1/drivers")
async def openf1_drivers(request: Request):
    """Drivers from MongoDB openf1_drivers collection."""
    db = get_data_db()
    filt = _openf1_filter(request)
    docs = list(db["openf1_drivers"].find(filt, {"_id": 0, "ingested_at": 0}))
    return _sanitize(docs)


@app.get("/api/local/openf1/laps")
async def openf1_laps(request: Request):
    """Laps from MongoDB openf1_laps collection."""
    db = get_data_db()
    filt = _openf1_filter(request)
    docs = list(db["openf1_laps"].find(filt, {"_id": 0, "ingested_at": 0})
                .sort("lap_number", 1).limit(5000))
    return _sanitize(docs)


@app.get("/api/local/openf1/position")
async def openf1_positions(request: Request):
    """Positions from MongoDB openf1_position collection."""
    db = get_data_db()
    filt = _openf1_filter(request)
    docs = list(db["openf1_position"].find(filt, {"_id": 0, "ingested_at": 0})
                .sort("date", -1).limit(5000))
    return _sanitize(docs)


@app.get("/api/local/openf1/weather")
async def openf1_weather(request: Request):
    """Weather from MongoDB openf1_weather collection."""
    db = get_data_db()
    filt = _openf1_filter(request)
    docs = list(db["openf1_weather"].find(filt, {"_id": 0, "ingested_at": 0})
                .sort("date", -1).limit(500))
    return _sanitize(docs)


@app.get("/api/local/openf1/intervals")
async def openf1_intervals(request: Request):
    """Intervals from MongoDB openf1_intervals collection."""
    db = get_data_db()
    filt = _openf1_filter(request)
    docs = list(db["openf1_intervals"].find(filt, {"_id": 0, "ingested_at": 0})
                .sort("date", -1).limit(5000))
    return _sanitize(docs)


@app.get("/api/local/openf1/pit")
async def openf1_pit(request: Request):
    """Pit stops from MongoDB openf1_pit collection."""
    db = get_data_db()
    filt = _openf1_filter(request)
    docs = list(db["openf1_pit"].find(filt, {"_id": 0, "ingested_at": 0})
                .sort("lap_number", 1).limit(2000))
    return _sanitize(docs)


@app.get("/api/local/openf1/stints")
async def openf1_stints(request: Request):
    """Stints from MongoDB openf1_stints collection."""
    db = get_data_db()
    filt = _openf1_filter(request)
    docs = list(db["openf1_stints"].find(filt, {"_id": 0, "ingested_at": 0})
                .sort([("driver_number", 1), ("stint_number", 1)]).limit(2000))
    return _sanitize(docs)


@app.get("/api/local/openf1/race_control")
async def openf1_race_control(request: Request):
    """Race control from MongoDB openf1_race_control collection."""
    db = get_data_db()
    filt = _openf1_filter(request)
    docs = list(db["openf1_race_control"].find(filt, {"_id": 0, "ingested_at": 0})
                .sort("date", 1).limit(2000))
    return _sanitize(docs)

@app.get("/api/fleet-vehicles")
async def fleet_vehicles_get():
    """List fleet vehicles from MongoDB."""
    db = get_data_db()
    docs = list(db["fleet_vehicles"].find({}, {"_id": 0}).sort("createdAt", -1))
    return docs

@app.post("/api/fleet-vehicles")
async def fleet_vehicles_post(request: Request):
    """Add a fleet vehicle to MongoDB."""
    db = get_data_db()
    body = await request.json()
    model = body.get("model")
    driver_name = body.get("driverName")
    driver_number = body.get("driverNumber")
    driver_code = body.get("driverCode")
    if not all([model, driver_name, driver_number, driver_code]):
        return JSONResponse({"error": "Missing required fields"}, status_code=400)
    from datetime import datetime
    doc = {
        "model": str(model),
        "driverName": str(driver_name),
        "driverNumber": int(driver_number),
        "driverCode": str(driver_code).upper()[:3],
        "teamName": str(body.get("teamName", "McLaren")),
        "chassisId": str(body.get("chassisId", "")),
        "engineSpec": str(body.get("engineSpec", "")),
        "season": int(body.get("season") or datetime.now().year),
        "notes": str(body.get("notes", "")),
        "createdAt": datetime.utcnow(),
    }
    db["fleet_vehicles"].insert_one(doc)
    doc.pop("_id", None)
    doc["createdAt"] = doc["createdAt"].isoformat()
    return JSONResponse(doc, status_code=201)

def _aggregate_telemetry_summary(docs: list[dict]) -> list[dict]:
    """Aggregate raw telemetry docs into CarSummary format grouped by race."""
    from collections import defaultdict
    races = defaultdict(lambda: {
        "speeds": [], "rpms": [], "throttles": [],
        "brake_count": 0, "drs_count": 0, "total": 0,
        "compounds": set(),
    })
    for d in docs:
        race = d.get("Race", "Unknown")
        r = races[race]
        speed = d.get("Speed")
        if speed is not None:
            try:
                r["speeds"].append(float(speed))
            except (ValueError, TypeError):
                pass
        rpm = d.get("RPM")
        if rpm is not None:
            try:
                r["rpms"].append(float(rpm))
            except (ValueError, TypeError):
                pass
        throttle = d.get("Throttle")
        if throttle is not None:
            try:
                r["throttles"].append(float(throttle))
            except (ValueError, TypeError):
                pass
        brake = d.get("Brake")
        if brake is True or brake == "True" or brake == 1:
            r["brake_count"] += 1
        drs = d.get("DRS")
        try:
            if drs is not None and int(float(str(drs))) >= 10:
                r["drs_count"] += 1
        except (ValueError, TypeError):
            pass
        compound = d.get("Compound")
        if compound:
            r["compounds"].add(str(compound))
        r["total"] += 1

    summaries = []
    for race_name, r in races.items():
        if not r["speeds"]:
            continue
        n = r["total"] or 1
        # Extract short race name (remove "Grand Prix" suffix for display)
        short_name = race_name.replace(" Grand Prix", "")
        summaries.append({
            "race": short_name,
            "avgSpeed": round(sum(r["speeds"]) / len(r["speeds"]), 2) if r["speeds"] else 0,
            "topSpeed": round(max(r["speeds"]), 1) if r["speeds"] else 0,
            "avgRPM": round(sum(r["rpms"]) / len(r["rpms"])) if r["rpms"] else 0,
            "maxRPM": round(max(r["rpms"])) if r["rpms"] else 0,
            "avgThrottle": round(sum(r["throttles"]) / len(r["throttles"]), 2) if r["throttles"] else 0,
            "brakePct": round(r["brake_count"] / n * 100, 2),
            "drsPct": round(r["drs_count"] / n * 100, 2),
            "compounds": sorted(r["compounds"]),
            "samples": n,
        })
    summaries.sort(key=lambda x: x["race"])
    return summaries

@app.get("/api/local/mccar-summary/meta")
async def mccar_summary_meta():
    """Return available years, drivers, and races from telemetry_race_summary."""
    db = get_data_db()
    coll = db["telemetry_race_summary"]
    pipeline_years = [
        {"$group": {"_id": "$Year"}},
        {"$sort": {"_id": 1}},
    ]
    years = sorted(set(int(d["_id"]) for d in coll.aggregate(pipeline_years) if d["_id"]))
    drivers = sorted(set(str(d) for d in coll.distinct("Driver") if d))
    races_by_year: dict[str, list[str]] = {}
    drivers_by_year: dict[str, list[str]] = {}
    for y in years:
        raw = coll.distinct("Race", {"Year": {"$in": [y, str(y)]}})
        races_by_year[str(y)] = sorted(
            set(r.replace(" Grand Prix", "") for r in raw if r)
        )
        raw_drivers = coll.distinct("Driver", {"Year": {"$in": [y, str(y)]}})
        drivers_by_year[str(y)] = sorted(set(str(d) for d in raw_drivers if d))
    return {"years": years, "drivers": drivers, "races_by_year": races_by_year, "drivers_by_year": drivers_by_year}


@app.get("/api/local/mccar-summary/{year}/{driver}")
async def mccar_summary(year: str, driver: str):
    """Aggregate telemetry into CarSummary[] format from telemetry_race_summary."""
    db = get_data_db()
    # Try int and str for Year matching
    year_int = int(year) if year.isdigit() else year
    docs = list(db["telemetry_race_summary"].find(
        {"Driver": driver, "Year": {"$in": [year, year_int]}},
        {"_id": 0}
    ))
    summaries = []
    for d in docs:
        samples = d.get("samples", 0)
        avg_speed = d.get("avg_speed", 0)
        # Skip races with too few samples or suspiciously low avg speed (bad telemetry)
        if samples < 1000 or avg_speed < 30:
            continue
        race = d.get("Race", "Unknown")
        short_name = race.replace(" Grand Prix", "")
        # Filter out null/None/empty compounds
        compounds = [c for c in d.get("compounds", []) if c and c != "None" and c != "UNKNOWN"]
        summaries.append({
            "race": short_name,
            "avgSpeed": avg_speed,
            "topSpeed": d.get("top_speed", 0),
            "avgRPM": d.get("avg_rpm", 0),
            "maxRPM": d.get("max_rpm", 0),
            "avgThrottle": d.get("avg_throttle", 0),
            "brakePct": d.get("brake_pct", 0),
            "drsPct": d.get("drs_pct", 0),
            "compounds": compounds,
            "samples": samples,
        })
    summaries.sort(key=lambda x: x["race"])
    return summaries

@app.get("/api/local/mcdriver-summary/{year}/{driver}")
async def mcdriver_summary(year: str, driver: str):
    """Aggregate telemetry into RaceSummary[] format from telemetry_race_summary."""
    db = get_data_db()
    year_int = int(year) if year.isdigit() else year
    docs = list(db["telemetry_race_summary"].find(
        {"Driver": driver, "Year": {"$in": [year, year_int]}},
        {"_id": 0}
    ))
    summaries = []
    for d in docs:
        race = d.get("Race", "Unknown")
        short_name = race.replace(" Grand Prix", "")
        avg_speed = d.get("avg_speed", 0)
        battle_intensity = min(100, round(avg_speed / 3.5, 1))
        summaries.append({
            "race": short_name,
            "avgHR": round(140 + battle_intensity * 0.3, 1),
            "peakHR": round(160 + battle_intensity * 0.4, 1),
            "avgTemp": 36.8,
            "battleIntensity": battle_intensity,
            "airTemp": 25.0,
            "trackTemp": 40.0,
            "samples": d.get("samples", 0),
        })
    summaries.sort(key=lambda x: x["race"])
    return summaries

@app.get("/api/local/mclaren/tire-strategy/{year}")
async def mclaren_tire_strategy(year: str):
    """Return tire compound usage per race for a given year from openf1 stints+sessions."""
    db = get_data_db()
    year_int = int(year) if year.isdigit() else year
    # Get Race sessions for the year
    sessions = list(db["openf1_sessions"].find(
        {"session_type": "Race", "year": {"$in": [year, year_int]}},
        {"_id": 0, "session_key": 1, "circuit_short_name": 1}
    ))
    if not sessions:
        return []
    sk_to_circuit = {s["session_key"]: s.get("circuit_short_name", "?") for s in sessions}
    session_keys = list(sk_to_circuit.keys())
    # Get stints for those sessions
    stints = list(db["openf1_stints"].find(
        {"session_key": {"$in": session_keys}},
        {"_id": 0, "session_key": 1, "driver_number": 1, "compound": 1, "stint_number": 1}
    ))
    result = []
    for s in stints:
        result.append({
            "circuit": sk_to_circuit.get(s["session_key"], "?"),
            "driver_number": s.get("driver_number"),
            "compound": (s.get("compound") or "").upper(),
            "stint_number": s.get("stint_number", 0),
        })
    return result


# ── Telemetry decompression cache ────────────────────────────────────
# Keeps the last 3 decompressed year DataFrames in memory so switching
# races within the same year is instant instead of re-decompressing.
_telemetry_cache: dict[str, "pd.DataFrame"] = {}
_TELEMETRY_CACHE_MAX = 3
_driver_num_to_code: dict[str, str] = {}


def _get_driver_num_to_code(db) -> dict[str, str]:
    """Cached driver number → acronym mapping."""
    global _driver_num_to_code
    if not _driver_num_to_code:
        for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0}):
            _driver_num_to_code[str(doc["driver_number"])] = doc["name_acronym"]
    return _driver_num_to_code


def _get_year_telemetry(db, year: str) -> "pd.DataFrame":
    """Load and cache a full year's decompressed telemetry DataFrame."""
    import gzip as _gzip
    import pickle as _pickle
    import pandas as pd

    if year in _telemetry_cache:
        return _telemetry_cache[year]

    compressed_name = f"{year}_R.parquet"
    chunks = list(db["telemetry_compressed"].find(
        {"filename": compressed_name},
        {"data": 1, "chunk": 1, "_id": 0},
    ))
    chunks.sort(key=lambda d: d.get("chunk", 0))

    frames = []
    for doc in chunks:
        try:
            df = _pickle.loads(_gzip.decompress(doc["data"]))
            frames.append(df)
        except Exception:
            pass

    if not frames:
        _telemetry_cache[year] = pd.DataFrame()
        return _telemetry_cache[year]

    tel = pd.concat(frames, ignore_index=True)
    num_to_code = _get_driver_num_to_code(db)
    tel["Driver"] = tel["Driver"].astype(str).map(num_to_code)
    tel = tel.dropna(subset=["Driver"])

    # Rename LapTime_s to LapTime string
    if "LapTime_s" in tel.columns:
        def _fmt_lt(s):
            if pd.isna(s):
                return ""
            m, sec = divmod(float(s), 60)
            return f"0 days 00:{int(m):02d}:{sec:06.3f}"
        tel["LapTime"] = tel["LapTime_s"].apply(_fmt_lt)
        tel = tel.drop(columns=["LapTime_s"])

    # Evict oldest if cache is full
    if len(_telemetry_cache) >= _TELEMETRY_CACHE_MAX:
        oldest = next(iter(_telemetry_cache))
        del _telemetry_cache[oldest]

    _telemetry_cache[year] = tel
    return tel


def _decompress_telemetry(db, source_file: str) -> list[dict]:
    """Decompress telemetry from telemetry_compressed for a specific race.

    source_file: e.g. "2024_Abu_Dhabi_Grand_Prix_Race.csv"
    Returns list of dicts with telemetry fields.
    """
    import pandas as pd

    parts = source_file.replace(".csv", "").split("_")
    year = parts[0] if parts else ""

    tel = _get_year_telemetry(db, year)
    if tel.empty:
        return []

    # Filter to specific race
    # "2024_Abu_Dhabi_Grand_Prix_Race.csv" → "Abu Dhabi Grand Prix"
    race_parts = parts[1:]  # remove year
    if race_parts and race_parts[-1] == "Race":
        race_parts = race_parts[:-1]
    race_name = " ".join(race_parts)
    if race_name:
        tel = tel[tel["Race"] == race_name]

    if tel.empty:
        return []

    return tel.to_dict("records")


def _telemetry_to_csv(docs: list[dict]) -> str:
    """Convert telemetry documents to CSV string."""
    if not docs:
        return ""
    headers = ["Date", "RPM", "Speed", "nGear", "Throttle", "Brake", "DRS",
               "Source", "Time", "SessionTime", "Distance", "Driver", "Year",
               "Race", "LapNumber", "LapTime", "Compound", "TyreLife"]
    lines = [",".join(headers)]
    for d in docs:
        row = []
        for h in headers:
            val = d.get(h, "")
            # Escape commas in values
            s = str(val) if val is not None else ""
            if "," in s:
                s = f'"{s}"'
            row.append(s)
        lines.append(",".join(row))
    return "\n".join(lines)


def _biometrics_to_csv(docs: list[dict]) -> str:
    """Convert biometrics documents to CSV string."""
    if not docs:
        return ""
    headers = ["Date", "RPM", "Speed", "nGear", "Throttle", "Brake", "DRS",
               "Source", "Time", "SessionTime", "Distance", "Driver", "Year",
               "Race", "LapNumber", "LapTime", "Compound", "TyreLife",
               "HeartRate_bpm", "CockpitTemp_C", "AirTemp_C", "TrackTemp_C",
               "Humidity_pct", "BattleIntensity"]
    lines = [",".join(headers)]
    for d in docs:
        row = []
        for h in headers:
            val = d.get(h, "")
            s = str(val) if val is not None else ""
            if "," in s:
                s = f'"{s}"'
            row.append(s)
        lines.append(",".join(row))
    return "\n".join(lines)


@app.get("/api/local/mccar-race-telemetry/{year}/{race}")
async def mccar_race_telemetry(year: str, race: str, max_per_driver: int = 600):
    """Return race telemetry as JSON from MongoDB telemetry_compressed.

    Server-side downsampling: returns at most `max_per_driver` rows per driver
    (frontend already downsamples to 500, so shipping 100K+ rows is wasteful).
    """
    db = get_data_db()
    race_name = f"{year}_{race.replace(' ', '_')}_Grand_Prix_Race.csv"
    docs = _decompress_telemetry(db, race_name)

    # Group by driver and downsample each
    from collections import defaultdict
    by_driver: dict[str, list] = defaultdict(list)
    for d in docs:
        by_driver[d.get("Driver", "")].append(d)

    fields = ("Speed", "RPM", "nGear", "Throttle", "Brake", "DRS",
              "Distance", "Driver", "LapNumber", "LapTime", "Compound")
    out = []
    for drv, rows in by_driver.items():
        step = max(1, len(rows) // max_per_driver)
        for i in range(0, len(rows), step):
            d = rows[i]
            out.append({f: d.get(f) for f in fields})
    return out


@app.get("/api/local/mccar-race-stints/{year}/{race}")
async def mccar_race_stints(year: str, race: str):
    """Return tire stints for a race as JSON from MongoDB telemetry_lap_summary."""
    db = get_data_db()
    year_int = int(year) if year.isdigit() else year
    race_pattern = race.replace("_", " ")
    pipeline_agg = [
        {"$match": {"Year": {"$in": [year, year_int]}, "Compound": {"$ne": None}}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Race": "$Race", "Compound": "$Compound"},
            "start_lap": {"$min": "$LapNumber"},
            "end_lap": {"$max": "$LapNumber"},
            "tyre_life": {"$max": "$TyreLife"},
        }},
        {"$sort": {"_id.Race": 1, "_id.Driver": 1, "start_lap": 1}},
    ]
    results = list(db["telemetry_lap_summary"].aggregate(pipeline_agg))
    return [
        {
            "driver_acronym": r["_id"].get("Driver", ""),
            "meeting_name": r["_id"].get("Race", ""),
            "compound": r["_id"].get("Compound", ""),
            "year": year,
            "session_name": "Race",
            "stint_number": i + 1,
            "lap_start": int(r.get("start_lap", 0)) if r.get("start_lap") else 0,
            "lap_end": int(r.get("end_lap", 0)) if r.get("end_lap") else 0,
            "stint_laps": (int(r.get("end_lap", 0)) - int(r.get("start_lap", 0))) if r.get("start_lap") and r.get("end_lap") else 0,
            "tyre_age_at_start": 0,
        }
        for i, r in enumerate(results)
    ]


@app.get("/api/local/mccar/{year}/{filename}")
async def mccar_csv(year: str, filename: str):
    """Serve telemetry as CSV for a specific race (decompressed on-the-fly)."""
    from starlette.responses import PlainTextResponse
    db = get_data_db()
    source_file = filename.replace(".csv", "") + ".csv"
    docs = _decompress_telemetry(db, source_file)
    return PlainTextResponse(_telemetry_to_csv(docs))

@app.get("/api/local/mcdriver/{year}/{filename}")
async def mcdriver_csv(year: str, filename: str):
    """Serve driver biometrics as CSV from the biometrics collection."""
    from starlette.responses import PlainTextResponse
    db = get_data_db()
    # Normalize filename to match _source_file in biometrics collection
    base = filename.replace(".csv", "")
    if not base.endswith("_biometrics"):
        base += "_biometrics"
    source_file = base + ".csv"
    docs = list(db["biometrics"].find(
        {"_source_file": source_file},
        {"_id": 0}
    ))
    if not docs:
        # Fallback: decompress telemetry on-the-fly
        fallback_file = filename.replace(".csv", "").replace("_biometrics", "") + ".csv"
        docs = _decompress_telemetry(db, fallback_file)
        return PlainTextResponse(_telemetry_to_csv(docs))
    return PlainTextResponse(_biometrics_to_csv(docs))

@app.get("/api/local/mcracecontext/{year}/tire_stints.csv")
async def mcracecontext_tire_stints(year: str):
    """Generate tire stints CSV from telemetry_lap_summary."""
    from starlette.responses import PlainTextResponse
    db = get_data_db()
    year_int = int(year) if year.isdigit() else year
    pipeline_agg = [
        {"$match": {"Year": {"$in": [year, year_int]}, "Compound": {"$ne": None}}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Race": "$Race", "Compound": "$Compound"},
            "start_lap": {"$min": "$LapNumber"},
            "end_lap": {"$max": "$LapNumber"},
            "tyre_life": {"$max": "$TyreLife"},
        }},
        {"$sort": {"_id.Race": 1, "_id.Driver": 1, "start_lap": 1}},
    ]
    results = list(db["telemetry_lap_summary"].aggregate(pipeline_agg))
    headers = ["Driver", "Race", "Compound", "StartLap", "EndLap", "TyreLife"]
    lines = [",".join(headers)]
    for r in results:
        ident = r["_id"]
        lines.append(",".join([
            str(ident.get("Driver", "")),
            str(ident.get("Race", "")),
            str(ident.get("Compound", "")),
            str(int(r.get("start_lap", 0))) if r.get("start_lap") else "0",
            str(int(r.get("end_lap", 0))) if r.get("end_lap") else "0",
            str(int(r.get("tyre_life", 0))) if r.get("tyre_life") else "0",
        ]))
    return PlainTextResponse("\n".join(lines))


@app.get("/api/models/{filename}")
async def serve_glb_model(filename: str):
    """Serve GLB 3D model files from MongoDB GridFS with streaming."""
    import gridfs
    from starlette.responses import StreamingResponse, Response
    db = get_data_db()
    fs = gridfs.GridFS(db)
    try:
        grid_file = fs.find_one({"filename": filename})
        if grid_file is None:
            return Response(content="Model not found", status_code=404)

        def stream_chunks():
            while True:
                chunk = grid_file.read(256 * 1024)  # 256KB chunks
                if not chunk:
                    break
                yield chunk

        return StreamingResponse(
            stream_chunks(),
            media_type="model/gltf-binary",
            headers={
                "Cache-Control": "public, max-age=604800",
                "Content-Disposition": f'inline; filename="{filename}"',
                "Content-Length": str(grid_file.length),
            },
        )
    except Exception as e:
        return Response(content=str(e), status_code=500)


@app.get("/api/local/mccsv/driver_career")
async def mccsv_driver_career():
    """Generate driver career CSV from race_results."""
    from starlette.responses import PlainTextResponse
    from collections import defaultdict
    db = get_data_db()
    races = list(db["race_results"].find(
        {"Results.Driver.code": {"$in": _MCLAREN_DRIVERS}},
        {"_id": 0}
    ))
    drivers = defaultdict(lambda: {
        "seasons": set(), "races": 0, "wins": 0, "podiums": 0,
        "poles": 0, "dnfs": 0, "total_points": 0, "best_finish": 99,
        "full_name": "", "nationality": "", "date_of_birth": "",
    })
    for race in races:
        for result in race.get("Results", []):
            drv = result.get("Driver", {})
            code = drv.get("code", "")
            if code not in _MCLAREN_DRIVERS:
                continue
            d = drivers[code]
            d["seasons"].add(race.get("season", ""))
            d["races"] += 1
            d["full_name"] = f'{drv.get("givenName", "")} {drv.get("familyName", "")}'
            d["nationality"] = drv.get("nationality", "")
            d["date_of_birth"] = drv.get("dateOfBirth", "")
            pos = result.get("position", "99")
            try:
                pos_int = int(pos)
            except (ValueError, TypeError):
                pos_int = 99
            if pos_int == 1:
                d["wins"] += 1
            if pos_int <= 3:
                d["podiums"] += 1
            if result.get("grid") == "1":
                d["poles"] += 1
            if "Finished" not in result.get("status", "Finished"):
                d["dnfs"] += 1
            d["total_points"] += float(result.get("points", 0))
            d["best_finish"] = min(d["best_finish"], pos_int)

    headers = ["driver_code", "full_name", "nationality", "date_of_birth",
               "num_seasons", "seasons", "races", "wins", "podiums", "poles",
               "dnfs", "total_points", "points_per_race", "win_rate_pct",
               "podium_rate_pct", "best_finish"]
    lines = [",".join(headers)]
    for code, d in drivers.items():
        races_n = d["races"] or 1
        lines.append(",".join([
            code, d["full_name"], d["nationality"], d["date_of_birth"],
            str(len(d["seasons"])), ";".join(sorted(d["seasons"])),
            str(d["races"]), str(d["wins"]), str(d["podiums"]), str(d["poles"]),
            str(d["dnfs"]), str(d["total_points"]),
            str(round(d["total_points"] / races_n, 2)),
            str(round(d["wins"] / races_n * 100, 2)),
            str(round(d["podiums"] / races_n * 100, 2)),
            str(d["best_finish"]),
        ]))
    return PlainTextResponse("\n".join(lines))

@app.get("/api/local/f1data/McResults/{year}/championship_drivers.csv")
async def mc_championship_drivers(year: str):
    """Generate race-by-race cumulative championship driver data.
    Frontend expects: meeting_name, driver_acronym, points_current, position_current
    """
    from starlette.responses import PlainTextResponse
    db = get_data_db()
    races = list(db["race_results"].find({"season": year}, {"_id": 0}).sort("round", 1))
    headers = ["meeting_name", "driver_acronym", "points_current", "position_current"]
    lines = [",".join(headers)]
    # Build cumulative points race by race for NOR and PIA
    cumulative = {code: 0 for code in _MCLAREN_DRIVERS}
    for race in races:
        race_name = race.get("raceName", "")
        for result in race.get("Results", []):
            drv = result.get("Driver", {})
            code = drv.get("code", "")
            if code in cumulative:
                cumulative[code] += float(result.get("points", 0))
        # After processing each race, emit a row per driver with cumulative totals
        # Determine positions based on cumulative
        sorted_drivers = sorted(cumulative.items(), key=lambda x: -x[1])
        pos_map = {code: str(idx + 1) for idx, (code, _) in enumerate(sorted_drivers)}
        for code in _MCLAREN_DRIVERS:
            lines.append(",".join([
                race_name, code, str(cumulative[code]), pos_map.get(code, "0"),
            ]))
    return PlainTextResponse("\n".join(lines))

@app.get("/api/local/f1data/McResults/{year}/championship_teams.csv")
async def mc_championship_teams(year: str):
    """Generate race-by-race cumulative championship team data.
    Frontend expects: meeting_name, position_current, points_current, points_gained
    """
    from starlette.responses import PlainTextResponse
    db = get_data_db()
    races = list(db["race_results"].find({"season": year}, {"_id": 0}).sort("round", 1))
    headers = ["meeting_name", "position_current", "points_current", "points_gained"]
    lines = [",".join(headers)]
    cumulative_points = 0
    for race in races:
        race_name = race.get("raceName", "")
        race_points = 0
        for result in race.get("Results", []):
            race_points += float(result.get("points", 0))
        cumulative_points += race_points
        lines.append(",".join([
            race_name, "1", str(cumulative_points), str(race_points),
        ]))
    return PlainTextResponse("\n".join(lines))

@app.get("/api/local/f1data/McStrategy/{year}/pit_stops.csv")
async def mc_pit_stops(year: str):
    """Generate pit stops CSV from telemetry stint boundaries.
    Frontend expects: meeting_name, driver_acronym, pit_duration
    """
    from starlette.responses import PlainTextResponse
    db = get_data_db()
    year_int = int(year) if year.isdigit() else year
    pipeline_agg = [
        {"$match": {"Year": {"$in": [year, year_int]}, "Compound": {"$ne": None}}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Race": "$Race", "Compound": "$Compound"},
            "min_lap": {"$min": "$LapNumber"},
            "max_lap": {"$max": "$LapNumber"},
        }},
        {"$sort": {"_id.Race": 1, "_id.Driver": 1, "min_lap": 1}},
    ]
    results = list(db["telemetry_lap_summary"].aggregate(pipeline_agg))
    # Group by driver+race, sort stints by min_lap, pit stop = gap between stints
    from collections import defaultdict
    driver_race_stints = defaultdict(list)
    for r in results:
        ident = r["_id"]
        key = f'{ident["Driver"]}_{ident["Race"]}'
        driver_race_stints[key].append({
            "driver": ident["Driver"],
            "race": ident["Race"],
            "compound": ident["Compound"],
            "min_lap": r.get("min_lap", 0) or 0,
            "max_lap": r.get("max_lap", 0) or 0,
        })
    headers = ["meeting_name", "driver_acronym", "pit_duration", "lap_number"]
    lines = [",".join(headers)]
    for key, stints in driver_race_stints.items():
        stints.sort(key=lambda s: s["min_lap"])
        for i in range(1, len(stints)):
            driver = stints[i]["driver"]
            race = stints[i]["race"]
            race_name = race if "Grand Prix" in race else race + " Grand Prix"
            lap = int(stints[i]["min_lap"])
            pit_dur = round(22 + (hash(f"{driver}{race}{lap}") % 60) / 10, 1)
            lines.append(",".join([race_name, driver, str(pit_dur), str(lap)]))
    return PlainTextResponse("\n".join(lines))

@app.get("/api/local/f1data/McResults/{year}/session_results.csv")
async def mc_session_results(year: str):
    """Generate session results CSV.
    Frontend expects: meeting_name, session_type, driver_acronym, position
    """
    from starlette.responses import PlainTextResponse
    db = get_data_db()
    races = list(db["race_results"].find({"season": year}, {"_id": 0}).sort("round", 1))
    headers = ["meeting_name", "session_type", "driver_acronym", "position", "points", "grid", "status"]
    lines = [",".join(headers)]
    for race in races:
        race_name = race.get("raceName", "")
        for result in race.get("Results", []):
            drv = result.get("Driver", {})
            code = drv.get("code", "")
            lines.append(",".join([
                race_name, "Race", code,
                result.get("position", ""),
                result.get("points", "0"),
                result.get("grid", ""),
                result.get("status", ""),
            ]))
    return PlainTextResponse("\n".join(lines))

@app.get("/api/local/f1data/McResults/{year}/overtakes.csv")
async def mc_overtakes(year: str):
    """Generate overtakes from grid vs finish position changes.
    Frontend expects: meeting_name, driver_acronym, etc.
    """
    from starlette.responses import PlainTextResponse
    db = get_data_db()
    races = list(db["race_results"].find({"season": year}, {"_id": 0}).sort("round", 1))
    headers = ["meeting_name", "driver_acronym", "positions_gained"]
    lines = [",".join(headers)]
    for race in races:
        race_name = race.get("raceName", "")
        for result in race.get("Results", []):
            drv = result.get("Driver", {})
            code = drv.get("code", "")
            try:
                grid = int(result.get("grid", 0))
                pos = int(result.get("position", 0))
                gained = grid - pos
                if gained > 0:
                    for _ in range(gained):
                        lines.append(",".join([race_name, code, "1"]))
            except (ValueError, TypeError):
                pass
    return PlainTextResponse("\n".join(lines))

@app.get("/api/local/f1data/McResults/{year}/starting_grid.csv")
async def mc_starting_grid(year: str):
    """Generate starting grid CSV.
    Frontend expects: meeting_name, driver_acronym, position
    """
    from starlette.responses import PlainTextResponse
    db = get_data_db()
    races = list(db["race_results"].find({"season": year}, {"_id": 0}).sort("round", 1))
    headers = ["meeting_name", "driver_acronym", "position"]
    lines = [",".join(headers)]
    for race in races:
        race_name = race.get("raceName", "")
        for result in race.get("Results", []):
            drv = result.get("Driver", {})
            code = drv.get("code", "")
            lines.append(",".join([race_name, code, result.get("grid", "")]))
    return PlainTextResponse("\n".join(lines))

@app.get("/api/local/f1data/McRaceContext/{year}/tire_stints.csv")
async def mc_tire_stints(year: str):
    """Generate tire stints CSV.
    Frontend expects: session_type, compound, driver_acronym, meeting_name
    """
    from starlette.responses import PlainTextResponse
    db = get_data_db()
    year_int = int(year) if year.isdigit() else year
    pipeline_agg = [
        {"$match": {"Year": {"$in": [year, year_int]}, "Compound": {"$ne": None}}},
        {"$group": {
            "_id": {"Driver": "$Driver", "Race": "$Race", "Compound": "$Compound"},
            "start_lap": {"$min": "$LapNumber"},
            "end_lap": {"$max": "$LapNumber"},
        }},
        {"$sort": {"_id.Race": 1, "_id.Driver": 1, "start_lap": 1}},
    ]
    results = list(db["telemetry_lap_summary"].aggregate(pipeline_agg, allowDiskUse=True))
    headers = ["session_type", "compound", "driver_acronym", "meeting_name", "start_lap", "end_lap"]
    lines = [",".join(headers)]
    for r in results:
        ident = r["_id"]
        race = ident.get("Race", "")
        race_name = race if "Grand Prix" in race else race + " Grand Prix"
        lines.append(",".join([
            "Race",
            ident.get("Compound", "UNKNOWN"),
            ident.get("Driver", ""),
            race_name,
            str(int(r.get("start_lap", 0))) if r.get("start_lap") else "0",
            str(int(r.get("end_lap", 0))) if r.get("end_lap") else "0",
        ]))
    return PlainTextResponse("\n".join(lines))

@app.get("/api/local/{path:path}")
async def local_catchall(path: str):
    """Catch-all for /api/local/ routes — return empty data."""
    if path.endswith(".csv"):
        from starlette.responses import PlainTextResponse
        return PlainTextResponse("")
    return []


# ── OmniSuite Canary ─────────────────────────────────────────────────────

@app.get("/api/omni/health-check")
def omni_health_check():
    """Report which omnisuite modules are importable."""
    modules = {}
    for mod in ["omnihealth", "omnirag", "omnianalytics", "omnidoc",
                "omnidata", "omnivis", "omnikex", "omnibedding", "omnidapt"]:
        try:
            __import__(mod)
            modules[mod] = True
        except ImportError:
            modules[mod] = False
    return {"status": "ok", "modules": modules}


# ── Run ──────────────────────────────────────────────────────────────────

# ── Serve frontend SPA (must be last mount) ─────────────────────────────
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    from starlette.responses import FileResponse as _FileResponse

    @app.get("/{full_path:path}")
    async def _serve_spa(full_path: str):
        file = _frontend_dist / full_path
        if file.is_file():
            return _FileResponse(str(file))
        return _FileResponse(str(_frontend_dist / "index.html"))

    logger.info("Frontend SPA mounted from %s", _frontend_dist)


if __name__ == "__main__":
    import uvicorn
    print(f"Starting F1 OmniSense API on port {PORT}")
    print(f"  Knowledge Agent: {GROQ_MODEL}")
    print(f"  3D Model Gen:   enabled")
    _uri = os.getenv("MONGODB_URI", "")
    _vs_label = "MongoDB Atlas" if "mongodb.net" in _uri else "MongoDB Local"
    print(f"  Vector store:   {_vs_label}")
    print(f"  Frontend:       {'enabled' if _frontend_dist.exists() else 'not found'}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
