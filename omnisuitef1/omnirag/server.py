"""FastAPI server exposing all OmniRAG capabilities.

Usage:
    uvicorn omnirag.server:app --host 0.0.0.0 --port 8200
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OmniRAG",
    description="Unified RAG service — semantic search, Q&A with citations, document ingestion.",
    version="0.1.0",
)


# ── Pydantic Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    k: int = 5
    category: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    category: Optional[str] = None
    min_score: float = 0.3


class TextIngestRequest(BaseModel):
    texts: List[str]
    source: str = "api"
    category: str = "text"


class AgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    k: int = 5
    category: Optional[str] = None


# ── Lazy Singletons ──────────────────────────────────────────────────

_vectorstore = None
_retriever = None
_chain = None
_conversations = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from omnirag.vectorstore import get_vectorstore
        _vectorstore = get_vectorstore()
    return _vectorstore


def _get_retriever():
    global _retriever
    if _retriever is None:
        from omnidoc.embedder import get_embedder
        from omnirag.retriever import RAGRetriever
        embedder = get_embedder()
        _retriever = RAGRetriever(
            vectorstore=_get_vectorstore(),
            embed_fn=embedder.embed_query,
        )
    return _retriever


def _get_chain():
    global _chain
    if _chain is None:
        from omnirag.qa_chain import RAGChain
        _chain = RAGChain(retriever=_get_retriever())
    return _chain


def _get_conversations():
    global _conversations
    if _conversations is None:
        from omnirag.conversation import ConversationManager
        _conversations = ConversationManager()
    return _conversations


_context_pipeline = None
_tool_registry = None
_agent = None


def _get_context_pipeline():
    global _context_pipeline
    if _context_pipeline is None:
        from omnirag.context_pipeline import ContextPipeline, retriever_as_source
        _context_pipeline = ContextPipeline()
        _context_pipeline.register(
            "vector_search",
            retriever_as_source(_get_retriever()),
            priority=0,
        )
    return _context_pipeline


def _get_tool_registry():
    global _tool_registry
    if _tool_registry is None:
        from omnirag.agent import ToolRegistry
        _tool_registry = ToolRegistry()
    return _tool_registry


def _get_agent():
    global _agent
    if _agent is None:
        from omnirag.agent import RAGAgent
        _agent = RAGAgent(
            chain=_get_chain(),
            tool_registry=_get_tool_registry(),
            conversation_manager=_get_conversations(),
        )
    return _agent


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    info = {"status": "ok", "timestamp": time.time()}
    try:
        vs = _get_vectorstore()
        info["document_count"] = vs.count()
    except Exception:
        info["document_count"] = "unavailable"
    return info


# ── Chat (RAG Q&A) ──────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    """RAG chat with conversation history and source citations."""
    chain = _get_chain()
    convos = _get_conversations()

    # Get or create session
    session_id = req.session_id
    if not session_id:
        session_id = convos.create_session()

    # Get history
    history = convos.get_history(session_id)

    # Ask with history
    if history:
        response = chain.ask_with_history(
            req.message, history, k=req.k, category=req.category,
        )
    else:
        response = chain.ask(
            req.message, k=req.k, category=req.category,
        )

    # Store conversation
    convos.append(session_id, "user", req.message)
    convos.append(session_id, "assistant", response.answer)

    result = response.to_dict()
    result["session_id"] = session_id
    return result


# ── Semantic Search ──────────────────────────────────────────────────

@app.post("/search")
async def search(req: SearchRequest):
    """Semantic search without LLM generation."""
    retriever = _get_retriever()
    results = retriever.search_enhanced(
        req.query, k=req.k, min_score=req.min_score, category=req.category,
    )
    return {"results": [r.to_dict() for r in results]}


# ── Document Ingestion ───────────────────────────────────────────────

@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
):
    """Upload and ingest a document into the vector store."""
    import tempfile
    from pathlib import Path
    from omnirag.ingest import ingest_file

    # Save upload to temp file
    suffix = Path(file.filename or "upload.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = ingest_file(
            tmp_path, _get_vectorstore(),
            category=category,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # Override filename from upload
        result.file_name = file.filename or result.file_name
        return result.to_dict()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ingest/text")
async def ingest_text_endpoint(req: TextIngestRequest):
    """Ingest raw text strings into the vector store."""
    from omnirag.ingest import ingest_texts

    result = ingest_texts(
        req.texts, _get_vectorstore(),
        source=req.source, category=req.category,
    )
    return result.to_dict()


# ── Visual Search (CLIP) ────────────────────────────────────────────

@app.get("/visual-search")
async def visual_search(
    q: str = Query(..., description="Text query for image search"),
    k: int = Query(8, description="Number of results"),
    index_path: Optional[str] = Query(None, description="Path to CLIP index JSON"),
):
    """Cross-modal search: text query → ranked images via CLIP embeddings."""
    import json
    import numpy as np

    if not index_path:
        return JSONResponse(
            status_code=400,
            content={"error": "index_path required (path to clip_index.json)"},
        )

    try:
        from omnidoc.embedder import get_embedder
        embedder = get_embedder()

        with open(index_path) as f:
            index = json.load(f)

        # Embed query in CLIP space
        query_vec = np.array(embedder.clip.embed_text(q))
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Cosine similarity against all image embeddings
        image_vecs = np.array([img["embedding"] for img in index["images"]])
        image_norms = image_vecs / np.linalg.norm(image_vecs, axis=1, keepdims=True)
        similarities = query_vec @ image_norms.T

        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            img = index["images"][int(idx)]
            results.append({
                "path": img.get("path", ""),
                "score": round(float(similarities[idx]), 4),
                "auto_tags": img.get("auto_tags", []),
            })

        return {"query": q, "results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Categories ───────────────────────────────────────────────────────

@app.get("/categories")
async def list_categories():
    """List all document categories in the vector store."""
    try:
        vs = _get_vectorstore()
        categories = vs.list_categories()
        return {"categories": categories}
    except Exception as e:
        return {"categories": [], "error": str(e)}


# ── Agent (Tool-Calling) ────────────────────────────────────────────

@app.post("/agent/chat")
async def agent_chat(req: AgentChatRequest):
    """Agent chat with tool calling and RAG fallback."""
    agent = _get_agent()
    response = agent.process_message(
        req.message,
        session_id=req.session_id,
        k=req.k,
        category=req.category,
    )
    return response.to_dict()


@app.post("/agent/chat/stream")
async def agent_chat_stream(req: AgentChatRequest):
    """SSE streaming agent chat with tool events."""
    from omnirag.streaming import stream_agent_response
    agent = _get_agent()
    return StreamingResponse(
        stream_agent_response(
            agent, req.message,
            session_id=req.session_id,
            k=req.k,
            category=req.category,
        ),
        media_type="text/event-stream",
    )


@app.get("/agent/tools")
async def agent_tools():
    """List registered agent tools."""
    registry = _get_tool_registry()
    return {"tools": registry.list_tools()}


# ── Context Pipeline ────────────────────────────────────────────────

@app.get("/context/sources")
async def context_sources():
    """List registered context pipeline sources."""
    pipeline = _get_context_pipeline()
    return {"sources": pipeline.list_sources()}
