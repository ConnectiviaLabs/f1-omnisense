"""OmniRAG APIRouter — enhanced RAG chat with conversation memory and cliff detection.

Endpoints:
    POST /api/omni/rag/chat                  — chat with server-side session memory
    POST /api/omni/rag/search                — standalone vector search (no LLM)
    GET  /api/omni/rag/sessions              — list active sessions
    DELETE /api/omni/rag/sessions/{session_id} — clear a session
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/omni/rag", tags=["OmniRAG"])

# ── F1 System Prompt (same as chat_server.py) ──────────────────────────

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


# ── Lazy singletons ─────────────────────────────────────────────────────

_store = None
_retriever = None
_chain = None
_convos = None


def _get_convos():
    global _convos
    if _convos is None:
        from omnirag import ConversationManager
        _convos = ConversationManager(max_messages=20, session_ttl=3600)
    return _convos


def _get_chain():
    global _store, _retriever, _chain
    if _chain is not None:
        return _chain

    from omnirag import AtlasStore, RAGRetriever, RAGChain
    from omnidoc.embedder import get_embedder

    _store = AtlasStore(
        uri=os.getenv("MONGODB_URI"),
        db_name=os.getenv("MONGODB_DB", "marip_f1"),
        collection_name="f1_knowledge",
        index_name="vector_index",
        embedding_dim=1024,  # BGE-large-en-v1.5
    )
    embedder = get_embedder(enable_clip=False)
    _retriever = RAGRetriever(
        vectorstore=_store,
        embed_fn=embedder.embed_query,
    )
    _chain = RAGChain(
        retriever=_retriever,
        llm_provider="auto",  # waterfall: ollama → groq → openai → anthropic
        system_prompt=SYSTEM_PROMPT,
    )
    return _chain


def _get_retriever():
    _get_chain()
    return _retriever


# ── Request/Response models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    provider: Optional[str] = None
    k: int = 8


class SearchRequest(BaseModel):
    query: str
    k: int = 8
    category: Optional[str] = None


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/chat")
def chat(req: ChatRequest):
    """Chat with OmniRAG — server-side session memory + cliff detection."""
    chain = _get_chain()
    convos = _get_convos()

    sid = req.session_id or convos.create_session()
    history = convos.get_history(sid)

    response = chain.ask_with_history(req.message, history, k=req.k)

    convos.append(sid, "user", req.message)
    convos.append(sid, "assistant", response.answer)

    return {
        "answer": response.answer,
        "sources": response.sources,
        "model_used": response.model_used,
        "session_id": sid,
    }


@router.post("/search")
def search(req: SearchRequest):
    """Standalone vector search with cliff detection — no LLM call."""
    retriever = _get_retriever()
    results = retriever.search_enhanced(
        req.query, k=req.k, category=req.category,
    )
    return {
        "query": req.query,
        "count": len(results),
        "results": [
            {
                "content": r.document.content,
                "score": round(r.score, 4),
                "rank": r.rank,
                "metadata": r.document.metadata,
            }
            for r in results
        ],
    }


@router.get("/sessions")
def list_sessions():
    """List active conversation sessions."""
    convos = _get_convos()
    sessions = convos.list_sessions()
    return {"count": len(sessions), "sessions": sessions}


@router.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    """Clear a conversation session."""
    convos = _get_convos()
    convos.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
