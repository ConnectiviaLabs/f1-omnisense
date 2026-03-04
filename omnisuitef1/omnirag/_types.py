"""Shared type definitions for the omnirag service."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RAGDocument:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    content_hash: str = ""

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        d = {
            "content": self.content,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
        }
        if include_embedding and self.embedding:
            d["embedding"] = self.embedding
        return d


@dataclass
class SearchResult:
    document: RAGDocument
    score: float
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.document.content,
            "metadata": self.document.metadata,
            "score": round(self.score, 4),
            "rank": self.rank,
        }


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    model_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "model_used": self.model_used,
        }


@dataclass
class IngestResult:
    file_name: str
    chunk_count: int
    content_hash: str = ""
    status: str = "success"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "file_name": self.file_name,
            "chunk_count": self.chunk_count,
            "content_hash": self.content_hash,
            "status": self.status,
        }
        if self.error:
            d["error"] = self.error
        return d
