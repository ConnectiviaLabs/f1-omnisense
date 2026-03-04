"""Vector store backends: MongoDB Atlas (primary) + ChromaDB (local fallback).

Usage:
    store = get_vectorstore()  # auto-detect: Atlas if MONGO_URI set, else ChromaDB
    store.upsert(documents, embeddings)
    results = store.similarity_search(query_embedding, k=5)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from omnirag._types import RAGDocument

logger = logging.getLogger(__name__)


# ── Protocol ─────────────────────────────────────────────────────────

@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Backend-agnostic vector store interface."""

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple]:
        """Return list of (RAGDocument, score) tuples."""
        ...

    def upsert(
        self,
        documents: List[RAGDocument],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Insert or update documents with embeddings. Returns count."""
        ...

    def delete(self, filter: Dict[str, Any]) -> int:
        """Delete documents matching filter. Returns count deleted."""
        ...

    def count(self) -> int:
        """Total document count."""
        ...


# ── MongoDB Atlas ────────────────────────────────────────────────────

class AtlasStore:
    """MongoDB Atlas with $vectorSearch.

    Requires Atlas M10+ cluster with vector search index.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: str = "rag_knowledge",
        index_name: str = "vector_index",
        embedding_dim: int = 1024,
    ):
        from pymongo import MongoClient

        self._uri = uri or os.getenv("MONGO_URI", "")
        self._db_name = db_name or os.getenv("MONGO_DB", "omnirag")
        self._collection_name = collection_name
        self._index_name = index_name
        self._embedding_dim = embedding_dim

        self._client = MongoClient(self._uri)
        self._db = self._client[self._db_name]
        self._collection = self._db[self._collection_name]

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple]:
        """Semantic search via $vectorSearch aggregation."""
        vector_search = {
            "$vectorSearch": {
                "index": self._index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": k * 10,
                "limit": k,
            }
        }
        if filter:
            vector_search["$vectorSearch"]["filter"] = {
                f"metadata.{key}": val for key, val in filter.items()
            }

        pipeline = [
            vector_search,
            {
                "$project": {
                    "page_content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                    "_id": 0,
                }
            },
        ]

        results = []
        for doc in self._collection.aggregate(pipeline):
            rag_doc = RAGDocument(
                content=doc.get("page_content", ""),
                metadata=doc.get("metadata", {}),
            )
            results.append((rag_doc, float(doc.get("score", 0.0))))

        return results

    def upsert(
        self,
        documents: List[RAGDocument],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Bulk insert documents with embeddings."""
        total = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]

            records = []
            for doc, emb in zip(batch_docs, batch_embs):
                records.append({
                    "page_content": doc.content,
                    "metadata": doc.metadata,
                    "embedding": emb,
                    "content_hash": doc.content_hash,
                })

            if records:
                self._collection.insert_many(records)
                total += len(records)

        return total

    def delete(self, filter: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter."""
        mongo_filter = {f"metadata.{k}": v for k, v in filter.items()}
        result = self._collection.delete_many(mongo_filter)
        return result.deleted_count

    def count(self) -> int:
        return self._collection.count_documents({})

    def ensure_vector_index(self):
        """Create Atlas vector search index. Requires M10+ cluster."""
        from pymongo.operations import SearchIndexModel

        index_def = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": self._embedding_dim,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "metadata.category"},
                {"type": "filter", "path": "metadata.data_type"},
            ]
        }
        model = SearchIndexModel(
            definition=index_def,
            name=self._index_name,
            type="vectorSearch",
        )
        try:
            self._collection.create_search_index(model)
            logger.info("Created vector index '%s'", self._index_name)
        except Exception as e:
            logger.warning("Vector index creation: %s", e)

    def list_categories(self) -> List[str]:
        """Return distinct categories in the collection."""
        return self._collection.distinct("metadata.category")


# ── ChromaDB (local fallback) ────────────────────────────────────────

class ChromaStore:
    """ChromaDB persistent local vector store.

    Auto-used when no MONGO_URI is configured.
    """

    def __init__(
        self,
        collection_name: str = "rag_knowledge",
        persist_dir: Optional[str] = None,
    ):
        import chromadb

        self._persist_dir = persist_dir or os.getenv(
            "CHROMA_DIR", "./vectorstore"
        )
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "l2"},
        )

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple]:
        """Search with L2 distance converted to 0-1 similarity."""
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self._collection.count() or 1),
        }
        if filter:
            kwargs["where"] = {f"category": filter.get("category", "")} if "category" in filter else None
            if kwargs["where"] is None:
                del kwargs["where"]

        raw = self._collection.query(**kwargs)

        results = []
        if raw["documents"] and raw["documents"][0]:
            docs = raw["documents"][0]
            dists = raw["distances"][0] if raw["distances"] else [0.0] * len(docs)
            metas = raw["metadatas"][0] if raw["metadatas"] else [{}] * len(docs)

            for content, dist, meta in zip(docs, dists, metas):
                score = max(0.0, 1.0 - dist / 2.0)
                rag_doc = RAGDocument(content=content, metadata=meta)
                results.append((rag_doc, score))

        return results

    def upsert(
        self,
        documents: List[RAGDocument],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Insert documents with embeddings."""
        import uuid

        total = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]

            ids = [uuid.uuid4().hex[:16] for _ in batch_docs]
            texts = [d.content for d in batch_docs]
            metas = [d.metadata for d in batch_docs]

            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=batch_embs,
                metadatas=metas,
            )
            total += len(batch_docs)

        return total

    def delete(self, filter: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter."""
        try:
            where = {k: v for k, v in filter.items()}
            results = self._collection.get(where=where)
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                return len(results["ids"])
        except Exception as e:
            logger.warning("ChromaDB delete: %s", e)
        return 0

    def count(self) -> int:
        return self._collection.count()

    def list_categories(self) -> List[str]:
        """Return distinct categories."""
        all_meta = self._collection.get()["metadatas"] or []
        cats = set()
        for m in all_meta:
            if m and "category" in m:
                cats.add(m["category"])
        return sorted(cats)


# ── Factory ──────────────────────────────────────────────────────────

def get_vectorstore(
    backend: str = "auto",
    collection_name: str = "rag_knowledge",
    **kwargs,
) -> VectorStoreProtocol:
    """Create vector store. Auto-detect: Atlas if MONGO_URI set, else ChromaDB.

    backend: "auto", "atlas", or "chroma"
    """
    if backend == "atlas" or (backend == "auto" and os.getenv("MONGO_URI")):
        logger.info("Using MongoDB Atlas vector store")
        return AtlasStore(collection_name=collection_name, **kwargs)

    logger.info("Using ChromaDB local vector store (no MONGO_URI set)")
    return ChromaStore(collection_name=collection_name, **kwargs)
