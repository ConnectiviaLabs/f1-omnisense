"""MongoDB Atlas Vector Store for F1 knowledge base.

Stores document chunks with 768-dim nomic-embed-text embeddings.
Supports $vectorSearch for semantic retrieval with category filtering.

Implements VectorStoreProtocol from retriever.py so it works with
DocumentRetriever out of the box.

Usage:
    from pipeline.vectorstore import AtlasVectorStore
    from pipeline.retriever import DocumentRetriever

    vs = AtlasVectorStore()
    retriever = DocumentRetriever(vs)
    results = retriever.search("tire compound regulations", k=5)
"""

from __future__ import annotations

import os
from pathlib import Path

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel


def _load_env():
    """Load .env from pipeline directory or project root."""
    try:
        from dotenv import load_dotenv
        for p in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
            if p.exists():
                load_dotenv(p)
                return
        load_dotenv()
    except ImportError:
        pass


# ── Constants ────────────────────────────────────────────────────────────

COLLECTION_NAME = "f1_knowledge"
INDEX_NAME = "vector_index"
EMBEDDING_DIM = 1024  # BGE-large-en-v1.5


class AtlasVectorStore:
    """MongoDB Atlas vector store with $vectorSearch support."""

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        collection_name: str = COLLECTION_NAME,
    ):
        _load_env()
        self._uri = uri or os.environ.get("MONGODB_URI", "")
        self._db_name = db_name or os.environ.get("MONGODB_DB", "marip_f1")
        self._collection_name = collection_name

        if not self._uri:
            raise ValueError("MONGODB_URI not set. Add it to .env or environment.")

        self._client = MongoClient(self._uri)
        self._db = self._client[self._db_name]
        self._collection = self._db[self._collection_name]

        print(f"  Atlas: {self._db_name}.{self._collection_name} ({self._uri[:40]}...)")

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count_documents({})

    def delete_collection(self):
        """Drop and recreate the collection."""
        self._collection.drop()
        self._collection = self._db[self._collection_name]
        print(f"  Dropped and recreated {self._collection_name}")

    # ── Index Management ─────────────────────────────────────────────────

    def ensure_vector_index(self):
        """Create Atlas vector search index if it doesn't exist.

        NOTE: Atlas vector search indexes must be created via the Atlas UI
        or Atlas Admin API for M0/M2/M5 clusters. This method attempts
        programmatic creation which works on M10+ clusters.
        For free-tier clusters, create the index manually in Atlas UI:

        Index name: vector_index
        Index definition:
        {
          "fields": [
            {
              "type": "vector",
              "path": "embedding",
              "numDimensions": 1024,
              "similarity": "cosine"
            },
            {
              "type": "filter",
              "path": "metadata.category"
            },
            {
              "type": "filter",
              "path": "metadata.data_type"
            }
          ]
        }
        """
        try:
            existing = list(self._collection.list_search_indexes())
            for idx in existing:
                if idx.get("name") == INDEX_NAME:
                    print(f"  Vector index '{INDEX_NAME}' already exists")
                    return

            search_index = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": EMBEDDING_DIM,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "metadata.category"},
                        {"type": "filter", "path": "metadata.data_type"},
                    ],
                },
                name=INDEX_NAME,
                type="vectorSearch",
            )
            self._collection.create_search_index(model=search_index)
            print(f"  Created vector index '{INDEX_NAME}' ({EMBEDDING_DIM}-dim, cosine)")
        except Exception as e:
            print(f"  Vector index creation failed: {e}")
            print("  Create it manually in Atlas UI (see docstring above)")

    # ── Upsert ───────────────────────────────────────────────────────────

    def upsert_documents(
        self,
        documents: list,
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> int:
        """Bulk upsert documents with their embeddings.

        Args:
            documents: LangChain Document objects (page_content + metadata).
            embeddings: Corresponding embedding vectors (768-dim).
            batch_size: Documents per insert batch.

        Returns:
            Number of documents inserted.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents ({len(documents)}) and embeddings ({len(embeddings)}) must match"
            )

        total = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_vecs = embeddings[i : i + batch_size]

            records = []
            for doc, vec in zip(batch_docs, batch_vecs):
                records.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": vec,
                })

            result = self._collection.insert_many(records)
            total += len(result.inserted_ids)

        return total

    # ── Search (VectorStoreProtocol) ─────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
        query_embedding: list[float] | None = None,
    ) -> list:
        """Semantic search using Atlas $vectorSearch.

        Either provide query_embedding directly, or pass query string
        (requires embedding it externally first).
        """
        if query_embedding is None:
            # Embed the query using BGE
            from omnidoc.embedder import get_embedder
            embedder = get_embedder(enable_clip=False)
            query_embedding = embedder.embed_query(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": k * 10,
                    "limit": k,
                }
            },
            {
                "$project": {
                    "page_content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                    "_id": 0,
                }
            },
        ]

        # Add filter if provided
        if filter:
            mongo_filter = {}
            for key, val in filter.items():
                mongo_filter[f"metadata.{key}"] = val
            pipeline[0]["$vectorSearch"]["filter"] = mongo_filter

        results = list(self._collection.aggregate(pipeline))

        # Convert to LangChain-style Document objects
        from langchain_core.documents import Document
        docs = []
        for r in results:
            docs.append(Document(
                page_content=r.get("page_content", ""),
                metadata={**r.get("metadata", {}), "_score": r.get("score", 0)},
            ))
        return docs

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        filter: dict | None = None,
    ) -> list:
        """MMR search — fetch more candidates then diversify.

        Falls back to similarity_search with fetch_k candidates,
        then applies simple MMR-style deduplication.
        """
        # For now, just return top-k similarity results
        return self.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple]:
        """Search with scores — returns (Document, score) tuples."""
        docs = self.similarity_search(query, k=k, filter=filter)
        return [(doc, doc.metadata.pop("_score", 0.0)) for doc in docs]
