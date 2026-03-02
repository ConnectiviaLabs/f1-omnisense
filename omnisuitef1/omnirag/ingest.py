"""Document ingestion: load → chunk → embed → upsert to vector store.

Imports document loading and chunking from omnidoc. Adds embedding + vectorstore upsert.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnirag._types import IngestResult, RAGDocument
from omnirag.vectorstore import VectorStoreProtocol

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


def _compute_hash(text: str) -> str:
    """SHA-256 hash of text (first 16 chars)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ingest_file(
    file_path: str | Path,
    vectorstore: VectorStoreProtocol,
    *,
    category: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> IngestResult:
    """Ingest a file into the vector store.

    1. Load document via omnidoc loaders
    2. Chunk via omnidoc chunkers
    3. Clean text
    4. Embed via omnidoc embedder
    5. Upsert to vectorstore

    Supports: PDF, DOCX, TXT, MD, CSV, JSON, HTML, and more.
    """
    from omnidoc.loaders import load_document, chunk_documents, clean_text
    from omnidoc.embedder import get_embedder

    file_path = Path(file_path)
    file_name = file_path.name
    file_type = file_path.suffix.lower()

    try:
        # 1. Load
        docs = load_document(str(file_path))
        if not docs:
            return IngestResult(
                file_name=file_name, chunk_count=0,
                status="error", error="No content extracted",
            )

        # 2. Chunk
        chunks, chunk_meta = chunk_documents(
            docs, file_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )

        # 3. Clean
        chunks = [clean_text(c) for c in chunks]
        chunks = [c for c in chunks if c.strip()]

        if not chunks:
            return IngestResult(
                file_name=file_name, chunk_count=0,
                status="error", error="No text after cleaning",
            )

        # Content hash for dedup
        full_text = "\n".join(chunks)
        content_hash = _compute_hash(full_text)

        # 4. Build RAGDocuments with metadata
        rag_docs = []
        for i, (chunk, meta) in enumerate(zip(chunks, chunk_meta)):
            doc_meta = {
                "source": file_name,
                "category": category or _auto_category(file_name),
                "data_type": file_type.lstrip("."),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            # Merge chunk-specific metadata (page, section, etc.)
            doc_meta.update(meta)

            rag_docs.append(RAGDocument(
                content=chunk,
                metadata=doc_meta,
                content_hash=_compute_hash(chunk),
            ))

        # 5. Embed
        embedder = get_embedder()
        embeddings = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            embeddings.extend(embedder.embed_texts(batch))

        # 6. Upsert
        count = vectorstore.upsert(rag_docs, embeddings, batch_size=BATCH_SIZE)
        logger.info("Ingested %s: %d chunks", file_name, count)

        return IngestResult(
            file_name=file_name,
            chunk_count=count,
            content_hash=content_hash,
            status="success",
        )

    except Exception as e:
        logger.error("Failed to ingest %s: %s", file_name, e)
        return IngestResult(
            file_name=file_name, chunk_count=0,
            status="error", error=str(e),
        )


def ingest_texts(
    texts: List[str],
    vectorstore: VectorStoreProtocol,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    source: str = "manual",
    category: str = "text",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> IngestResult:
    """Ingest raw text strings into the vector store.

    Useful for ingesting API responses, scraped content, or structured data.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from omnidoc.embedder import get_embedder

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for text in texts:
        all_chunks.extend(splitter.split_text(text))

    all_chunks = [c.strip() for c in all_chunks if c.strip()]
    if not all_chunks:
        return IngestResult(file_name=source, chunk_count=0, status="empty")

    content_hash = _compute_hash("\n".join(all_chunks))

    rag_docs = []
    base_meta = metadata or {}
    for i, chunk in enumerate(all_chunks):
        doc_meta = {
            "source": source,
            "category": category,
            "chunk_index": i,
            "total_chunks": len(all_chunks),
            **base_meta,
        }
        rag_docs.append(RAGDocument(
            content=chunk,
            metadata=doc_meta,
            content_hash=_compute_hash(chunk),
        ))

    embedder = get_embedder()
    embeddings = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        embeddings.extend(embedder.embed_texts(batch))

    count = vectorstore.upsert(rag_docs, embeddings, batch_size=BATCH_SIZE)

    return IngestResult(
        file_name=source,
        chunk_count=count,
        content_hash=content_hash,
        status="success",
    )


def ingest_directory(
    dir_path: str | Path,
    vectorstore: VectorStoreProtocol,
    *,
    extensions: Optional[List[str]] = None,
    category: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[IngestResult]:
    """Ingest all supported files from a directory."""
    from omnidoc.loaders import get_supported_formats

    dir_path = Path(dir_path)
    allowed = set(extensions or get_supported_formats())

    results = []
    for file_path in sorted(dir_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in allowed:
            result = ingest_file(
                file_path, vectorstore,
                category=category,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            results.append(result)

    return results


def _auto_category(filename: str) -> str:
    """Simple auto-categorization by filename patterns."""
    lower = filename.lower()
    if "spec" in lower or "criteria" in lower:
        return "specification"
    if "drawing" in lower or "diagram" in lower or "p&id" in lower:
        return "drawing"
    if "report" in lower:
        return "report"
    if "manual" in lower or "guide" in lower:
        return "manual"
    if "standard" in lower or "code" in lower:
        return "standard"
    return "document"
