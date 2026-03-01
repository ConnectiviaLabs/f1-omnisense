"""OmniDoc APIRouter — unified document processing with dual embeddings and deep extraction.

Endpoints:
    POST /api/omni/doc/process       — process a document (text + tables + images + embeddings)
    POST /api/omni/doc/upload        — process + ingest into RAG knowledge base (replaces legacy /upload)
    GET  /api/omni/doc/formats       — list supported file formats
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/omni/doc", tags=["OmniDoc"])

SUPPORTED_FORMATS = {
    ".pdf", ".docx", ".txt", ".csv", ".json", ".md",
    ".html", ".xml", ".rtf", ".odt", ".xlsx", ".xls", ".pptx",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}


@router.post("/process")
async def process_document(
    file: UploadFile = File(...),
    embed: bool = Form(True),
    extract_tables: bool = Form(True),
    extract_images: bool = Form(True),
    deep_extract: bool = Form(False),
    deep_extract_mode: str = Form("cloud"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
):
    """Process a document using OmniDoc — returns chunks, embeddings, tables, images."""
    from omnidoc import process_document as _process

    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported format: {ext}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = _process(
            tmp_path,
            embed=embed,
            extract_tables=extract_tables,
            extract_images=extract_images,
            deep_extract=deep_extract,
            deep_extract_mode=deep_extract_mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            save_json=False,
        )

        response = result.to_dict(include_embeddings=False)
        response["filename"] = filename
        response["status"] = "ok"
        response["num_chunks"] = len(result.chunks)
        response["num_tables"] = len(result.tables)
        response["num_images"] = len(result.images)

        return response

    except Exception as e:
        logger.exception("OmniDoc processing failed for %s", filename)
        raise HTTPException(500, f"Processing failed: {e}")
    finally:
        os.unlink(tmp_path)


@router.post("/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    deep_extract: bool = Form(False),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
):
    """Process with OmniDoc then ingest chunks into the RAG knowledge base.

    Enhanced replacement for the legacy /upload endpoint — produces richer
    metadata (tables, tags, page numbers) and uses the same NomicEmbedder
    (768-dim) for Atlas Vector Search compatibility.
    """
    from omnidoc import process_document as _process

    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported format: {ext}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. Process with OmniDoc (skip OmniDoc's BGE embeddings — we use NomicEmbedder for Atlas compat)
        result = _process(
            tmp_path,
            embed=False,  # We'll embed with NomicEmbedder below
            extract_tables=True,
            extract_images=True,
            deep_extract=deep_extract,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            save_json=False,
        )

        if not result.chunks:
            return {"filename": filename, "status": "error", "error": "No text extracted"}

        # 2. Build LangChain Documents with rich OmniDoc metadata
        from langchain_core.documents import Document

        docs = []
        for i, chunk in enumerate(result.chunks):
            meta = result.chunk_metadata[i] if i < len(result.chunk_metadata) else {}
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "data_type": "uploaded_document",
                    "category": "user_upload",
                    "source": filename,
                    "page": meta.get("page", 0),
                    "chunk": i + 1,
                    "total_chunks": len(result.chunks),
                    "tags": meta.get("tags", []),
                    "section": meta.get("section", ""),
                },
            ))

        # 3. Embed with BGE (1024-dim, matches Atlas vector_index)
        from omnidoc.embedder import get_embedder
        embedder = get_embedder(enable_clip=False)
        texts = [doc.page_content for doc in docs]
        embeddings = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            embeddings.extend(embedder.embed_texts(batch))

        # 4. Upsert to Atlas
        from pipeline.vectorstore import AtlasVectorStore
        vs = AtlasVectorStore()
        count = vs.upsert_documents(docs, embeddings)

        return {
            "filename": filename,
            "status": "ok",
            "chunks": count,
            "text_length": sum(len(c) for c in result.chunks),
            "tables": len(result.tables),
            "images": len(result.images),
            "deep_extracted": deep_extract,
        }

    except Exception as e:
        logger.exception("OmniDoc upload failed for %s", filename)
        return {"filename": filename, "status": "error", "error": str(e)}
    finally:
        os.unlink(tmp_path)


@router.get("/formats")
def list_formats():
    """List supported document formats."""
    return {
        "formats": sorted(SUPPORTED_FORMATS),
        "categories": {
            "documents": [".pdf", ".docx", ".txt", ".md", ".rtf", ".odt", ".html", ".xml"],
            "spreadsheets": [".csv", ".json", ".xlsx", ".xls"],
            "presentations": [".pptx"],
            "images": [".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
        },
    }
