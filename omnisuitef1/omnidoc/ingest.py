"""Unified document ingestion — single entry point combining all modules.

Usage:
    from omnidoc import process_document

    result = process_document("report.pdf")

    # Access results
    result.chunks          # text chunks
    result.embeddings      # BGE 1024-dim vectors (parallel to chunks)
    result.tables          # extracted tables (PDF only)
    result.images          # extracted image paths (PDF only)
    result.image_embeddings  # CLIP 512-dim vectors (parallel to images)
    result.deep_extraction # 5-pass Groq Vision results (PDF only, optional)
    result.metadata        # file info, timing, costs
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

# ── Regex patterns for metadata enrichment (from cadAI + F1) ─────────────

RE_EQUIPMENT = re.compile(r"\b[CPEF][A-Z]?\d{4}[A-Z]?\b")
RE_LINE_REF = re.compile(r"\b\d{2,3}-[A-Z]-\d{1,3}\b")
RE_KKS = re.compile(r"\b\d{2}[A-Z]{2,3}\d{2}[A-Z]{2}\d{3}\b")
RE_PIPING_CLASS = re.compile(r"\b[A-D]\d[A-Z]\b")
RE_STANDARD = re.compile(r"\b(?:ASME|ANSI|API|ASTM|ISO|DIN|EN)\s*[A-Z]?\d[\d.]*(?:-\d+)?\b")


@dataclass
class ProcessedDocument:
    """Complete processing result for a single document."""

    file_path: str
    file_type: str
    content_hash: str = ""

    # Text content
    chunks: List[str] = field(default_factory=list)
    chunk_metadata: List[Dict[str, Any]] = field(default_factory=list)

    # BGE embeddings (1024-dim, parallel to chunks)
    embeddings: List[List[float]] = field(default_factory=list)

    # Tables (PDF only)
    tables: List[Dict[str, Any]] = field(default_factory=list)

    # Images
    images: List[Dict[str, Any]] = field(default_factory=list)
    image_embeddings: List[List[float]] = field(default_factory=list)

    # Deep extraction (PDF only, optional)
    deep_extraction: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_embeddings: bool = True) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d: Dict[str, Any] = {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "chunks": [
                {
                    "text": text,
                    "metadata": self.chunk_metadata[i] if i < len(self.chunk_metadata) else {},
                    **({"embedding": self.embeddings[i]} if include_embeddings and i < len(self.embeddings) else {}),
                }
                for i, text in enumerate(self.chunks)
            ],
            "tables": self.tables,
            "images": [
                {
                    **{k: str(v) if isinstance(v, Path) else v for k, v in img.items()},
                    **({"embedding": self.image_embeddings[i]} if include_embeddings and i < len(self.image_embeddings) else {}),
                }
                for i, img in enumerate(self.images)
            ],
        }
        if self.deep_extraction:
            d["deep_extraction"] = self.deep_extraction
        return d

    def save_json(self, path: str | Path | None = None, include_embeddings: bool = True) -> Path:
        """Save results to a JSON file.

        Args:
            path: Output file path. If None, saves next to the source file.
            include_embeddings: Include embedding vectors (large). Set False for compact output.

        Returns:
            Path to the saved JSON file.
        """
        if path is None:
            src = Path(self.file_path)
            path = src.parent / f"{src.stem}_omnidoc.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict(include_embeddings=include_embeddings)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("Saved results → %s (%.1f MB)", path, size_mb)
        return path


def process_document(
    file_path: str | Path,
    *,
    embed: bool = True,
    extract_tables: bool = True,
    extract_images: bool = True,
    deep_extract: bool = False,
    deep_extract_mode: str = "cloud",
    deep_extract_passes: Optional[List[int]] = None,
    output_dir: Optional[str | Path] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    save_json: bool = True,
    json_path: Optional[str | Path] = None,
    include_embeddings_in_json: bool = True,
) -> ProcessedDocument:
    """Process any supported document into chunks, embeddings, tables, and images.

    Args:
        file_path: Path to the document file.
        embed: Generate BGE text embeddings + CLIP image embeddings.
        extract_tables: Extract tables from PDFs (pdfplumber + Camelot).
        extract_images: Extract embedded images from PDFs + CLIP embed them.
        deep_extract: Run 5-pass Groq Vision extraction (PDF only).
        deep_extract_mode: "cloud" (Groq) or "edge" (Ollama).
        deep_extract_passes: Specific passes to run (1-5). None = all.
        output_dir: Directory for rendered images / deep extraction output.
        chunk_size: Text chunk size in characters.
        chunk_overlap: Overlap between chunks.
        save_json: Auto-save results to JSON file.
        json_path: Custom path for JSON output. None = next to source file.
        include_embeddings_in_json: Include embedding vectors in JSON (large).

    Returns:
        ProcessedDocument with all extracted data.
    """
    t0 = time.time()
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    out_dir = Path(output_dir) if output_dir else file_path.parent / f".omnidoc_{file_path.stem}"

    # ── SHA-256 content hash (from OmniSense — enables dedup) ────────────
    content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

    result = ProcessedDocument(
        file_path=str(file_path),
        file_type=ext,
        content_hash=content_hash,
        metadata={"file_name": file_path.name, "file_size_bytes": file_path.stat().st_size,
                  "content_hash": content_hash},
    )

    is_pdf = ext == ".pdf"
    is_image = ext in IMAGE_EXTENSIONS

    # ── Load and chunk text ──────────────────────────────────────────────
    if is_pdf:
        _process_pdf_text(file_path, result, chunk_size, chunk_overlap)
    elif is_image:
        _process_image(file_path, result)
    else:
        _process_generic(file_path, ext, result, chunk_size, chunk_overlap)

    # ── Enrich chunk metadata with regex patterns (from cadAI + F1) ─────
    _enrich_chunk_metadata(result)

    # ── PDF: OCR fallback for zero-text pages (from cadAI + F1) ──────────
    if is_pdf and result.metadata.get("total_words", 0) == 0:
        _try_ocr_fallback(file_path, result, chunk_size, chunk_overlap)

    # ── PDF: table extraction ────────────────────────────────────────────
    if is_pdf and extract_tables:
        _process_tables(file_path, result)

    # ── PDF: image extraction ────────────────────────────────────────────
    if is_pdf and extract_images:
        _process_pdf_images(file_path, out_dir, result)

    # ── Embeddings ───────────────────────────────────────────────────────
    if embed:
        _generate_embeddings(result)

    # ── PDF: deep extraction (5-pass vision) ─────────────────────────────
    if is_pdf and deep_extract:
        _process_deep_extraction(
            file_path, out_dir, result,
            mode=deep_extract_mode,
            pass_list=deep_extract_passes,
        )

    result.metadata["processing_time_s"] = round(time.time() - t0, 2)
    result.metadata["chunk_count"] = len(result.chunks)
    result.metadata["table_count"] = len(result.tables)
    result.metadata["image_count"] = len(result.images)

    logger.info(
        "Processed %s: %d chunks, %d tables, %d images (%.1fs)",
        file_path.name, len(result.chunks), len(result.tables),
        len(result.images), result.metadata["processing_time_s"],
    )

    # ── Save to JSON ─────────────────────────────────────────────────────
    if save_json:
        saved_path = result.save_json(
            path=json_path, include_embeddings=include_embeddings_in_json,
        )
        result.metadata["json_output_path"] = str(saved_path)

    return result


# ── Internal processors ──────────────────────────────────────────────────


def _process_pdf_text(
    file_path: Path, result: ProcessedDocument,
    chunk_size: int, chunk_overlap: int,
):
    """Extract text from PDF via PyMuPDF, then chunk."""
    from omnidoc.pdf_extractor import extract_text_pymupdf
    from omnidoc.loaders import chunk_documents, clean_text
    from langchain_core.documents import Document

    pages = extract_text_pymupdf(file_path)
    result.metadata["page_count"] = len(pages)
    result.metadata["total_words"] = sum(p["word_count"] for p in pages)

    # Convert to LangChain Documents for consistent chunking
    docs = [
        Document(page_content=p["text"], metadata={"page": p["page"] - 1})
        for p in pages
    ]
    result.chunks, result.chunk_metadata = chunk_documents(
        docs, ".pdf", chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )


def _process_image(file_path: Path, result: ProcessedDocument):
    """Handle standalone image files."""
    result.images.append({
        "path": file_path,
        "source": "standalone",
        "format": file_path.suffix.lstrip("."),
    })


def _process_generic(
    file_path: Path, ext: str, result: ProcessedDocument,
    chunk_size: int, chunk_overlap: int,
):
    """Load and chunk any non-PDF, non-image format via LangChain."""
    from omnidoc.loaders import load_document, chunk_documents

    docs = load_document(str(file_path))
    result.chunks, result.chunk_metadata = chunk_documents(
        docs, ext, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )


def _enrich_chunk_metadata(result: ProcessedDocument):
    """Extract equipment tags, pipe refs, KKS codes, standards from each chunk.

    From cadAI enhanced_extract.py + F1 ingest.py — enables filtered vector search.
    """
    for i, text in enumerate(result.chunks):
        if i >= len(result.chunk_metadata):
            break
        meta = result.chunk_metadata[i]

        equipment = RE_EQUIPMENT.findall(text)
        if equipment:
            meta["equipment_tags"] = ",".join(sorted(set(equipment)))

        line_refs = RE_LINE_REF.findall(text)
        if line_refs:
            meta["line_refs"] = ",".join(sorted(set(line_refs)))

        kks = RE_KKS.findall(text)
        if kks:
            meta["kks_codes"] = ",".join(sorted(set(kks)))

        piping = RE_PIPING_CLASS.findall(text)
        if piping:
            meta["piping_classes"] = ",".join(sorted(set(piping)))

        standards = RE_STANDARD.findall(text)
        if standards:
            meta["standards"] = ",".join(sorted(set(standards)))


def _try_ocr_fallback(
    file_path: Path, result: ProcessedDocument,
    chunk_size: int, chunk_overlap: int,
):
    """EasyOCR fallback for image-based PDFs with zero native text.

    From cadAI drawing_ocr.py + F1 ocr.py.
    """
    try:
        import easyocr
        import fitz
        import numpy as np
    except ImportError:
        logger.debug("EasyOCR not available — skipping OCR fallback")
        return

    try:
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        doc = fitz.open(str(file_path))
        all_text = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            zoom = 200 / 72.0  # 200 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            results = reader.readtext(img_np)
            page_text = " ".join(
                text for _, text, conf in results if conf >= 0.15
            )
            if page_text.strip():
                all_text.append(page_text.strip())

        doc.close()

        if all_text:
            from omnidoc.loaders import chunk_documents, clean_text
            from langchain_core.documents import Document

            docs = [
                Document(page_content=t, metadata={"page": i})
                for i, t in enumerate(all_text)
            ]
            result.chunks, result.chunk_metadata = chunk_documents(
                docs, ".pdf", chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            )
            result.metadata["ocr_applied"] = True
            result.metadata["total_words"] = sum(len(t.split()) for t in all_text)
            logger.info("OCR: extracted %d words from %d pages",
                        result.metadata["total_words"], len(all_text))
    except Exception as e:
        logger.warning("OCR fallback failed: %s", e)


def _process_tables(file_path: Path, result: ProcessedDocument):
    """Extract tables from PDF using pdfplumber + Camelot."""
    try:
        from omnidoc.table_extractor import extract_tables
        result.tables = extract_tables(file_path)
        logger.info("Tables: %d extracted from %s", len(result.tables), file_path.name)
    except Exception as e:
        logger.warning("Table extraction failed for %s: %s", file_path.name, e)


def _process_pdf_images(
    file_path: Path, output_dir: Path, result: ProcessedDocument,
):
    """Extract embedded images from PDF."""
    try:
        from omnidoc.pdf_extractor import extract_images_from_pdf
        img_dir = output_dir / "extracted_images"
        images = extract_images_from_pdf(file_path, img_dir)
        result.images.extend(images)
        logger.info("Images: %d extracted from %s", len(images), file_path.name)
    except Exception as e:
        logger.warning("Image extraction failed for %s: %s", file_path.name, e)


def _generate_embeddings(result: ProcessedDocument):
    """Generate BGE text embeddings + CLIP image embeddings."""
    from omnidoc.embedder import get_embedder

    embedder = get_embedder()

    # Text embeddings (BGE 1024-dim)
    if result.chunks:
        try:
            result.embeddings = embedder.embed_texts(result.chunks)
            logger.info("BGE embeddings: %d vectors (1024-dim)", len(result.embeddings))
        except Exception as e:
            logger.warning("BGE embedding failed: %s", e)

    # Image embeddings (CLIP 512-dim)
    image_paths = [
        Path(img["path"]) for img in result.images
        if Path(img["path"]).exists()
    ]
    if image_paths:
        try:
            result.image_embeddings = embedder.embed_images(image_paths)
            logger.info("CLIP embeddings: %d vectors (512-dim)", len(result.image_embeddings))
        except Exception as e:
            logger.warning("CLIP embedding failed: %s", e)


def _process_deep_extraction(
    file_path: Path, output_dir: Path, result: ProcessedDocument,
    mode: str, pass_list: Optional[List[int]],
):
    """Run 5-pass Groq/Ollama vision extraction."""
    try:
        from omnidoc.deep_extractor import run_deep_extraction

        deep_result = run_deep_extraction(
            pdf_path=file_path,
            output_dir=output_dir,
            mode=mode,
            pass_list=pass_list,
        )
        result.deep_extraction = deep_result
        cost = deep_result.get("metadata", {}).get("total_cost_usd", 0)
        result.metadata["deep_extraction_cost_usd"] = cost
        logger.info("Deep extraction complete (cost: $%.4f)", cost)
    except Exception as e:
        logger.warning("Deep extraction failed for %s: %s", file_path.name, e)
        result.deep_extraction = {"error": str(e)}
