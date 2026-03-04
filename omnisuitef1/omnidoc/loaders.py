"""Document format loaders — 12+ types via LangChain.

Supports: PDF, HTML, DOCX, TXT, MD, CSV, JSON, XML, RTF, ODT, Excel, PowerPoint, images.
Adapted from OmniSense enhanced_document_processor.py.
"""

from __future__ import annotations

import logging
import pathlib
import re
from typing import Any, Dict, List, Tuple

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredXMLLoader,
    UnstructuredRTFLoader,
    UnstructuredODTLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

logger = logging.getLogger(__name__)

# ── Optional format support ──────────────────────────────────────────────

try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False

try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    PPT_SUPPORT = True
except ImportError:
    PPT_SUPPORT = False

try:
    from langchain_community.document_loaders import UnstructuredImageLoader
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False


# ── Loader mapping ────────────────────────────────────────────────────────

def _json_loader(file_path: str) -> Any:
    return JSONLoader(file_path=file_path, jq_schema=".", text_content=False)


def build_loader_mapping() -> Dict[str, Any]:
    """Map file extensions to their LangChain loader classes."""
    loaders: Dict[str, Any] = {
        # Documents
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".markdown": UnstructuredMarkdownLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".rtf": UnstructuredRTFLoader,
        ".odt": UnstructuredODTLoader,
        # Web/Markup
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".xml": UnstructuredXMLLoader,
        # Data files
        ".csv": CSVLoader,
        ".json": _json_loader,
        ".jsonl": _json_loader,
    }

    if EXCEL_SUPPORT:
        loaders[".xlsx"] = UnstructuredExcelLoader
        loaders[".xls"] = UnstructuredExcelLoader

    if PPT_SUPPORT:
        loaders[".pptx"] = UnstructuredPowerPointLoader
        loaders[".ppt"] = UnstructuredPowerPointLoader

    if IMAGE_SUPPORT:
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"):
            loaders[ext] = UnstructuredImageLoader

    return loaders


LOADER_MAP = build_loader_mapping()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}


def get_supported_formats() -> List[str]:
    """Return sorted list of supported file extensions."""
    return sorted(LOADER_MAP.keys())


# ── Text cleaning ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excessive whitespace and control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Load document ─────────────────────────────────────────────────────────

def load_document(file_path: str) -> List[Document]:
    """Load a document using the appropriate LangChain loader.

    Args:
        file_path: Path to the document file.

    Returns:
        List of LangChain Document objects.
    """
    ext = pathlib.Path(file_path).suffix.lower()
    loader_class = LOADER_MAP.get(ext)

    if not loader_class:
        supported = ", ".join(get_supported_formats())
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {supported}")

    if ext in (".json", ".jsonl"):
        loader = _json_loader(file_path)
    elif callable(loader_class):
        loader = loader_class(file_path)
    else:
        raise ValueError(f"Invalid loader for {ext}")

    return loader.load()


# ── Structure-aware chunking ──────────────────────────────────────────────

def chunk_documents(
    docs: List[Document],
    file_type: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Structure-aware chunking that respects page/section/row boundaries.

    Returns:
        (chunk_texts, chunk_metadata) — parallel lists.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks: List[str] = []
    meta: List[Dict[str, Any]] = []

    if file_type == ".pdf":
        # PDFs: PyPDFLoader gives one Document per page
        for page_idx, doc in enumerate(docs):
            page_num = doc.metadata.get("page", page_idx) + 1
            page_chunks = splitter.split_text(clean_text(doc.page_content))
            for chunk_text in page_chunks:
                chunks.append(chunk_text)
                meta.append({"page": page_num, "source": f"Page {page_num}"})

    elif file_type in (".md", ".markdown"):
        # Markdown: split on headers first, then by size
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"),
            ],
        )
        full_text = "\n\n".join(doc.page_content for doc in docs)
        header_docs = md_splitter.split_text(full_text)
        for hd in header_docs:
            section = " > ".join(
                hd.metadata.get(k, "")
                for k in ("h1", "h2", "h3", "h4")
                if hd.metadata.get(k)
            ) or "Top"
            sub_chunks = splitter.split_text(clean_text(hd.page_content))
            for chunk_text in sub_chunks:
                chunks.append(chunk_text)
                meta.append({"section": section, "source": section})

    elif file_type == ".csv":
        # CSV: one Document per row — keep rows intact
        for row_idx, doc in enumerate(docs):
            text = clean_text(doc.page_content)
            if text:
                chunks.append(text)
                meta.append({"row": row_idx + 1, "source": f"Row {row_idx + 1}"})

    else:
        # Default: combine all, split by character
        full_text = " ".join(doc.page_content for doc in docs)
        clean = clean_text(full_text)
        plain_chunks = splitter.split_text(clean)
        for i, chunk_text in enumerate(plain_chunks):
            chunks.append(chunk_text)
            meta.append({"chunk_index": i, "source": f"Chunk {i + 1}"})

    return chunks, meta
