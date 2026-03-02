"""Enhanced PDF processing via PyMuPDF (fitz) — text + image extraction.

Adapted from cadAI enhanced_extract.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def extract_text_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text per page using PyMuPDF.

    Returns:
        [{"page": 1, "text": "...", "word_count": 42}, ...]
    """
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text") or ""
        pages.append({
            "page": page_num + 1,
            "text": text,
            "word_count": len(text.split()),
        })

    doc.close()
    return pages


def extract_images_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    min_size: int = 100,
) -> List[Dict[str, Any]]:
    """Extract embedded images from a PDF.

    Args:
        pdf_path: Path to PDF file.
        output_dir: Directory to save extracted images.
        min_size: Minimum width or height in pixels to keep an image.

    Returns:
        [{"page": 1, "index": 0, "path": Path, "width": 800, "height": 600, "format": "png"}, ...]
    """
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_ext = base_image.get("ext", "png")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < min_size and height < min_size:
                    continue

                img_filename = f"page{page_num + 1}_img{img_idx + 1}.{img_ext}"
                img_path = output_dir / img_filename

                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                images.append({
                    "page": page_num + 1,
                    "index": img_idx,
                    "path": img_path,
                    "width": width,
                    "height": height,
                    "format": img_ext,
                })

            except Exception as e:
                logger.debug("Failed to extract image xref=%d from %s: %s", xref, pdf_path.name, e)

    doc.close()
    return images
