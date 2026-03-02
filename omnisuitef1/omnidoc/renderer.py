"""PDF → multi-view image renderer with quadrants, zooming, and enhancement.

Converts each PDF page into multiple image views:
  - Full page at 3x zoom
  - 4 overlapping quadrants (10% overlap)
  - Auto-detected zoom regions (tables, diagrams)

Cached via manifest files — re-rendering skipped if output exists and PDF unchanged.
Adapted from F1/cadAI renderer.py.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 300_000_000

# ── Configuration ─────────────────────────────────────────────────────────

RENDER_ZOOM = 3
OVERLAP = 0.10
MAX_PIXELS = 30_000_000
MIN_REGION_PX = 200
TABLE_ZOOM = 2.0
DIAGRAM_ZOOM = 1.5
MAX_B64_BYTES = 4_500_000
MAX_DIMENSION = 7900


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class ZoomedRegion:
    path: Path
    region_type: str  # "table" or "diagram"
    bbox: tuple[int, int, int, int]
    zoom_factor: float
    source_page: int


@dataclass
class PageViews:
    page_num: int
    full: Path
    quadrants: dict[str, Path] = field(default_factory=dict)
    zoomed: list[ZoomedRegion] = field(default_factory=list)
    native_text: str = ""
    native_tables: list = field(default_factory=list)
    word_count: int = 0


# ── Image processing ──────────────────────────────────────────────────────

def enhance_image(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.DETAIL)
    return img


def _encode_single(img: Image.Image) -> str:
    px = img.size[0] * img.size[1]
    if px > MAX_PIXELS:
        scale = (MAX_PIXELS / px) ** 0.5
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS
        )
    for _ in range(5):
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode()
        if len(encoded) <= MAX_B64_BYTES:
            return encoded
        img = img.resize(
            (int(img.size[0] * 0.7), int(img.size[1] * 0.7)), Image.LANCZOS
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def img_to_b64(img_or_path) -> str:
    if isinstance(img_or_path, (str, Path)):
        img = Image.open(img_or_path)
    else:
        img = img_or_path
    w, h = img.size
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        scale = min(MAX_DIMENSION / w, MAX_DIMENSION / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return _encode_single(img)


def split_quadrants(img: Image.Image, overlap: float = OVERLAP) -> dict[str, Image.Image]:
    w, h = img.size
    hw = int(w * (0.5 + overlap))
    hh = int(h * (0.5 + overlap))
    ow = int(w * (0.5 - overlap))
    oh = int(h * (0.5 - overlap))
    return {
        "q1_top_left": img.crop((0, 0, hw, hh)),
        "q2_top_right": img.crop((ow, 0, w, hh)),
        "q3_bottom_left": img.crop((0, oh, hw, h)),
        "q4_bottom_right": img.crop((ow, oh, w, h)),
    }


# ── Zoom region detection ─────────────────────────────────────────────────

def _detect_table_regions(
    pdf_path: Path, page_num: int, rendered_w: int, rendered_h: int
) -> list[tuple[int, int, int, int]]:
    regions = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return regions
            page = pdf.pages[page_num]
            pw, ph = float(page.width), float(page.height)
            sx = rendered_w / pw
            sy = rendered_h / ph
            tables = page.find_tables()
            for tbl in tables:
                bbox = tbl.bbox
                rx0 = max(0, int(bbox[0] * sx) - 10)
                ry0 = max(0, int(bbox[1] * sy) - 10)
                rx1 = min(rendered_w, int(bbox[2] * sx) + 10)
                ry1 = min(rendered_h, int(bbox[3] * sy) + 10)
                if (rx1 - rx0) >= MIN_REGION_PX and (ry1 - ry0) >= MIN_REGION_PX:
                    regions.append((rx0, ry0, rx1, ry1))
    except Exception:
        pass
    return regions


def _extract_native_tables(pdf_path: Path, page_num: int) -> list[dict]:
    tables_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return tables_data
            page = pdf.pages[page_num]
            for tbl in page.extract_tables():
                if tbl and len(tbl) > 1:
                    headers = [str(c or "").strip() for c in tbl[0]]
                    rows = [
                        [str(c or "").strip() for c in row]
                        for row in tbl[1:]
                        if any(c for c in row)
                    ]
                    if headers and rows:
                        tables_data.append({"headers": headers, "rows": rows})
    except Exception:
        pass
    return tables_data


def detect_and_zoom_regions(
    img: Image.Image, pdf_path: Path, page_num: int,
    output_dir: Path, stem: str,
) -> list[ZoomedRegion]:
    regions = []
    w, h = img.size
    table_bboxes = _detect_table_regions(pdf_path, page_num, w, h)
    for i, bbox in enumerate(table_bboxes[:3]):
        crop = img.crop(bbox)
        new_w = int(crop.size[0] * TABLE_ZOOM)
        new_h = int(crop.size[1] * TABLE_ZOOM)
        zoomed = crop.resize((new_w, new_h), Image.LANCZOS)
        zoomed = enhance_image(zoomed)
        out_path = output_dir / f"{stem}_zoom_table_{i}.png"
        zoomed.save(out_path, optimize=True)
        regions.append(ZoomedRegion(
            path=out_path, region_type="table", bbox=bbox,
            zoom_factor=TABLE_ZOOM, source_page=page_num,
        ))
    return regions


# ── Page rendering ─────────────────────────────────────────────────────────

def render_page(
    doc: fitz.Document, page_idx: int, pdf_path: Path, output_dir: Path,
) -> PageViews:
    stem = f"page_{page_idx + 1:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    page = doc[page_idx]
    mat = fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_enhanced = enhance_image(img)

    full_path = output_dir / f"{stem}_full.png"
    img_enhanced.save(full_path, optimize=True)

    native_text = page.get_text("text") or ""
    word_count = len(native_text.split())
    native_tables = _extract_native_tables(pdf_path, page_idx)

    quads = split_quadrants(img_enhanced)
    quad_paths = {}
    for name, quad_img in quads.items():
        qpath = output_dir / f"{stem}_{name}.png"
        quad_img.save(qpath, optimize=True)
        quad_paths[name] = qpath

    zoomed = detect_and_zoom_regions(img_enhanced, pdf_path, page_idx, output_dir, stem)

    return PageViews(
        page_num=page_idx + 1, full=full_path, quadrants=quad_paths,
        zoomed=zoomed, native_text=native_text, native_tables=native_tables,
        word_count=word_count,
    )


# ── PDF rendering with caching ────────────────────────────────────────────

def _pdf_stem(pdf_path: Path) -> str:
    stem = pdf_path.stem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")[:80]


def render_pdf(
    pdf_path: Path, output_base: Path, force: bool = False,
) -> list[PageViews]:
    """Render all pages of a PDF into multi-view images. Cached via manifest."""
    stem = _pdf_stem(pdf_path)
    output_dir = output_base / stem

    manifest_path = output_dir / "_manifest.json"
    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        pdf_mtime = os.path.getmtime(pdf_path)
        if manifest.get("pdf_mtime") == pdf_mtime:
            pages = []
            for pm in manifest["pages"]:
                pv = PageViews(
                    page_num=pm["page_num"], full=Path(pm["full"]),
                    quadrants={k: Path(v) for k, v in pm["quadrants"].items()},
                    zoomed=[
                        ZoomedRegion(
                            path=Path(z["path"]), region_type=z["region_type"],
                            bbox=tuple(z["bbox"]), zoom_factor=z["zoom_factor"],
                            source_page=z["source_page"],
                        )
                        for z in pm.get("zoomed", [])
                    ],
                    native_text=pm.get("native_text", ""),
                    native_tables=pm.get("native_tables", []),
                    word_count=pm.get("word_count", 0),
                )
                pages.append(pv)
            logger.info("[cached] %s: %d pages", stem, len(pages))
            return pages

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    pages = []

    for i in range(total):
        pv = render_page(doc, i, pdf_path, output_dir)
        pages.append(pv)

    doc.close()

    manifest = {
        "pdf_path": str(pdf_path),
        "pdf_mtime": os.path.getmtime(pdf_path),
        "stem": stem,
        "page_count": total,
        "pages": [
            {
                "page_num": pv.page_num,
                "full": str(pv.full),
                "quadrants": {k: str(v) for k, v in pv.quadrants.items()},
                "zoomed": [
                    {
                        "path": str(z.path), "region_type": z.region_type,
                        "bbox": list(z.bbox), "zoom_factor": z.zoom_factor,
                        "source_page": z.source_page,
                    }
                    for z in pv.zoomed
                ],
                "native_text": pv.native_text,
                "native_tables": pv.native_tables,
                "word_count": pv.word_count,
            }
            for pv in pages
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    zoom_count = sum(len(pv.zoomed) for pv in pages)
    logger.info("[rendered] %s: %d pages, %d quadrants, %d zoom regions",
                stem, total, total * 4, zoom_count)

    return pages
