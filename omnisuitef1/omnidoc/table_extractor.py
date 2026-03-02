"""Table extraction from PDFs — pdfplumber (primary) + Camelot (fallback).

Adapted from cadAI enhanced_extract.py and pdf_enhanced.py.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

TABLE_EXTRACTION_TIMEOUT = 30  # seconds per page

# ── Optional imports ──────────────────────────────────────────────────────

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False


# ── pdfplumber extraction ─────────────────────────────────────────────────

def extract_tables_pdfplumber(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract tables using pdfplumber with per-page timeout protection."""
    if not HAS_PDFPLUMBER:
        logger.warning("pdfplumber not installed — skipping table extraction")
        return []

    tables = []
    timeout_pages = []

    def _extract_page(page, page_num):
        page_tables = []
        try:
            raw_tables = page.extract_tables()
            for table_idx, table in enumerate(raw_tables):
                if table and len(table) > 1:
                    cleaned = []
                    for row in table:
                        cleaned_row = [
                            str(cell).strip() if cell else ""
                            for cell in row
                        ]
                        if any(cleaned_row):
                            cleaned.append(cleaned_row)
                    if len(cleaned) > 1:
                        page_tables.append({
                            "page": page_num + 1,
                            "table_index": table_idx + 1,
                            "headers": cleaned[0],
                            "rows": cleaned[1:],
                            "num_rows": len(cleaned) - 1,
                            "num_cols": len(cleaned[0]) if cleaned else 0,
                            "method": "pdfplumber",
                        })
        except Exception:
            pass
        return page_tables

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_extract_page, page, page_num)
                        try:
                            page_tables = future.result(timeout=TABLE_EXTRACTION_TIMEOUT)
                            tables.extend(page_tables)
                        except FuturesTimeoutError:
                            timeout_pages.append(page_num + 1)
                            logger.warning(
                                "Table extraction timeout on %s page %d",
                                pdf_path.name, page_num + 1,
                            )
                except Exception:
                    pass
    except Exception as e:
        logger.warning("pdfplumber failed on %s: %s", pdf_path.name, e)

    if timeout_pages:
        logger.info("Timeout pages for %s: %s", pdf_path.name, timeout_pages)

    return tables


# ── Camelot extraction ────────────────────────────────────────────────────

def extract_tables_camelot(pdf_path: Path, max_pages: int = 10) -> List[Dict[str, Any]]:
    """Extract tables using Camelot — lattice first, stream fallback."""
    if not HAS_CAMELOT:
        logger.warning("Camelot not installed — skipping Camelot table extraction")
        return []

    tables = []
    pages_str = f"1-{max_pages}"

    try:
        # Try lattice mode first (bordered tables)
        camelot_tables = camelot.read_pdf(str(pdf_path), pages=pages_str, flavor="lattice")
        for table in camelot_tables:
            df = table.df
            if len(df) > 1:
                tables.append({
                    "page": table.page,
                    "headers": df.iloc[0].tolist(),
                    "rows": df.iloc[1:].values.tolist(),
                    "num_rows": len(df) - 1,
                    "accuracy": table.accuracy,
                    "method": "camelot_lattice",
                })

        # If no tables found, try stream mode (borderless)
        if not tables:
            camelot_tables = camelot.read_pdf(str(pdf_path), pages=pages_str, flavor="stream")
            for table in camelot_tables:
                df = table.df
                if len(df) > 1:
                    tables.append({
                        "page": table.page,
                        "headers": df.iloc[0].tolist(),
                        "rows": df.iloc[1:].values.tolist(),
                        "num_rows": len(df) - 1,
                        "accuracy": table.accuracy,
                        "method": "camelot_stream",
                    })

    except Exception as e:
        logger.warning("Camelot extraction failed on %s: %s", Path(pdf_path).name, e)

    return tables


# ── Combined extraction ───────────────────────────────────────────────────

def extract_tables(pdf_path: Path, max_pages: int = 10) -> List[Dict[str, Any]]:
    """Combined table extraction: pdfplumber primary, Camelot fills gaps.

    Returns all tables found, with 'method' field indicating source.
    """
    # Primary: pdfplumber (fast, per-page timeout)
    tables = extract_tables_pdfplumber(pdf_path)
    plumber_pages = {t["page"] for t in tables}

    # Fallback: Camelot for pages pdfplumber missed
    if HAS_CAMELOT:
        camelot_tables = extract_tables_camelot(pdf_path, max_pages=max_pages)
        for t in camelot_tables:
            if t["page"] not in plumber_pages:
                tables.append(t)

    tables.sort(key=lambda t: (t["page"], t.get("table_index", 0)))
    return tables
