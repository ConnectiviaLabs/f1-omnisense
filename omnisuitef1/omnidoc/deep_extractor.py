"""5-pass deep extraction orchestration + merge + master tracker.

Combines cadAI's extract.py, merge.py, and tracker.py into one standalone module.
Supports cloud (Groq Maverick) and edge (Ollama gemma + qwen) modes.

Usage:
    from omnidoc.deep_extractor import run_deep_extraction

    results = run_deep_extraction(
        pdf_path=Path("report.pdf"),
        output_dir=Path("./output"),
        mode="cloud",  # or "edge"
    )
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnidoc.extraction_passes import (
    ALL_PASSES, get_pass, select_images, should_process_page, PassDefinition,
)
from omnidoc.renderer import PageViews, render_pdf
from omnidoc.vision_client import (
    GroqVisionClient, GemmaClient, QwenVLClient, ModelResponse,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Merge helpers (from cadAI merge.py)
# ═══════════════════════════════════════════════════════════════════════════

def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _similarity(a: str | None, b: str | None) -> float:
    na, nb = _normalize(a), _normalize(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _tag_key(tag: str | None) -> str:
    if not tag:
        return ""
    return re.sub(r"[\s\-_/]", "", str(tag).upper())


def _merge_dicts(a: dict | None, b: dict | None) -> dict:
    if not a:
        return dict(b) if b else {}
    if not b:
        return dict(a)
    merged = {}
    for key in set(a) | set(b):
        av, bv = a.get(key), b.get(key)
        if av is None:
            merged[key] = bv
        elif bv is None:
            merged[key] = av
        elif isinstance(av, list) and isinstance(bv, list):
            merged[key] = av if len(av) >= len(bv) else bv
        elif isinstance(av, dict) and isinstance(bv, dict):
            merged[key] = _merge_dicts(av, bv)
        elif isinstance(av, str) and isinstance(bv, str):
            merged[key] = av if len(av) >= len(bv) else bv
        else:
            merged[key] = av
    return merged


def _merge_lists_by_key(list_a: list, list_b: list, key_field: str) -> list:
    a_map = {_normalize(item.get(key_field)): item for item in (list_a or [])}
    b_map = {_normalize(item.get(key_field)): item for item in (list_b or [])}
    merged = []
    for key in sorted(set(a_map) | set(b_map)):
        if not key:
            continue
        ai, bi = a_map.get(key), b_map.get(key)
        merged.append(_merge_dicts(ai, bi) if ai and bi else (ai or bi))
    return merged


def _build_sources(a, b, na, nb):
    s = []
    if a:
        s.append(na)
    if b:
        s.append(nb)
    return s


def merge_results(
    pass_num: int,
    model_a: dict | None,
    model_b: dict | None,
    name_a: str = "gemma",
    name_b: str = "qwen",
) -> dict:
    """Merge two model results for one pass. Single-model: model_b=None → passthrough."""
    a = model_a or {}
    b = model_b or {}

    if not a and not b:
        return {"merged": {}, "consensus": 0.0, "sources": []}

    # For single-model mode, passthrough
    if not b:
        return {"merged": a, "consensus": 1.0, "sources": [name_a]}
    if not a:
        return {"merged": b, "consensus": 1.0, "sources": [name_b]}

    # Generic merge: union lists, prefer longer strings, merge dicts
    merged = {}
    agreement = 0
    total = 0

    for key in set(a) | set(b):
        if key.startswith("_") or key == "page_number":
            continue
        av, bv = a.get(key), b.get(key)

        if isinstance(av, list) and isinstance(bv, list):
            # Merge lists of dicts by first key field, or concatenate
            if av and isinstance(av[0], dict):
                key_field = next(
                    (k for k in ("tag", "ref", "id", "table_id", "material", "service")
                     if k in av[0]),
                    None,
                )
                if key_field:
                    merged[key] = _merge_lists_by_key(av, bv, key_field)
                else:
                    merged[key] = av + bv
            else:
                # Simple list: union
                seen = set()
                combined = []
                for item in av + bv:
                    norm = _normalize(str(item))
                    if norm not in seen:
                        seen.add(norm)
                        combined.append(item)
                merged[key] = combined
        elif isinstance(av, dict) and isinstance(bv, dict):
            merged[key] = _merge_dicts(av, bv)
        elif isinstance(av, bool) or isinstance(bv, bool):
            merged[key] = (av or False) or (bv or False)
        elif av is not None and bv is not None:
            total += 1
            if _similarity(str(av), str(bv)) > 0.7:
                agreement += 1
            merged[key] = av if len(str(av)) >= len(str(bv)) else bv
        else:
            merged[key] = av if av is not None else bv

    consensus = agreement / total if total > 0 else 0.0

    return {
        "merged": merged,
        "consensus": round(consensus, 2),
        "sources": _build_sources(a, b, name_a, name_b),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Master Tracker (from cadAI tracker.py)
# ═══════════════════════════════════════════════════════════════════════════

class MasterTracker:
    """Aggregates extraction results into a master registry."""

    def __init__(self):
        self.items: dict[str, dict[str, dict]] = defaultdict(dict)
        self.metadata = {
            "generated": "",
            "pdfs_processed": 0,
            "total_pages": 0,
            "models_used": [],
            "total_api_calls": 0,
            "total_cost_usd": 0.0,
        }
        self._pdf_set: set[str] = set()
        self._model_set: set[str] = set()
        self._api_calls = 0
        self._total_cost = 0.0

    def record_api_call(self, model: str, cost_usd: float):
        self._api_calls += 1
        self._total_cost += cost_usd
        self._model_set.add(model)

    def ingest_merged(self, pdf_stem: str, pass_num: int, merged: dict):
        self._pdf_set.add(pdf_stem)
        data = merged.get("merged", {})
        consensus = merged.get("consensus", 0.0)
        sources = merged.get("sources", [])

        # Generic ingestion: walk all list/dict fields and upsert items
        for key, value in data.items():
            if key.startswith("_") or not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                # Determine item ID from common key fields
                item_id = None
                for id_field in ("tag", "ref", "id", "table_id", "kks", "material", "standard"):
                    if item.get(id_field):
                        item_id = _tag_key(str(item[id_field]))
                        break
                if not item_id:
                    item_id = f"{pdf_stem}_p{pass_num}_{key}_{len(self.items.get(key, {}))}"

                self._upsert(key, item_id, item, pdf_stem, pass_num, consensus, sources)

        # Handle scalar fields from pass 1 (overview)
        if pass_num == 1:
            doc_data = {k: v for k, v in data.items() if not isinstance(v, list)}
            if doc_data:
                self._upsert("documents", pdf_stem, doc_data, pdf_stem, 1, consensus, sources)

    def _upsert(self, category, item_id, data, pdf, pass_num, consensus, sources):
        if item_id not in self.items[category]:
            self.items[category][item_id] = {
                "data": data,
                "sources": [],
                "consensus_scores": [],
            }
        entry = self.items[category][item_id]
        for k, v in data.items():
            if v is not None and (k not in entry["data"] or entry["data"][k] is None):
                entry["data"][k] = v
        entry["sources"].append({"pdf": pdf, "pass": pass_num, "models": sources, "consensus": consensus})
        entry["consensus_scores"].append(consensus)

    def finalize_metadata(self, total_pages: int = 0):
        self.metadata.update({
            "generated": datetime.now(timezone.utc).isoformat(),
            "pdfs_processed": len(self._pdf_set),
            "total_pages": total_pages,
            "models_used": sorted(self._model_set),
            "total_api_calls": self._api_calls,
            "total_cost_usd": round(self._total_cost, 2),
        })

    def export_json(self, path: Path):
        output = {"metadata": self.metadata, "categories": {}}
        for category, items in sorted(self.items.items()):
            cat_items = []
            for item_id, entry in sorted(items.items()):
                scores = entry["consensus_scores"]
                avg = sum(scores) / len(scores) if scores else 0.0
                n_models = len(set(m for s in entry["sources"] for m in s.get("models", [])))
                cat_items.append({
                    "id": item_id,
                    "status": "confirmed" if n_models >= 2 and avg > 0.5 else "single_source",
                    "consensus": round(avg, 2),
                    "pdfs": sorted(set(s["pdf"] for s in entry["sources"])),
                    "data": entry["data"],
                })
            output["categories"][category] = {"total": len(cat_items), "items": cat_items}

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        total = sum(len(items) for items in self.items.values())
        logger.info("Master tracker: %d items across %d categories → %s",
                     total, len(self.items), path)

    def to_dict(self) -> dict:
        """Return tracker data as a dict (for inline storage)."""
        result = {}
        for category, items in self.items.items():
            result[category] = {}
            for item_id, entry in items.items():
                result[category][item_id] = entry["data"]
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Extraction orchestration (from cadAI extract.py)
# ═══════════════════════════════════════════════════════════════════════════

def _count_items(parsed: dict) -> int:
    count = 0
    for v in parsed.values():
        if isinstance(v, list):
            count += len(v)
        elif isinstance(v, dict):
            count += 1
        elif v is not None and v != "" and v is not False:
            count += 1
    return count


def _combine_pages(result: dict) -> dict:
    """Combine per-page data into a single flat dict."""
    combined: dict = {}
    for page_entry in result.get("pages", []):
        data = page_entry.get("data")
        if not data or not isinstance(data, dict):
            continue
        for key, value in data.items():
            if key.startswith("_") or key == "page_number":
                continue
            if isinstance(value, list):
                combined.setdefault(key, [])
                if isinstance(combined[key], list):
                    combined[key].extend(value)
            elif isinstance(value, dict):
                combined.setdefault(key, {})
                if isinstance(combined[key], dict):
                    combined[key].update(value)
            elif value is not None and key not in combined:
                combined[key] = value
    return combined


def extract_page(client, page_views: PageViews, pass_def: PassDefinition) -> ModelResponse | None:
    if not should_process_page(page_views, pass_def):
        return None
    images = select_images(page_views, pass_def)
    if not images:
        return None
    prompt = pass_def.user_prompt.replace("<int>", str(page_views.page_num))
    return client.analyze(images=images, prompt=prompt, system=pass_def.system_prompt)


def extract_pdf_pass(
    all_views: list[PageViews], client, pass_def: PassDefinition,
) -> dict:
    """Run one pass on all pages of one PDF with one model."""
    pages_data = []
    total_latency = 0.0
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0
    items_found = 0

    for views in all_views:
        resp = extract_page(client, views, pass_def)
        if resp is None:
            continue
        if resp.raw_text.startswith("ERROR:"):
            pages_data.append({"page": views.page_num, "error": resp.raw_text, "data": None})
        else:
            parsed = resp.parsed or {}
            pages_data.append({
                "page": views.page_num, "data": parsed,
                "tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out,
            })
            items_found += _count_items(parsed)

        total_latency += resp.latency_s
        total_tokens_in += resp.tokens_in
        total_tokens_out += resp.tokens_out
        total_cost += resp.cost_usd
        time.sleep(0.3)

    return {
        "model": client.name,
        "pass_num": pass_def.number,
        "pass_name": pass_def.name,
        "pages": pages_data,
        "summary": {
            "pages_processed": len(pages_data),
            "items_found": items_found,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_cost_usd": round(total_cost, 6),
            "total_latency_s": round(total_latency, 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_deep_extraction(
    pdf_path: Path,
    output_dir: Path,
    mode: str = "cloud",
    pass_list: Optional[List[int]] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run full 5-pass deep extraction on a single PDF.

    Args:
        pdf_path: Path to PDF file.
        output_dir: Directory for rendered images and results.
        mode: "cloud" (Groq Maverick) or "edge" (Ollama gemma + qwen).
        pass_list: Run only these passes (1-5). None = all 5.
        resume: Skip if output already exists.

    Returns:
        {
            "tracker": dict,          # Master tracker data (all categories)
            "pass_results": list,      # Raw per-pass merged results
            "metadata": dict,          # Cost, latency, pages processed
            "master_tracker_path": str, # Path to exported JSON (if saved)
        }
    """
    images_dir = output_dir / "images"
    passes = [get_pass(n) for n in pass_list] if pass_list else ALL_PASSES

    # Initialize model clients
    clients = {}
    if mode == "cloud":
        clients["groq"] = GroqVisionClient()
    else:
        try:
            clients["gemma"] = GemmaClient()
        except (ValueError, ImportError) as e:
            logger.warning("Gemma: skipped (%s)", e)
        try:
            clients["qwen"] = QwenVLClient()
        except (ValueError, ImportError) as e:
            logger.warning("Qwen: skipped (%s)", e)

    if not clients:
        raise RuntimeError(
            f"No vision models available for mode={mode}. "
            f"{'Set GROQ_API_KEY' if mode == 'cloud' else 'Run: ollama pull gemma3:4b'}"
        )

    # Render PDF
    logger.info("Rendering %s ...", pdf_path.name)
    all_views = render_pdf(pdf_path, images_dir)
    total_pages = len(all_views)

    # Initialize tracker
    tracker = MasterTracker()
    model_names = list(clients.keys())
    name_a = model_names[0]
    name_b = model_names[1] if len(model_names) > 1 else model_names[0]

    all_pass_results = []

    for pass_def in passes:
        logger.info("Pass %d/%d: %s", pass_def.number, len(passes), pass_def.name)

        results_by_model = {}
        for model_name, client in clients.items():
            result = extract_pdf_pass(all_views, client, pass_def)
            results_by_model[model_name] = result
            cost = result.get("summary", {}).get("total_cost_usd", 0.0)
            tracker.record_api_call(client.MODEL, cost)
            if len(clients) > 1:
                time.sleep(1.0)

        # Merge across models
        a_combined = _combine_pages(results_by_model.get(name_a, {}))
        b_combined = _combine_pages(results_by_model.get(name_b, {})) if name_b != name_a else None
        merged = merge_results(pass_def.number, a_combined, b_combined, name_a, name_b)

        tracker.ingest_merged(pdf_path.stem, pass_def.number, merged)
        all_pass_results.append(merged)

    # Finalize
    tracker.finalize_metadata(total_pages=total_pages)
    tracker_path = output_dir / "master_tracker.json"
    tracker.export_json(tracker_path)

    return {
        "tracker": tracker.to_dict(),
        "pass_results": all_pass_results,
        "metadata": tracker.metadata,
        "master_tracker_path": str(tracker_path),
    }
