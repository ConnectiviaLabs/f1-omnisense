"""CLI orchestrator for multi-model PDF deep extraction.

Supports two modes:
  Edge:  2 local Ollama models (gemma3 + qwen3-vl), free, consensus scoring
  Cloud: 1 Groq cloud model (Llama 4 Maverick), paid, higher quality

Usage:
    # Edge mode (default — local Ollama, 2 models, consensus)
    python -m pipeline
    python -m pipeline --mode edge --pdf "TECHNICAL REGULATIONS"

    # Cloud mode (Groq Maverick, single model)
    python -m pipeline --mode cloud
    python -m pipeline --mode cloud --pdf "race_briefing" --passes "1,3"

    # Common flags (both modes)
    python -m pipeline --resume          # Skip existing results
    python -m pipeline --render-only     # Images only, no model calls
    python -m pipeline --report-only     # Rebuild report from existing
    python -m pipeline --limit 5         # First N PDFs only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .merge import merge_results
from .models import GemmaClient, QwenVLClient, GroqVisionClient, ModelResponse
from .passes import ALL_PASSES, get_pass, select_images, should_process_page
from .renderer import PageViews, discover_pdfs, render_pdf
from .tracker import MasterTracker

# ── Paths ───────────────────────────────────────────────────────────────

PIPELINE_DIR = Path(__file__).parent
OUTPUT_DIR = PIPELINE_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
EXTRACTIONS_DIR = OUTPUT_DIR / "extractions"
MERGED_DIR = OUTPUT_DIR / "merged"


# ── Helpers ─────────────────────────────────────────────────────────────

def _extraction_path(pdf_stem: str, model_name: str, pass_num: int) -> Path:
    """Path for a single extraction result file."""
    d = EXTRACTIONS_DIR / pdf_stem
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{model_name}_pass{pass_num}_{get_pass(pass_num).name}.json"


def _merged_path(pdf_stem: str, pass_num: int) -> Path:
    d = MERGED_DIR / pdf_stem
    d.mkdir(parents=True, exist_ok=True)
    return d / f"pass{pass_num}_{get_pass(pass_num).name}_merged.json"


def _load_existing(path: Path) -> dict | None:
    """Load existing extraction if file exists (for resume)."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _save_result(path: Path, data: dict):
    """Save extraction result to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Per-page extraction ────────────────────────────────────────────────

def extract_page(
    client,
    page_views: PageViews,
    pass_def,
) -> ModelResponse | None:
    """Run one extraction pass on one page with one model.

    Returns ModelResponse or None if page should be skipped.
    """
    if not should_process_page(page_views, pass_def):
        return None

    images = select_images(page_views, pass_def)
    if not images:
        return None

    prompt = pass_def.user_prompt.replace("<int>", str(page_views.page_num))

    return client.analyze(
        images=images,
        prompt=prompt,
        system=pass_def.system_prompt,
    )


# ── Per-PDF extraction ─────────────────────────────────────────────────

def extract_pdf_pass(
    pdf_stem: str,
    all_views: list[PageViews],
    client,
    pass_def,
    resume: bool = False,
) -> dict:
    """Run one pass on all pages of one PDF with one model.

    Returns combined result dict with all pages' data merged.
    """
    model_name = client.name
    out_path = _extraction_path(pdf_stem, model_name, pass_def.number)

    # Resume: skip if already exists
    if resume:
        existing = _load_existing(out_path)
        if existing and existing.get("pages"):
            n_pages = len(existing["pages"])
            print(f"    {model_name}: resumed ({n_pages} pages cached)")
            return existing

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
            print(f"      Page {views.page_num}: {resp.raw_text[:80]}")
            pages_data.append({
                "page": views.page_num,
                "error": resp.raw_text,
                "data": None,
            })
        else:
            parsed = resp.parsed or {}
            pages_data.append({
                "page": views.page_num,
                "data": parsed,
                "tokens_in": resp.tokens_in,
                "tokens_out": resp.tokens_out,
            })
            items_found += _count_items(parsed)

        total_latency += resp.latency_s
        total_tokens_in += resp.tokens_in
        total_tokens_out += resp.tokens_out
        total_cost += resp.cost_usd

        # Small delay between pages (GPU cool for edge, rate limit for cloud)
        time.sleep(0.3)

    result = {
        "pdf": pdf_stem,
        "model": model_name,
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

    _save_result(out_path, result)
    cost_str = f", ${total_cost:.4f}" if total_cost > 0 else ""
    print(f"    {model_name:8s} ... {items_found:3d} items ({total_latency:.1f}s{cost_str})")
    return result


def _count_items(parsed: dict) -> int:
    """Count extractable items in a parsed response."""
    count = 0
    for v in parsed.values():
        if isinstance(v, list):
            count += len(v)
        elif isinstance(v, dict):
            count += 1
        elif v is not None and v != "" and v is not False:
            count += 1
    return count


# ── Merge pass results across models ───────────────────────────────────

def merge_pdf_pass(
    pdf_stem: str,
    pass_num: int,
    model_a_result: dict | None,
    model_b_result: dict | None,
    name_a: str = "gemma",
    name_b: str = "qwen",
) -> dict:
    """Merge two model results for one PDF + pass."""
    model_a_combined = _combine_pages(model_a_result) if model_a_result else None
    model_b_combined = _combine_pages(model_b_result) if model_b_result else None

    merged = merge_results(pass_num, model_a_combined, model_b_combined, name_a, name_b)

    out_path = _merged_path(pdf_stem, pass_num)
    _save_result(out_path, merged)

    consensus = merged.get("consensus", 0.0)
    n_items = _count_items(merged.get("merged", {}))
    sources = merged.get("sources", [])
    source_str = ", ".join(str(s) for s in sources) if sources else "none"
    print(f"    Merged: {n_items} items (consensus: {consensus:.0%}, sources: {source_str})")
    return merged


def _combine_pages(result: dict) -> dict:
    """Combine per-page extraction data into a single flat dict.

    Lists are concatenated, scalars take the first non-null value.
    """
    combined: dict = {}
    for page_entry in result.get("pages", []):
        data = page_entry.get("data")
        if not data or not isinstance(data, dict):
            continue
        for key, value in data.items():
            if key.startswith("_") or key == "page_number":
                continue
            if isinstance(value, list):
                if key not in combined:
                    combined[key] = []
                if isinstance(combined[key], list):
                    combined[key].extend(value)
                else:
                    combined[key] = [combined[key]] + value
            elif isinstance(value, dict):
                if key not in combined:
                    combined[key] = {}
                if isinstance(combined[key], dict):
                    combined[key].update(value)
                else:
                    combined[key] = value
            elif value is not None and key not in combined:
                combined[key] = value
    return combined


# ── Main pipeline ──────────────────────────────────────────────────────

def run_pipeline(
    mode: str = "edge",
    pdf_dir: str | Path | None = None,
    pdf_filter: str | None = None,
    pass_filter: int | None = None,
    pass_list: list[int] | None = None,
    model_filter: str | None = None,
    resume: bool = False,
    render_only: bool = False,
    report_only: bool = False,
    limit: int | None = None,
):
    """Main pipeline execution."""
    print("=" * 70)
    print("  PDF Deep Extraction Pipeline")
    if mode == "cloud":
        print("  Mode: Cloud (Groq Maverick — single model, higher quality)")
    else:
        print("  Mode: Edge (Ollama — 2 local models, consensus scoring)")
    print("=" * 70)

    # ── Report-only mode ────────────────────────────────────────────────
    if report_only:
        print("\n[Report-only mode] Generating from existing extractions...\n")
        _generate_report_from_existing()
        return

    # ── Discover PDFs ───────────────────────────────────────────────────
    if pdf_dir is None:
        print("  ERROR: --pdf-dir is required (directory containing PDFs to process)")
        sys.exit(1)

    pdf_base = Path(pdf_dir)
    if not pdf_base.exists():
        print(f"  ERROR: PDF directory not found: {pdf_base}")
        sys.exit(1)

    pdfs = discover_pdfs(pdf_base)
    if pdf_filter:
        filter_lower = pdf_filter.lower()
        pdfs = [p for p in pdfs if filter_lower in p.stem.lower()]
        if not pdfs:
            print(f"  No PDFs matching '{pdf_filter}'")
            return

    if limit:
        pdfs = pdfs[:limit]

    print(f"\n  Found {len(pdfs)} PDFs to process\n")

    # ── Select passes ──────────────────────────────────────────────────
    if pass_list:
        passes = [get_pass(n) for n in pass_list]
    elif pass_filter:
        passes = [get_pass(pass_filter)]
    else:
        passes = ALL_PASSES
    print(f"  Passes: {', '.join(f'{p.number}:{p.name}' for p in passes)}")

    # ── Initialize models ──────────────────────────────────────────────
    clients = {}

    if mode == "cloud":
        # Cloud mode: single Groq Maverick
        try:
            clients["groq"] = GroqVisionClient()
        except (ValueError, ImportError) as e:
            print(f"\n  ERROR: Groq client failed: {e}")
            print("  Ensure GROQ_API_KEY is set in .env or environment")
            sys.exit(1)
    else:
        # Edge mode: local Ollama models
        if model_filter in (None, "gemma"):
            try:
                clients["gemma"] = GemmaClient()
                print(f"  Gemma: ready ({clients['gemma'].MODEL})")
            except ValueError as e:
                print(f"  Gemma: skipped ({e})")

        if model_filter in (None, "qwen"):
            try:
                clients["qwen"] = QwenVLClient()
                print(f"  Qwen: ready ({clients['qwen'].MODEL})")
            except ValueError as e:
                print(f"  Qwen: skipped ({e})")

    if not clients and not render_only:
        print("\n  ERROR: No models available.")
        if mode == "cloud":
            print("  Check GROQ_API_KEY in environment or .env")
        else:
            print("  Run: ollama pull qwen3.5:9b")
        sys.exit(1)

    # ── Initialize tracker ─────────────────────────────────────────────
    tracker = MasterTracker()
    total_pages = 0

    # ── Process each PDF ───────────────────────────────────────────────
    for pdf_idx, pdf_path in enumerate(pdfs):
        pdf_stem = pdf_path.stem
        print(f"\n[{pdf_idx+1}/{len(pdfs)}] {pdf_stem}")

        # Render images
        print(f"  Rendering pages...")
        all_views = render_pdf(pdf_path, IMAGES_DIR)
        n_pages = len(all_views)
        total_pages += n_pages
        n_zoomed = sum(len(v.zoomed) for v in all_views)
        print(f"  {n_pages} pages rendered ({n_zoomed} zoom regions detected)")

        if render_only:
            continue

        # Get model names for merge
        model_names = list(clients.keys())
        name_a = model_names[0] if len(model_names) > 0 else "gemma"
        name_b = model_names[1] if len(model_names) > 1 else "qwen"

        # Run passes
        for pass_def in passes:
            print(f"  Pass {pass_def.number}/{len(passes)}: {pass_def.name}")

            results_by_model: dict[str, dict] = {}
            for model_name, client in clients.items():
                result = extract_pdf_pass(
                    pdf_stem, all_views, client, pass_def, resume=resume,
                )
                results_by_model[model_name] = result
                cost = result.get("summary", {}).get("total_cost_usd", 0.0)
                tracker.record_api_call(client.MODEL, cost)

                # Delay between models to let GPU swap (edge only)
                if len(clients) > 1:
                    time.sleep(1.0)

            # Merge across models (single-model: model_b_r=None → passthrough)
            model_a_r = results_by_model.get(name_a)
            model_b_r = results_by_model.get(name_b)
            merged = merge_pdf_pass(pdf_stem, pass_def.number,
                                    model_a_r, model_b_r, name_a, name_b)

            # Ingest into tracker
            tracker.ingest_merged(pdf_stem, pass_def.number, merged)

    if render_only:
        print(f"\n  Render complete: {total_pages} pages across {len(pdfs)} PDFs")
        print(f"  Images saved to: {IMAGES_DIR}")
        return

    # ── Export results ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Generating reports...")

    tracker.finalize_metadata(total_pages=total_pages)
    tracker.export_json(OUTPUT_DIR / "master_tracker.json")
    tracker.export_markdown(OUTPUT_DIR / "extraction_report.md")

    total_cost = tracker.metadata.get("total_cost_usd", 0.0)
    print("\n  Done!")
    print(f"  API calls: {tracker.metadata['total_api_calls']}")
    if mode == "cloud":
        print(f"  Cost: ${total_cost:.4f} (Groq API)")
    else:
        print(f"  Cost: $0.00 (local models)")
    print(f"  Results: {OUTPUT_DIR}")


def _generate_report_from_existing():
    """Rebuild tracker + report from existing merged JSON files."""
    tracker = MasterTracker()
    total_pages = 0

    merged_dirs = sorted(MERGED_DIR.iterdir()) if MERGED_DIR.exists() else []
    if not merged_dirs:
        print("  No merged results found. Run extraction first.")
        return

    for pdf_dir in merged_dirs:
        if not pdf_dir.is_dir():
            continue
        pdf_stem = pdf_dir.name
        for merged_file in sorted(pdf_dir.glob("pass*_merged.json")):
            try:
                with open(merged_file) as f:
                    merged = json.load(f)
                pass_num = int(merged_file.stem.split("_")[0].replace("pass", ""))
                tracker.ingest_merged(pdf_stem, pass_num, merged)
            except (json.JSONDecodeError, ValueError, OSError) as e:
                print(f"  Skipping {merged_file.name}: {e}")

    # Count pages from images dir
    if IMAGES_DIR.exists():
        for pdf_dir in IMAGES_DIR.iterdir():
            if pdf_dir.is_dir():
                total_pages += len(list(pdf_dir.glob("page_*_full.png")))

    tracker.finalize_metadata(total_pages=total_pages)
    tracker.export_json(OUTPUT_DIR / "master_tracker.json")
    tracker.export_markdown(OUTPUT_DIR / "extraction_report.md")
    print("  Report generation complete.")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model PDF deep extraction (Edge: Ollama, Cloud: Groq)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pdf-dir", type=str, required=True,
                        help="Directory containing PDFs to process")
    parser.add_argument("--mode", type=str, choices=["edge", "cloud"], default="edge",
                        help="'edge' = local Ollama (2 models, consensus), "
                             "'cloud' = Groq Maverick (single model, higher quality)")
    parser.add_argument("--pdf", type=str, help="Filter: process only PDFs matching this name")
    parser.add_argument("--pass", type=int, dest="pass_num", choices=[1, 2, 3, 4, 5],
                        help="Run only this pass number (1-5)")
    parser.add_argument("--passes", type=str,
                        help="Comma-separated pass numbers to run (e.g. '1,2,4')")
    parser.add_argument("--model", type=str, choices=["gemma", "qwen", "groq"],
                        help="Use only this model (edge: gemma/qwen, cloud: groq)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip existing extraction files")
    parser.add_argument("--render-only", action="store_true",
                        help="Only render PDF images, no model calls")
    parser.add_argument("--report-only", action="store_true",
                        help="Generate report from existing extractions")
    parser.add_argument("--limit", type=int,
                        help="Process only first N PDFs")
    args = parser.parse_args()

    # --passes "1,2,4" overrides --pass
    pass_list = None
    if args.passes:
        pass_list = [int(x.strip()) for x in args.passes.split(",")]
    elif args.pass_num:
        pass_list = [args.pass_num]

    run_pipeline(
        mode=args.mode,
        pdf_dir=args.pdf_dir,
        pdf_filter=args.pdf,
        pass_filter=args.pass_num,
        pass_list=pass_list,
        model_filter=args.model,
        resume=args.resume,
        render_only=args.render_only,
        report_only=args.report_only,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
