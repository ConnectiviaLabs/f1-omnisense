"""
Enrichment Runner
─────────────────
Runs all enrichment fetchers in sequence. Each is idempotent —
only new data since the last run is fetched.

Usage:
    python -m pipeline.enrichment.run_all
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from updater._db import get_db


STEPS = [
    ("Circuit Intelligence", "enrichment.fetch_circuits"),
    ("Overtake Data", "enrichment.fetch_overtakes"),
    ("Air Density", "enrichment.fetch_air_density"),
    ("Jolpica Full Grid", "enrichment.fetch_jolpica_full"),
    ("Driver Performance Markers", "driver_performance_markers"),
]


def main():
    start = time.time()
    db = get_db()
    print("=" * 60)
    print("  Enrichment Runner — Incremental Update")
    print("=" * 60)

    results = {}

    for i, (label, module_name) in enumerate(STEPS, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(STEPS)}] {label}")
        print(f"{'─' * 60}")

        try:
            mod = __import__(module_name, fromlist=["main"])
            mod.main()
            results[label] = "OK"
        except Exception as e:
            print(f"  !! {label} failed: {e}")
            results[label] = f"FAILED: {e}"

    elapsed = time.time() - start

    # Final inventory
    print(f"\n{'=' * 60}")
    print("  Final Collection Inventory")
    print(f"{'=' * 60}")
    collections = sorted(db.list_collection_names())
    total_docs = 0
    for name in collections:
        count = db[name].estimated_document_count()
        total_docs += count
        print(f"  {name:<40} {count:>10,}")
    print(f"  {'TOTAL':<40} {total_docs:>10,}")

    print(f"\n{'=' * 60}")
    print("  Step Results")
    print(f"{'=' * 60}")
    for label, status in results.items():
        icon = "OK" if status == "OK" else "FAIL"
        print(f"  [{icon}] {label}")

    print(f"\n  Completed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
