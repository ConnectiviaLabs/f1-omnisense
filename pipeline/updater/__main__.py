"""CLI entry for the updater module.

Usage:
    python -m pipeline.updater                     # auto-sync current year
    python -m pipeline.updater --year 2026         # sync specific year
    python -m pipeline.updater --full-refresh      # re-fetch everything
    python -m pipeline.updater --status            # show data state
    python -m pipeline.updater --serve             # start API on port 8103
"""

import argparse
import json
import sys
from datetime import datetime

from .updater import LiveUpdater


def _json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    parser = argparse.ArgumentParser(description="F1 Live Update Pipeline")
    parser.add_argument("--year", type=int, help="Year to sync (default: current)")
    parser.add_argument("--full-refresh", action="store_true", help="Re-fetch all data")
    parser.add_argument("--skip-fastf1", action="store_true", help="Skip FastF1 fetch")
    parser.add_argument("--skip-profiles", action="store_true", help="Skip profile recompute")
    parser.add_argument("--status", action="store_true", help="Show current data state")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        from .server import router
        from fastapi import FastAPI

        app = FastAPI(title="F1 Updater API")
        app.include_router(router)
        uvicorn.run(app, host="0.0.0.0", port=8103)
        return

    updater = LiveUpdater()

    if args.status:
        status = updater.status()
        print(json.dumps(status, indent=2, default=_json_serial))
        return

    summary = updater.sync(
        year=args.year,
        full_refresh=args.full_refresh,
        skip_fastf1=args.skip_fastf1,
        skip_profiles=args.skip_profiles,
    )


if __name__ == "__main__":
    main()
