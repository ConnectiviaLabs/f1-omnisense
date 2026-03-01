"""CLI entry for the opponents module.

Usage:
    python -m pipeline.opponents                           # show stats
    python -m pipeline.opponents --driver hamilton         # full profile
    python -m pipeline.opponents --driver hamilton --circuits  # circuit breakdowns
    python -m pipeline.opponents --driver hamilton --compounds # tyre profiles
    python -m pipeline.opponents --compare hamilton,max_verstappen,norris
    python -m pipeline.opponents --leaderboard undercut_aggression_score
    python -m pipeline.opponents --serve                   # start API server
"""

import argparse
import json
import sys
from datetime import datetime

from .profiler import OpponentProfiler


def _json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _pp(data):
    print(json.dumps(data, indent=2, default=_json_serial))


def main():
    parser = argparse.ArgumentParser(description="F1 Opponent Profiles")
    parser.add_argument("--driver", type=str, help="Driver ID (e.g. hamilton)")
    parser.add_argument("--circuits", action="store_true", help="Show circuit profiles")
    parser.add_argument("--compounds", action="store_true", help="Show compound profiles")
    parser.add_argument("--compare", type=str, help="Comma-separated driver IDs")
    parser.add_argument("--leaderboard", type=str, help="Metric name to rank by")
    parser.add_argument("--top", type=int, default=10, help="Top N for leaderboard")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        from .server import router
        from fastapi import FastAPI

        app = FastAPI(title="F1 Opponent Profiles API")
        app.include_router(router)
        uvicorn.run(app, host="0.0.0.0", port=8102)
        return

    op = OpponentProfiler()

    if args.compare:
        ids = [d.strip() for d in args.compare.split(",")]
        _pp(op.compare_drivers(ids))
    elif args.leaderboard:
        _pp(op.get_leaderboard(args.leaderboard, args.top))
    elif args.driver:
        if args.circuits:
            _pp(op.get_driver_circuits(args.driver))
        elif args.compounds:
            _pp(op.get_driver_compounds(args.driver))
        else:
            doc = op.get_driver(args.driver)
            if doc is None:
                print(f"No profile for '{args.driver}'")
                sys.exit(1)
            _pp(doc)
    else:
        _pp(op.stats())


if __name__ == "__main__":
    main()
