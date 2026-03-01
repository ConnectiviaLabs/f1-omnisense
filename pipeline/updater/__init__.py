"""F1 Live Update Pipeline.

Fetches new data from OpenF1 + Jolpica APIs, optionally FastF1,
upserts into marip_f1 MongoDB, and recomputes opponent profiles.

Public API:
    from pipeline.updater import LiveUpdater
    from pipeline.updater import router  # FastAPI router

Usage:
    updater = LiveUpdater()
    updater.sync()            # auto-detect and sync current year
    updater.sync(year=2026)   # sync specific year
    updater.status()          # show data state and gaps
"""

from .updater import LiveUpdater
from .server import router

__all__ = ["LiveUpdater", "router"]
