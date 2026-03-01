"""F1 Opponent Profiling module.

Query and serve the Chunk F opponent data from marip_f1 MongoDB.

Public API:
    from pipeline.opponents import OpponentProfiler
    from pipeline.opponents import router  # FastAPI router

Usage:
    op = OpponentProfiler()
    op.get_driver("hamilton")
    op.compare_drivers(["hamilton", "max_verstappen", "norris"])
    op.get_leaderboard("undercut_aggression_score", top_n=5)
"""

from .profiler import OpponentProfiler
from .server import router

__all__ = ["OpponentProfiler", "router"]
