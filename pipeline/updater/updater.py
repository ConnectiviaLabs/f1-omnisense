"""LiveUpdater — orchestrator for the full data refresh pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pymongo.database import Database

from ._db import get_db
from . import openf1_fetcher, jolpica_fetcher, fastf1_fetcher, profile_refresher

logger = logging.getLogger(__name__)


class LiveUpdater:
    """Orchestrates the full update pipeline.

    Usage:
        from pipeline.updater import LiveUpdater
        updater = LiveUpdater()
        updater.sync()           # auto-detect gaps and sync
        updater.sync(year=2026)  # sync specific year
    """

    def __init__(self, db: Database | None = None):
        self.db = db or get_db()

    def sync(
        self,
        year: int | None = None,
        full_refresh: bool = False,
        skip_fastf1: bool = False,
        skip_profiles: bool = False,
    ) -> dict:
        """Run the full update pipeline.

        Steps:
          1. OpenF1 API → openf1_* collections
          2. Jolpica API → patch career stats in opponent_profiles
          3. FastF1 lib → fastf1_laps, fastf1_weather (if installed)
          4. Recompute derived profiles

        Args:
            year: Year to sync. None = current year.
            full_refresh: Re-fetch even if data exists.
            skip_fastf1: Skip FastF1 fetch (useful if not installed).
            skip_profiles: Skip profile recomputation.

        Returns:
            Summary dict of all operations.
        """
        if year is None:
            year = datetime.now().year

        print("=" * 60)
        print(f"  F1 Live Update Pipeline — {year}")
        print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
        print("=" * 60)

        summary = {"year": year, "started_at": datetime.now(timezone.utc).isoformat()}

        # Step 1: OpenF1
        try:
            openf1_results = openf1_fetcher.sync(self.db, year, full_refresh)
            summary["openf1"] = openf1_results
        except Exception as e:
            logger.error("OpenF1 sync failed: %s", e)
            summary["openf1"] = {"error": str(e)}

        # Step 2: Jolpica career stats
        try:
            jolpica_count = jolpica_fetcher.sync(self.db, years=list(range(2018, year + 1)))
            summary["jolpica"] = {"profiles_patched": jolpica_count}
        except Exception as e:
            logger.error("Jolpica sync failed: %s", e)
            summary["jolpica"] = {"error": str(e)}

        # Step 3: FastF1 (optional)
        if not skip_fastf1:
            try:
                ff1_results = fastf1_fetcher.sync(self.db, year)
                summary["fastf1"] = ff1_results
            except Exception as e:
                logger.error("FastF1 sync failed: %s", e)
                summary["fastf1"] = {"error": str(e)}
        else:
            summary["fastf1"] = {"skipped": True}

        # Step 4: Refresh profiles
        if not skip_profiles:
            try:
                profile_results = profile_refresher.refresh(self.db)
                summary["profiles"] = profile_results
            except Exception as e:
                logger.error("Profile refresh failed: %s", e)
                summary["profiles"] = {"error": str(e)}
        else:
            summary["profiles"] = {"skipped": True}

        # Log
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.db["pipeline_log"].insert_one({
            "chunk": "live_update",
            "status": "complete",
            "year": year,
            "summary": summary,
            "timestamp": datetime.now(timezone.utc),
        })

        print("\n" + "=" * 60)
        print("  Pipeline complete.")
        print("=" * 60)
        return summary

    def status(self) -> dict:
        """Report current data state and gaps."""
        db = self.db
        info = {}

        # Collection counts
        collections = [
            "openf1_sessions", "openf1_laps", "openf1_intervals",
            "openf1_position", "openf1_stints", "openf1_pit",
            "openf1_race_control", "openf1_weather", "openf1_drivers",
            "fastf1_laps", "fastf1_weather", "telemetry_compressed",
            "opponent_profiles", "opponent_circuit_profiles",
            "opponent_compound_profiles", "circuit_pit_loss_times",
        ]
        info["collections"] = {}
        for col in collections:
            info["collections"][col] = db[col].estimated_document_count()

        # Year coverage
        openf1_years = sorted(db["openf1_sessions"].distinct("year"))
        ff1_years = sorted(db["fastf1_laps"].distinct("Year"))
        info["openf1_years"] = openf1_years
        info["fastf1_years"] = ff1_years

        # Latest session
        latest = db["openf1_sessions"].find_one(sort=[("date_start", -1)])
        if latest:
            info["latest_session"] = {
                "name": latest.get("session_name"),
                "circuit": latest.get("circuit_short_name"),
                "date": str(latest.get("date_start", ""))[:10],
                "year": latest.get("year"),
            }

        # Last pipeline run
        last_run = db["pipeline_log"].find_one(
            {"chunk": "live_update"}, sort=[("timestamp", -1)]
        )
        if last_run:
            info["last_sync"] = str(last_run.get("timestamp", ""))

        # Gaps
        current_year = datetime.now().year
        info["gaps"] = {
            "openf1_missing_years": [y for y in range(2023, current_year + 1) if y not in openf1_years],
            "fastf1_missing_years": [y for y in range(2018, current_year + 1) if y not in ff1_years],
        }

        return info
