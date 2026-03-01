"""OpponentProfiler — query API for the Chunk F opponent data.

Usage:
    from pipeline.opponents import OpponentProfiler

    op = OpponentProfiler()
    op.get_driver("hamilton")
    op.compare_drivers(["hamilton", "max_verstappen", "norris"])
    op.get_leaderboard("undercut_aggression_score", top_n=5)
"""

from __future__ import annotations

from pymongo.database import Database

from ._db import get_db, COL_PIT_LOSS, COL_PROFILES, COL_CIRCUIT, COL_COMPOUND

NO_ID = {"_id": 0}

# Fields shown in the compact driver list
_SUMMARY_FIELDS = {
    "_id": 0,
    "driver_id": 1,
    "driver_code": 1,
    "forename": 1,
    "surname": 1,
    "nationality": 1,
    "total_races": 1,
    "total_wins": 1,
    "total_podiums": 1,
    "seasons": 1,
}


class OpponentProfiler:
    """Read-only query interface over the 4 opponent collections."""

    def __init__(self, db: Database | None = None):
        self.db = db or get_db()

    # ── Single-driver queries ──────────────────────────────────────────

    def get_driver(self, driver_id: str) -> dict | None:
        """Full opponent profile for one driver."""
        return self.db[COL_PROFILES].find_one(
            {"driver_id": driver_id}, NO_ID
        )

    def get_driver_circuits(
        self,
        driver_id: str,
        circuit: str | None = None,
    ) -> list[dict]:
        """Circuit-specific profiles for a driver."""
        filt: dict = {"driver_id": driver_id}
        if circuit:
            filt["circuit"] = circuit
        return list(self.db[COL_CIRCUIT].find(filt, NO_ID))

    def get_driver_compounds(self, driver_id: str) -> list[dict]:
        """Tyre compound profiles for a driver."""
        return list(self.db[COL_COMPOUND].find({"driver_id": driver_id}, NO_ID))

    # ── Multi-driver queries ───────────────────────────────────────────

    def get_all_drivers(self) -> list[dict]:
        """Summary list of all profiled drivers."""
        return list(
            self.db[COL_PROFILES]
            .find({}, _SUMMARY_FIELDS)
            .sort("driver_id", 1)
        )

    def compare_drivers(
        self,
        driver_ids: list[str],
        metrics: list[str] | None = None,
    ) -> list[dict]:
        """Side-by-side profiles for multiple drivers.

        If metrics is given, only those fields (plus driver_id) are returned.
        """
        filt = {"driver_id": {"$in": driver_ids}}
        if metrics:
            proj = {"_id": 0, "driver_id": 1, "driver_code": 1,
                    "forename": 1, "surname": 1}
            for m in metrics:
                proj[m] = 1
        else:
            proj = NO_ID
        return list(self.db[COL_PROFILES].find(filt, proj))

    def search_drivers(self, **filters) -> list[dict]:
        """Filter drivers by field criteria.

        Examples:
            search_drivers(nationality="British")
            search_drivers(total_races={"$gte": 100})
        """
        return list(self.db[COL_PROFILES].find(filters, NO_ID))

    # ── Leaderboard / ranking ──────────────────────────────────────────

    def get_leaderboard(
        self,
        metric: str,
        top_n: int = 10,
        ascending: bool = False,
    ) -> list[dict]:
        """Top N drivers by any numeric metric.

        Returns [{driver_id, driver_code, forename, surname, <metric>}].
        """
        direction = 1 if ascending else -1
        proj = {
            "_id": 0,
            "driver_id": 1,
            "driver_code": 1,
            "forename": 1,
            "surname": 1,
            metric: 1,
        }
        return list(
            self.db[COL_PROFILES]
            .find({metric: {"$exists": True, "$ne": None}}, proj)
            .sort(metric, direction)
            .limit(top_n)
        )

    # ── Circuit queries ────────────────────────────────────────────────

    def get_pit_loss(self, circuit: str) -> dict | None:
        """Pit stop loss time stats for a circuit."""
        return self.db[COL_PIT_LOSS].find_one({"circuit": circuit}, NO_ID)

    def get_all_pit_losses(self) -> list[dict]:
        """All circuits sorted by median pit loss ascending."""
        return list(
            self.db[COL_PIT_LOSS]
            .find({}, NO_ID)
            .sort("median_total_pit_s", 1)
        )

    def get_circuit_overview(self, circuit: str) -> dict:
        """Combined pit loss + all driver profiles for a circuit."""
        pit = self.get_pit_loss(circuit)
        drivers = list(self.db[COL_CIRCUIT].find({"circuit": circuit}, NO_ID))
        return {
            "circuit": circuit,
            "pit_loss": pit,
            "drivers": drivers,
            "driver_count": len(drivers),
        }

    def get_circuit_drivers(self, circuit: str) -> list[dict]:
        """All drivers who have raced at a circuit."""
        return list(
            self.db[COL_CIRCUIT]
            .find({"circuit": circuit}, NO_ID)
            .sort("avg_finish_position", 1)
        )

    # ── Compound queries ───────────────────────────────────────────────

    def get_compound_rankings(
        self,
        compound: str,
        metric: str = "degradation_slope",
        top_n: int = 10,
    ) -> list[dict]:
        """Rank drivers by a metric on a specific compound."""
        return list(
            self.db[COL_COMPOUND]
            .find({"compound": compound}, NO_ID)
            .sort(metric, 1)
            .limit(top_n)
        )

    # ── Stats / meta ──────────────────────────────────────────────────

    def stats(self) -> dict:
        """Collection doc counts and available metrics."""
        profile_sample = self.db[COL_PROFILES].find_one({}, NO_ID)
        metrics = sorted(profile_sample.keys()) if profile_sample else []
        return {
            "profiles": self.db[COL_PROFILES].estimated_document_count(),
            "circuit_profiles": self.db[COL_CIRCUIT].estimated_document_count(),
            "compound_profiles": self.db[COL_COMPOUND].estimated_document_count(),
            "pit_loss_circuits": self.db[COL_PIT_LOSS].estimated_document_count(),
            "available_metrics": metrics,
        }
