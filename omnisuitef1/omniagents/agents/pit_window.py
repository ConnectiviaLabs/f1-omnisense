"""Agent 03 — PitWindow Agent.

Monitors tire age, lap deltas, and gaps to calculate optimal pit windows.
Reacts to weather changes from Agent 02.

SUB: f1:weather:update
PUB: f1:pit:window:open, f1:strategy:recommend
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from omniagents._types import EventSeverity
from omniagents.base import F1Agent

logger = logging.getLogger(__name__)

# Typical F1 pit stop loss (seconds) — varies by circuit
DEFAULT_PIT_LOSS = 22.0


class PitWindowAgent(F1Agent):
    name = "pit_window"
    description = (
        "Calculates optimal pit stop windows using tire degradation rates, "
        "gap analysis, and undercut/overcut viability. Reacts to weather changes "
        "for compound switch decisions."
    )
    subscriptions = ["f1:weather:update"]
    publications = ["f1:pit:window:open", "f1:strategy:recommend"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weather_context: Dict[int, Dict] = {}  # session_key → last weather event

    async def on_event(self, topic: str, event: Dict[str, Any]):
        """React to weather changes — cache context for pit window recalculation."""
        if topic == "f1:weather:update":
            sk = event.get("payload", {}).get("session_key")
            if sk:
                self._weather_context[sk] = event.get("payload", {})
                logger.info("[pit_window] Weather context updated for session %s", sk)

    async def run_analysis(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate pit windows for a session."""

        # 1. Load stints, laps, intervals
        stints_df, laps_df, intervals_df = await asyncio.gather(
            asyncio.to_thread(self._load_stints, session_key),
            asyncio.to_thread(self._load_laps, session_key, driver_number),
            asyncio.to_thread(self._load_intervals, session_key, driver_number),
        )

        if laps_df.empty:
            return {"status": "no_data", "session_key": session_key}

        # 2. Calculate degradation per stint
        stint_analysis = self._analyze_stints(laps_df, stints_df)

        # 3. Calculate gap-based pit windows
        pit_windows = self._calculate_pit_windows(stint_analysis, intervals_df, driver_number)

        # 4. Factor in weather context
        weather = self._weather_context.get(session_key, {})
        if weather.get("rain_detected"):
            for pw in pit_windows:
                pw["weather_factor"] = "rain_incoming"
                pw["urgency"] = "immediate"

        # 5. LLM reasoning
        insight = None
        if pit_windows:
            insight = await self.reason(
                f"Analyze pit window strategy for session {session_key}"
                f"{f' driver #{driver_number}' if driver_number else ''}. "
                "Consider tire degradation rates, gap data, undercut threats, and any weather factors. "
                "Recommend optimal pit lap and compound choice.",
                data_context={
                    "pit_windows": pit_windows[:5],
                    "stint_analysis": stint_analysis[:5],
                    "weather": weather if weather else "dry conditions",
                },
            )

        # 6. Publish
        severity = EventSeverity.HIGH if any(pw.get("urgency") == "immediate" for pw in pit_windows) else EventSeverity.MEDIUM

        await self.publish(
            topic="f1:pit:window:open",
            payload={
                "session_key": session_key,
                "driver_number": driver_number,
                "pit_windows": pit_windows,
                "summary": insight or f"{len(pit_windows)} pit windows identified",
            },
            severity=severity,
            session_key=session_key,
            driver_number=driver_number,
        )

        if pit_windows:
            await self.publish(
                topic="f1:strategy:recommend",
                payload={
                    "source": "pit_window",
                    "session_key": session_key,
                    "driver_number": driver_number,
                    "pit_windows": pit_windows,
                    "recommendation": insight or "Pit windows calculated",
                },
                severity=severity,
                session_key=session_key,
                driver_number=driver_number,
            )

        output = {
            "session_key": session_key,
            "driver_number": driver_number,
            "stint_analysis": stint_analysis,
            "pit_windows": pit_windows,
            "insight": insight,
        }
        await self.save_output("agent_pit_windows", output)
        return {"summary": insight or f"{len(pit_windows)} pit windows", **output}

    # ── Data loading ────────────────────────────────────────────────────────

    def _load_stints(self, session_key: int) -> pd.DataFrame:
        if self._db is None:
            return pd.DataFrame()
        docs = list(self._db["openf1_stints"].find(
            {"session_key": session_key}, {"_id": 0}
        ))
        return pd.DataFrame(docs) if docs else pd.DataFrame()

    def _load_laps(self, session_key: int, driver_number: Optional[int]) -> pd.DataFrame:
        if self._db is None:
            return pd.DataFrame()
        query: Dict[str, Any] = {"session_key": session_key}
        if driver_number:
            query["driver_number"] = driver_number
        docs = list(self._db["openf1_laps"].find(query, {"_id": 0}).sort("lap_number", 1))
        return pd.DataFrame(docs) if docs else pd.DataFrame()

    def _load_intervals(self, session_key: int, driver_number: Optional[int]) -> pd.DataFrame:
        if self._db is None:
            return pd.DataFrame()
        query: Dict[str, Any] = {"session_key": session_key}
        if driver_number:
            query["driver_number"] = driver_number
        docs = list(self._db["openf1_intervals"].find(query, {"_id": 0}).sort("date", 1))
        return pd.DataFrame(docs) if docs else pd.DataFrame()

    # ── Analysis ────────────────────────────────────────────────────────────

    def _analyze_stints(self, laps_df: pd.DataFrame, stints_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze degradation per stint."""
        stints: List[Dict[str, Any]] = []

        if stints_df.empty or "lap_start" not in stints_df.columns:
            # Infer stints from lap data
            if "lap_duration" in laps_df.columns and "lap_number" in laps_df.columns:
                durations = laps_df["lap_duration"].dropna()
                if len(durations) > 5:
                    deg_rate = self._calc_degradation_rate(durations.values)
                    stints.append({
                        "stint": 1,
                        "compound": "unknown",
                        "laps": len(durations),
                        "degradation_rate_ms_per_lap": round(deg_rate * 1000, 1),
                        "avg_lap_time": round(float(durations.mean()), 3),
                    })
            return stints

        drivers = stints_df["driver_number"].unique() if "driver_number" in stints_df.columns else [None]

        for drv in drivers:
            drv_stints = stints_df[stints_df["driver_number"] == drv] if drv else stints_df
            for _, stint_row in drv_stints.iterrows():
                lap_start = stint_row.get("lap_start", 0)
                lap_end = stint_row.get("lap_end", lap_start + 20)
                compound = stint_row.get("compound", "unknown")

                stint_laps = laps_df[
                    (laps_df["lap_number"] >= lap_start) & (laps_df["lap_number"] <= lap_end)
                ]
                if "driver_number" in laps_df.columns and drv:
                    stint_laps = stint_laps[stint_laps["driver_number"] == drv]

                if "lap_duration" in stint_laps.columns and len(stint_laps) > 2:
                    durations = stint_laps["lap_duration"].dropna().values
                    deg_rate = self._calc_degradation_rate(durations)
                    stints.append({
                        "driver_number": drv,
                        "stint": int(stint_row.get("stint_number", 0)),
                        "compound": str(compound).upper() if compound else "UNKNOWN",
                        "lap_start": int(lap_start),
                        "lap_end": int(lap_end),
                        "laps": len(stint_laps),
                        "degradation_rate_ms_per_lap": round(deg_rate * 1000, 1),
                        "avg_lap_time": round(float(stint_laps["lap_duration"].mean()), 3),
                    })
        return stints

    @staticmethod
    def _calc_degradation_rate(durations: np.ndarray) -> float:
        """Linear degradation rate (seconds per lap)."""
        if len(durations) < 3:
            return 0.0
        x = np.arange(len(durations))
        coeffs = np.polyfit(x, durations, 1)
        return float(coeffs[0])

    def _calculate_pit_windows(
        self,
        stint_analysis: List[Dict],
        intervals_df: pd.DataFrame,
        driver_number: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Calculate pit windows based on degradation and gaps."""
        windows: List[Dict[str, Any]] = []

        for stint in stint_analysis:
            deg_rate = stint.get("degradation_rate_ms_per_lap", 0) / 1000
            laps_done = stint.get("laps", 0)
            compound = stint.get("compound", "UNKNOWN")

            if deg_rate <= 0:
                continue

            # Estimate remaining useful laps (when degradation exceeds pit loss benefit)
            if deg_rate > 0:
                remaining_useful = int(DEFAULT_PIT_LOSS / deg_rate) if deg_rate > 0.1 else 99
            else:
                remaining_useful = 99

            optimal_pit_lap = stint.get("lap_start", 0) + laps_done + max(0, remaining_useful - 3)

            # Gap analysis for undercut threat
            undercut_viable = False
            gap_behind = None
            if not intervals_df.empty and "gap_to_leader" in intervals_df.columns:
                last_intervals = intervals_df.tail(5)
                if not last_intervals.empty:
                    gap_behind = float(last_intervals["gap_to_leader"].median())
                    undercut_viable = gap_behind is not None and gap_behind < DEFAULT_PIT_LOSS * 0.15

            windows.append({
                "driver_number": stint.get("driver_number", driver_number),
                "current_compound": compound,
                "stint_laps": laps_done,
                "degradation_sec_per_lap": round(deg_rate, 3),
                "optimal_pit_lap": optimal_pit_lap,
                "remaining_useful_laps": remaining_useful,
                "undercut_viable": undercut_viable,
                "gap_behind_sec": round(gap_behind, 2) if gap_behind else None,
                "pit_loss_estimate_sec": DEFAULT_PIT_LOSS,
                "urgency": "immediate" if remaining_useful <= 3 else "soon" if remaining_useful <= 6 else "monitoring",
            })

        windows.sort(key=lambda w: w.get("remaining_useful_laps", 99))
        return windows
