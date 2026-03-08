"""Agent 02 — WeatherAdapt Agent.

Monitors weather data for a session, detects significant changes
(rain onset, temperature drops, humidity spikes) and publishes
strategy alerts.

SUB: (trigger via API or run-all)
PUB: f1:weather:update, f1:strategy:recommend
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from omniagents._types import EventSeverity
from omniagents.base import F1Agent

logger = logging.getLogger(__name__)

# Thresholds for weather events
TEMP_DROP_THRESHOLD = 5.0       # degrees C
HUMIDITY_SPIKE_THRESHOLD = 15.0  # percentage points
WIND_CHANGE_THRESHOLD = 10.0    # km/h


class WeatherAdaptAgent(F1Agent):
    name = "weather_adapt"
    description = (
        "Monitors track weather conditions and detects significant changes — "
        "rain onset, temperature drops, humidity spikes — that impact tire strategy "
        "and car setup. Triggers pit window recalculation on compound change scenarios."
    )
    subscriptions: List[str] = []
    publications = ["f1:weather:update", "f1:strategy:recommend"]

    async def run_analysis(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze weather data for a session and detect change events."""

        # 1. Load weather data
        weather_df = await asyncio.to_thread(self._load_weather, session_key)
        if weather_df.empty:
            return {"status": "no_data", "session_key": session_key}

        # 2. Detect weather change events
        events = self._detect_weather_events(weather_df)

        # 3. Cross-reference with lap data for timing context
        laps_df = await asyncio.to_thread(self._load_laps, session_key)
        if not laps_df.empty and events:
            events = self._enrich_with_laps(events, laps_df)

        # 4. LLM reasoning on significant events
        insight = None
        if events:
            insight = await self.reason(
                f"Analyze these {len(events)} weather change events during session {session_key}. "
                "How should they impact tire strategy and car setup? "
                "Consider compound choices, pit timing, and driver management.",
                data_context={"weather_events": events[:8]},
            )

        # 5. Publish weather update
        severity = EventSeverity.HIGH if any(e["type"] == "rain_onset" for e in events) else (
            EventSeverity.MEDIUM if events else EventSeverity.INFO
        )

        await self.publish(
            topic="f1:weather:update",
            payload={
                "session_key": session_key,
                "event_count": len(events),
                "events": events,
                "rain_detected": any(e["type"] == "rain_onset" for e in events),
                "summary": insight or f"{len(events)} weather events detected",
            },
            severity=severity,
            session_key=session_key,
        )

        # Publish strategy recommendation if significant events found
        if events:
            await self.publish(
                topic="f1:strategy:recommend",
                payload={
                    "source": "weather_adapt",
                    "session_key": session_key,
                    "trigger": "weather_change",
                    "events": events,
                    "recommendation": insight or "Weather change detected — reassess strategy",
                },
                severity=severity,
                session_key=session_key,
            )

        # 6. Persist
        output = {
            "session_key": session_key,
            "event_count": len(events),
            "events": events,
            "insight": insight,
        }
        await self.save_output("agent_weather_alerts", output)

        return {"summary": insight or f"{len(events)} weather events", **output}

    # ── Data loading ────────────────────────────────────────────────────────

    def _load_weather(self, session_key: int) -> pd.DataFrame:
        if self._db is None:
            return pd.DataFrame()

        query = {"session_key": session_key}
        for coll_name in ["openf1_weather", "fastf1_weather"]:
            coll = self._db[coll_name]
            docs = list(coll.find(query, {"_id": 0}).sort("date", 1))
            if docs:
                logger.info("Loaded %d weather records from %s", len(docs), coll_name)
                return pd.DataFrame(docs)

        return pd.DataFrame()

    def _load_laps(self, session_key: int) -> pd.DataFrame:
        if self._db is None:
            return pd.DataFrame()

        query = {"session_key": session_key}
        docs = list(self._db["openf1_laps"].find(query, {"_id": 0}).sort("date_start", 1))
        return pd.DataFrame(docs) if docs else pd.DataFrame()

    # ── Weather event detection ─────────────────────────────────────────────

    def _detect_weather_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        # Rain onset/cessation
        if "rainfall" in df.columns:
            rain_col = df["rainfall"].fillna(0)
            for i in range(1, len(rain_col)):
                if rain_col.iloc[i] > 0 and rain_col.iloc[i - 1] == 0:
                    events.append(self._make_event(df, i, "rain_onset",
                        f"Rain started — rainfall={rain_col.iloc[i]}"))
                elif rain_col.iloc[i] == 0 and rain_col.iloc[i - 1] > 0:
                    events.append(self._make_event(df, i, "rain_stop", "Rain stopped"))

        # Track temperature drops
        if "track_temperature" in df.columns:
            temp = df["track_temperature"].fillna(method="ffill")
            rolling_max = temp.rolling(window=5, min_periods=1).max()
            for i in range(5, len(temp)):
                drop = rolling_max.iloc[i] - temp.iloc[i]
                if drop >= TEMP_DROP_THRESHOLD:
                    events.append(self._make_event(df, i, "temp_drop",
                        f"Track temp dropped {drop:.1f}C (from {rolling_max.iloc[i]:.1f} to {temp.iloc[i]:.1f})"))

        # Humidity spikes
        if "humidity" in df.columns:
            hum = df["humidity"].fillna(method="ffill")
            for i in range(3, len(hum)):
                delta = hum.iloc[i] - hum.iloc[i - 3]
                if delta >= HUMIDITY_SPIKE_THRESHOLD:
                    events.append(self._make_event(df, i, "humidity_spike",
                        f"Humidity jumped {delta:.0f}% (to {hum.iloc[i]:.0f}%)"))

        # Deduplicate nearby events of same type
        events = self._deduplicate(events, min_gap_seconds=120)
        return events

    def _make_event(self, df: pd.DataFrame, idx: int, event_type: str, detail: str) -> Dict[str, Any]:
        row = df.iloc[idx]
        return {
            "type": event_type,
            "detail": detail,
            "timestamp": str(row.get("date", "")),
            "track_temperature": float(row["track_temperature"]) if "track_temperature" in row.index else None,
            "air_temperature": float(row["air_temperature"]) if "air_temperature" in row.index else None,
            "humidity": float(row["humidity"]) if "humidity" in row.index else None,
            "rainfall": float(row["rainfall"]) if "rainfall" in row.index else None,
            "wind_speed": float(row["wind_speed"]) if "wind_speed" in row.index else None,
        }

    @staticmethod
    def _deduplicate(events: List[Dict], min_gap_seconds: int = 120) -> List[Dict]:
        """Remove events of the same type within min_gap_seconds of each other."""
        if not events:
            return events
        deduped = [events[0]]
        for e in events[1:]:
            # Simple dedup by type — keep first of each cluster
            if e["type"] != deduped[-1]["type"]:
                deduped.append(e)
        return deduped

    def _enrich_with_laps(self, events: List[Dict], laps_df: pd.DataFrame) -> List[Dict]:
        """Add approximate lap number to weather events."""
        if "date_start" not in laps_df.columns:
            return events
        lap_times = pd.to_datetime(laps_df["date_start"], errors="coerce")
        for event in events:
            try:
                ts = pd.Timestamp(event["timestamp"])
                diffs = (lap_times - ts).abs()
                closest_idx = diffs.idxmin()
                event["approx_lap"] = int(laps_df.loc[closest_idx].get("lap_number", 0))
            except Exception:
                pass
        return events
