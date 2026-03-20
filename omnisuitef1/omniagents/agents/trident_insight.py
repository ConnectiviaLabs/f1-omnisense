"""Agent 07 — TridentInsight Agent.

Subscribes to 5 agent topics and auto-generates a 4-section Trident report
after a 60-second debounce window. Also supports manual triggering via
run_analysis().

SUB: f1:telemetry:anomaly, f1:strategy:recommend, f1:predictive:maintenance,
     knowledge:fused, f1:pit:window:open
PUB: trident:report:generated
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from omniagents._types import EventSeverity
from omniagents.base import F1Agent

logger = logging.getLogger(__name__)

_DEBOUNCE_SECONDS = 60
_STALE_SECONDS = 3600  # report considered stale after 1 hour
_GROQ_MODEL = os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile")


class TridentInsightAgent(F1Agent):
    name = "trident_insight"
    description = (
        "Synthesises cross-agent intelligence into structured Trident reports. "
        "Produces 4-section analysis (Key Insights, Anomaly Patterns, Forecast "
        "Signals, Recommendations) by merging live agent events with historical "
        "MongoDB data. Reports auto-generate after a 60-second debounce window "
        "once upstream agents publish events."
    )
    subscriptions = [
        "f1:telemetry:anomaly",
        "f1:strategy:recommend",
        "f1:predictive:maintenance",
        "knowledge:fused",
        "f1:pit:window:open",
    ]
    publications = ["trident:report:generated"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_buffer: List[Dict[str, Any]] = []
        self._debounce_task: Optional[asyncio.Task] = None

    # ── Event handling ──────────────────────────────────────────────────────

    async def on_event(self, topic: str, event: Dict[str, Any]):
        """Buffer incoming events and reset the debounce timer."""
        payload = event.get("payload", event)
        self._event_buffer.append({
            "topic": topic,
            "agent": event.get("agent", "unknown"),
            "timestamp": event.get("timestamp", time.time()),
            "severity": event.get("severity", "info"),
            "driver_number": payload.get("driver_number"),
            "data": payload,
        })
        logger.info(
            "[trident_insight] Buffered event from %s (%d total)",
            topic, len(self._event_buffer),
        )

        # Reset debounce timer
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.ensure_future(self._debounce_fire())

    async def _debounce_fire(self):
        """Wait for the debounce window then auto-generate a report."""
        try:
            await asyncio.sleep(_DEBOUNCE_SECONDS)
            logger.info("[trident_insight] Debounce timer fired — auto-generating report")
            scope, entity = self._infer_scope()
            await self._build_report(scope=scope, entity=entity, session_key=None)
        except asyncio.CancelledError:
            pass  # timer was reset by a new event
        except Exception:
            logger.exception("[trident_insight] Auto-generation failed")

    # ── Main analysis entrypoint ────────────────────────────────────────────

    async def run_analysis(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Manual trigger: generate a Trident report for the given scope."""
        scope = "driver" if driver_number else "grid"
        entity = str(driver_number) if driver_number else None
        return await self._build_report(
            scope=scope,
            entity=entity,
            session_key=session_key,
        )

    # ── Report generation ───────────────────────────────────────────────────

    async def _build_report(
        self,
        scope: str = "grid",
        entity: Optional[str] = None,
        session_key: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the 4-section Trident report."""
        start = time.time()
        now = datetime.now(timezone.utc)

        # --- Cache check: return fresh report if not stale ---
        cached = await self._check_cache(scope, entity, now)
        if cached is not None:
            logger.info("[trident_insight] Returning cached report for %s/%s", scope, entity)
            return cached

        # --- Gather MongoDB context (off the event loop) ---
        mongo_ctx = await asyncio.to_thread(self._gather_mongo_data, scope, entity)

        # --- RAG deep search for document grounding ---
        rag_context = ""
        if self._deep_search_override or True:  # TridentInsight always uses deep search
            rag_queries = [
                f"McLaren {scope} race analysis intelligence",
                f"McLaren anomaly detection telemetry {entity or 'grid'}",
                f"McLaren forecast maintenance schedule",
            ]
            rag_blocks = await asyncio.gather(
                *(self.deep_search_context(q, k=3) for q in rag_queries)
            )
            rag_context = "\n---\n".join(
                block for block in rag_blocks
                if block != "No relevant context found."
            )

        # --- Format buffered events by category ---
        categorised = self._format_buffered_events()

        # --- Merge buffered agent events into a summary string ---
        event_summary = self._summarise_buffer()
        buffered_count = len(self._event_buffer)

        # --- 3 parallel LLM calls for first 3 sections ---
        insights_prompt = (
            "Generate a concise Key Insights section for a McLaren race intelligence report.\n"
            f"Scope: {scope}, Entity: {entity or 'all'}\n\n"
            f"Agent events:\n{event_summary}\n\n"
            f"Fused intelligence events:\n{categorised.get('fused', 'N/A')}\n\n"
            f"KeX / structured data:\n{mongo_ctx.get('kex', 'N/A')}\n"
            f"{mongo_ctx.get('structured', 'N/A')}"
        )
        anomaly_prompt = (
            "Generate a concise Anomaly Patterns section for a McLaren race intelligence report.\n"
            f"Scope: {scope}, Entity: {entity or 'all'}\n\n"
            f"Anomaly / maintenance events:\n{categorised.get('anomaly', 'N/A')}\n\n"
            f"Anomaly data:\n{mongo_ctx.get('anomaly', 'N/A')}"
        )
        forecast_prompt = (
            "Generate a concise Forecast Signals section for a McLaren race intelligence report.\n"
            f"Scope: {scope}, Entity: {entity or 'all'}\n\n"
            f"Strategy / pit events:\n{categorised.get('strategy', 'N/A')}\n\n"
            f"Forecast data:\n{mongo_ctx.get('forecast', 'N/A')}"
        )

        if rag_context:
            rag_section = f"\n\nRelevant technical documentation:\n{rag_context}"
            insights_prompt += rag_section
            anomaly_prompt += rag_section
            forecast_prompt += rag_section

        insights_text, anomaly_text, forecast_text = await asyncio.gather(
            self.reason(insights_prompt),
            self.reason(anomaly_prompt),
            self.reason(forecast_prompt),
        )

        # --- Sequential 4th call: Recommendations depends on the other 3 ---
        reco_prompt = (
            "Based on the following three analysis sections, generate a concise "
            "Recommendations section for a McLaren race intelligence report. "
            "Prioritise actionable items.\n\n"
            f"Key Insights:\n{insights_text or 'N/A'}\n\n"
            f"Anomaly Patterns:\n{anomaly_text or 'N/A'}\n\n"
            f"Forecast Signals:\n{forecast_text or 'N/A'}"
        )
        reco_text = await self.reason(reco_prompt)

        # --- Build report document ---
        ts_str = now.strftime("%Y%m%d%H%M%S")
        report_id = f"trident_{scope}_{entity or 'all'}_{ts_str}"

        report: Dict[str, Any] = {
            "report_id": report_id,
            "scope": scope,
            "entity": entity,
            "generated_at": now,
            "stale_after": datetime.fromtimestamp(
                now.timestamp() + _STALE_SECONDS, tz=timezone.utc,
            ),
            "source": "agent",
            "event_count": buffered_count,
            "sections": {
                "key_insights": {
                    "title": "Key Insights",
                    "content": insights_text or "No insights generated.",
                },
                "anomaly_patterns": {
                    "title": "Anomaly Patterns",
                    "content": anomaly_text or "No anomaly patterns detected.",
                },
                "forecast_signals": {
                    "title": "Forecast Signals",
                    "content": forecast_text or "No forecast signals available.",
                },
                "recommendations": {
                    "title": "Recommendations",
                    "content": reco_text or "No recommendations generated.",
                },
            },
            "metadata": {
                "model_used": _GROQ_MODEL,
                "generation_time_s": round(time.time() - start, 2),
                "deep_search": bool(rag_context),
            },
        }

        # --- Persist ---
        if self._db is not None:
            try:
                await asyncio.to_thread(
                    self._db["trident_reports"].update_one,
                    {"scope": scope, "entity": entity},
                    {"$set": report},
                    True,  # upsert
                )
            except Exception:
                logger.exception("[trident_insight] Failed to upsert trident_reports")

            await self.save_output("trident_reports_history", report.copy())

        # --- Publish event ---
        await self.publish(
            topic="trident:report:generated",
            payload={
                "report_id": report_id,
                "scope": scope,
                "entity": entity,
                "event_count": buffered_count,
                "summary": (insights_text or "")[:200],
            },
            severity=EventSeverity.INFO,
            session_key=session_key,
        )

        # --- Clear buffer after successful generation ---
        self._event_buffer.clear()

        self.set_output(report, session_key)
        return report

    # ── Helpers ─────────────────────────────────────────────────────────────

    async def _check_cache(
        self, scope: str, entity: Optional[str], now: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Return a cached report if one exists and is not stale."""
        if self._db is None:
            return None
        try:
            doc = await asyncio.to_thread(
                self._db["trident_reports"].find_one,
                {"scope": scope, "entity": entity, "stale_after": {"$gt": now}},
                {"_id": 0},
            )
            return doc  # None if no fresh report found
        except Exception:
            logger.exception("[trident_insight] Cache check failed")
            return None

    def _infer_scope(self) -> tuple[str, Optional[str]]:
        """Inspect buffered events to determine scope and entity.

        If every event references the same driver_number, scope to that driver.
        Otherwise default to grid-wide.
        """
        driver_numbers = set()
        for evt in self._event_buffer:
            dn = evt.get("driver_number")
            if dn is not None:
                driver_numbers.add(dn)

        if len(driver_numbers) == 1:
            dn = driver_numbers.pop()
            return "driver", str(dn)
        return "grid", None

    def _format_buffered_events(self) -> Dict[str, str]:
        """Group buffered events by category and return formatted strings.

        Categories:
          anomaly — topics containing 'anomaly' or 'maintenance'
          strategy — topics containing 'strategy' or 'pit'
          fused — topics containing 'fused' or 'knowledge'
        """
        buckets: Dict[str, List[str]] = {"anomaly": [], "strategy": [], "fused": []}

        for evt in self._event_buffer:
            topic = evt.get("topic", "")
            data = evt.get("data", {})
            agent = evt.get("agent", "?")
            sev = evt.get("severity", "info")
            summary = (
                data.get("summary")
                or data.get("insight")
                or data.get("recommendation")
                or str(data)[:150]
            )
            line = f"[{sev}] {topic} ({agent}): {summary}"

            topic_lower = topic.lower()
            if "anomaly" in topic_lower or "maintenance" in topic_lower:
                buckets["anomaly"].append(line)
            elif "strategy" in topic_lower or "pit" in topic_lower:
                buckets["strategy"].append(line)
            elif "fused" in topic_lower or "knowledge" in topic_lower:
                buckets["fused"].append(line)

        return {k: "\n".join(v) if v else "No events." for k, v in buckets.items()}

    def _gather_mongo_data(self, scope: str, entity: Optional[str]) -> Dict[str, str]:
        """Pull data from MongoDB via advantage_data helpers. Degrades gracefully."""
        result: Dict[str, str] = {}
        if self._db is None:
            return result

        try:
            from pipeline.advantage_data import (
                gather_anomaly_data,
                gather_forecast_data,
                gather_kex_data,
                gather_structured_context,
            )
            result["structured"] = gather_structured_context(self._db, scope, entity)
            result["kex"] = gather_kex_data(self._db, scope, entity)
            result["anomaly"] = gather_anomaly_data(self._db, scope, entity)
            result["forecast"] = gather_forecast_data(self._db, scope, entity)
        except ImportError:
            logger.warning("[trident_insight] pipeline.advantage_data not available — skipping MongoDB context")
        except Exception:
            logger.exception("[trident_insight] Error gathering MongoDB context")

        return result

    def _summarise_buffer(self) -> str:
        """Build a text summary of buffered agent events."""
        if not self._event_buffer:
            return "No agent events buffered."

        lines: List[str] = []
        for evt in self._event_buffer[-30:]:  # cap at 30 most recent
            topic = evt.get("topic", "?")
            agent = evt.get("agent", "?")
            sev = evt.get("severity", "info")
            data = evt.get("data", {})
            summary = (
                data.get("summary")
                or data.get("insight")
                or data.get("recommendation")
                or str(data)[:150]
            )
            lines.append(f"[{sev}] {topic} ({agent}): {summary}")

        if len(self._event_buffer) > 30:
            lines.append(f"... and {len(self._event_buffer) - 30} more events")

        return "\n".join(lines)
