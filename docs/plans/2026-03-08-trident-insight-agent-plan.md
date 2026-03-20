# TridentInsightAgent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TridentInsightAgent as the 7th omniagent — event-driven, debounced auto-generation of 4-section convergence reports, database-wide synthesis, and user feedback ratings.

**Architecture:** TridentInsightAgent extends F1Agent, subscribes to 5 existing event topics, debounces for 60s, then pulls from MongoDB + buffered events to produce reports via 4 parallel LLM calls. Shared data-gathering functions extracted to `pipeline/advantage_data.py`. Frontend gets feedback UI and SSE auto-refresh.

**Tech Stack:** Python (F1Agent base class, asyncio, Groq LLM), MongoDB, FastAPI, React/TypeScript

---

### Task 1: Extract shared data-gathering functions into `pipeline/advantage_data.py`

**Files:**
- Create: `pipeline/advantage_data.py`
- Modify: `pipeline/advantage_router.py:109-289`

**Step 1: Create `pipeline/advantage_data.py` with extracted functions**

```python
"""Shared Trident data-gathering functions.

Used by both advantage_router.py (manual endpoint) and
TridentInsightAgent (event-driven agent).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def gather_structured_context(db, scope: str, entity: str | None) -> str:
    """Pull structured metrics from Victory collections for comparative reasoning."""
    parts = []

    if scope == "grid" or scope == "team":
        filt: dict = {}
        if scope == "team" and entity:
            filt["team"] = {"$regex": f"^{entity}$", "$options": "i"}
        teams = list(db["victory_team_kb"].find(filt, {"_id": 0, "embedding": 0, "narrative": 0}).limit(13))
        if teams:
            parts.append("TEAM PERFORMANCE COMPARISON:")
            for t in teams:
                meta = t.get("metadata") or {}
                car = meta.get("car") or {}
                con = (car.get("constructor") or {}) if isinstance(car, dict) else {}
                telem = (car.get("telemetry") or {}) if isinstance(car, dict) else {}
                health = (car.get("health") or {}) if isinstance(car, dict) else {}
                strat = meta.get("strategy") or {}
                parts.append(
                    f"  {t.get('team')}: wins={con.get('total_wins', '?')}, "
                    f"podiums={con.get('total_podiums', '?')}, points={con.get('total_points', '?')}, "
                    f"avg_finish={con.get('avg_finish_position', '?')}, "
                    f"dnf_rate={con.get('dnf_rate', '?')}, "
                    f"avg_speed={telem.get('avg_speed', '?')}kph, "
                    f"overall_health={health.get('overall_health', '?')}%, "
                    f"undercut_aggression={strat.get('team_undercut_aggression', '?')}, "
                    f"one_stop_freq={strat.get('team_one_stop_freq', '?')}, "
                    f"avg_tyre_life={strat.get('team_avg_tyre_life', '?')} laps"
                )

    if scope == "driver" and entity:
        sp = db["victory_strategy_profiles"].find_one(
            {"driver_code": entity.upper()}, {"_id": 0, "embedding": 0, "narrative": 0}
        )
        if sp:
            ps = sp.get("pit_strategy", {})
            pe = sp.get("pit_execution", {})
            parts.append(f"DRIVER STRATEGY ({entity.upper()}):")
            parts.append(
                f"  undercut_aggression={ps.get('undercut_aggression', '?')}, "
                f"tyre_extension_bias={ps.get('tyre_extension_bias', '?')}, "
                f"one_stop_freq={ps.get('one_stop_freq', '?')}, "
                f"avg_first_stop_lap={ps.get('avg_first_stop_lap', '?')}, "
                f"pit_stops={pe.get('total_stops', '?')}, "
                f"avg_pit_duration={pe.get('avg_duration_s', '?')}s, "
                f"best_pit={pe.get('best_duration_s', '?')}s"
            )
            comps = sp.get("compound_profiles") or []
            if comps:
                parts.append("  Compound profiles:")
                for c in comps:
                    parts.append(
                        f"    {c.get('compound', '?')}: laps={c.get('total_laps', '?')}, "
                        f"avg_lap={c.get('avg_lap_time_s', '?')}s, "
                        f"tyre_life={c.get('avg_tyre_life', '?')} laps"
                    )

        rivals = list(db["victory_strategy_profiles"].find(
            {"driver_code": {"$ne": entity.upper()}},
            {"_id": 0, "embedding": 0, "narrative": 0, "compound_profiles": 0}
        ).limit(5))
        if rivals:
            parts.append(f"RIVAL STRATEGY COMPARISON (vs {entity.upper()}):")
            for r in rivals:
                rps = r.get("pit_strategy") or {}
                rpe = r.get("pit_execution") or {}
                parts.append(
                    f"  {r.get('driver_code', '?')}/{r.get('team', '?')}: "
                    f"undercut={rps.get('undercut_aggression', '?')}, "
                    f"one_stop={rps.get('one_stop_freq', '?')}, "
                    f"avg_pit={rpe.get('avg_duration_s', '?')}s"
                )

    return "\n\n".join(parts) if parts else ""


def gather_kex_data(db, scope: str, entity: str | None) -> str:
    """Gather KeX briefing data for Key Insights section."""
    parts = []

    for doc in db["kex_briefings"].find({}, {"_id": 0}).sort("year", -1).limit(2):
        text = doc.get("text") or doc.get("briefing") or doc.get("summary", "")
        if text:
            parts.append(f"[{doc.get('year', '?')} Briefing] {text[:500]}")

    filt = {}
    if scope == "driver" and entity:
        filt["driver_code"] = entity.upper()
    driver_limit = 5 if scope == "grid" else 10
    for doc in db["kex_driver_briefings"].find(filt, {"_id": 0}).sort("generated_at", -1).limit(driver_limit):
        text = doc.get("text") or doc.get("summary", "")
        if text:
            parts.append(f"[{doc.get('driver_code', '?')}] {text[:300]}")

    return "\n\n".join(parts) if parts else "No KeX briefing data available."


def gather_anomaly_data(db, scope: str, entity: str | None) -> str:
    """Gather anomaly data for Anomaly Patterns section."""
    parts = []

    snapshot = db["anomaly_scores_snapshot"].find_one({}, {"_id": 0}) or {}
    drivers = snapshot.get("drivers", [])
    if scope == "driver" and entity:
        drivers = [d for d in drivers if d.get("driver_code", "").upper() == entity.upper()]
    elif scope == "team" and entity:
        drivers = [d for d in drivers if (d.get("team", "") or "").lower() == entity.lower()]

    for d in drivers[:10]:
        code = d.get("driver_code", "?")
        races = d.get("races", [])
        if races:
            latest = races[-1]
            systems = latest.get("systems", {})
            critical = [s for s, v in systems.items() if v.get("level") == "critical"]
            warning = [s for s, v in systems.items() if v.get("level") == "warning"]
            parts.append(
                f"[{code}] Race: {latest.get('race', '?')} — "
                f"Critical: {critical or 'none'}, Warning: {warning or 'none'}"
            )

    filt = {}
    if scope == "driver" and entity:
        filt["driver_code"] = entity.upper()
    for doc in db["kex_anomaly_briefings"].find(filt, {"_id": 0}).limit(5):
        text = doc.get("text") or doc.get("summary", "")
        if text:
            parts.append(f"[Anomaly Brief: {doc.get('driver_code', '?')}] {text[:300]}")

    return "\n\n".join(parts) if parts else "No anomaly data available."


def gather_forecast_data(db, scope: str, entity: str | None) -> str:
    """Gather forecast/trend data for Forecast Signals section."""
    parts = []

    filt = {}
    if scope == "driver" and entity:
        filt["driver_code"] = entity.upper()
    elif scope == "team" and entity:
        filt["team"] = {"$regex": f"^{entity}$", "$options": "i"}

    telem_limit = 10 if scope == "grid" else 20
    for doc in db["telemetry_race_summary"].find(filt, {"_id": 0}).sort("year", -1).limit(telem_limit):
        code = doc.get("driver_code", "?")
        race = doc.get("race", "?")
        year = doc.get("year", "?")
        metrics = {k: v for k, v in doc.items()
                   if isinstance(v, (int, float)) and k not in ("year", "_id")}
        top_metrics = dict(list(metrics.items())[:6])
        parts.append(f"[{code} {race} {year}] {top_metrics}")

    brief_filt = {}
    if scope == "driver" and entity:
        brief_filt["driver_code"] = entity.upper()
    for doc in db["kex_forecast_briefings"].find(brief_filt, {"_id": 0}).limit(5):
        text = doc.get("text") or doc.get("summary", "")
        if text:
            parts.append(f"[Forecast Brief: {doc.get('driver_code', '?')}] {text[:300]}")

    car_filt = {}
    if scope == "driver" and entity:
        car_filt["driver_code"] = entity.upper()
    elif scope == "team" and entity:
        car_filt["team"] = {"$regex": f"^{entity}$", "$options": "i"}
    for doc in db["kex_car_telemetry_briefings"].find(car_filt, {"_id": 0}).limit(5):
        text = doc.get("text") or doc.get("summary", "")
        if text:
            parts.append(f"[Car Telemetry Brief: {doc.get('driver_code', doc.get('team', '?'))}] {text[:300]}")

    return "\n\n".join(parts) if parts else "No forecast/trend data available."
```

**Step 2: Update `pipeline/advantage_router.py` to import from shared module**

Replace lines 109-289 (the four `_gather_*` functions) with imports:

```python
from pipeline.advantage_data import (
    gather_structured_context,
    gather_kex_data,
    gather_anomaly_data,
    gather_forecast_data,
)
```

Then update the 4 call sites in `trident_generate()` (lines 313-316) to drop the underscore prefix:

```python
    structured = gather_structured_context(db, scope, entity)
    kex_raw = gather_kex_data(db, scope, entity)
    anomaly_raw = gather_anomaly_data(db, scope, entity)
    forecast_raw = gather_forecast_data(db, scope, entity)
```

**Step 3: Verify the server starts**

Run: `cd /home/pedrad/javier_project_folder/f1 && PYTHONPATH=pipeline:. python -c "from pipeline.advantage_data import gather_kex_data; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pipeline/advantage_data.py pipeline/advantage_router.py
git commit -m "refactor: extract shared Trident data-gathering into advantage_data module"
```

---

### Task 2: Create `TridentInsightAgent`

**Files:**
- Create: `omnisuitef1/omniagents/agents/trident_insight.py`

**Step 1: Create the agent file**

```python
"""Agent 07 — TridentInsight Agent.

Event-driven proactive report synthesizer. Subscribes to all major agent
event topics, debounces for 60s, then auto-generates a 4-section convergence
report by combining buffered agent events with MongoDB intelligence data
(KeX briefings, anomaly snapshots, forecast results).

SUB: f1:telemetry:anomaly, f1:strategy:recommend, f1:predictive:maintenance,
     knowledge:fused, f1:pit:window:open
PUB: trident:report:generated
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from omniagents._types import EventSeverity
from omniagents.base import F1Agent

logger = logging.getLogger(__name__)

TRIDENT_SYSTEM = (
    "You are an elite F1 intelligence analyst for McLaren. "
    "Synthesize data into confident, actionable insights. "
    "Do NOT include disclaimers, caveats, or limitations. "
    "Present all findings directly. Use specific numbers and driver codes."
)

DEBOUNCE_SECONDS = 60
STALE_SECONDS = 1800       # 30 min for standard reports
DB_STALE_SECONDS = 600     # 10 min for database-wide reports


class TridentInsightAgent(F1Agent):
    name = "trident_insight"
    description = (
        "Proactive report synthesizer that combines knowledge from multiple "
        "agent outputs, KeX briefings, anomaly snapshots, and forecast data "
        "into structured 4-section convergence reports. Debounces incoming "
        "events for 60 seconds before auto-generating."
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

    # ── Event handling with debounce ─────────────────────────────────────

    async def on_event(self, topic: str, event: Dict[str, Any]):
        """Buffer incoming events and reset the debounce timer."""
        payload = event.get("payload", event)

        self._event_buffer.append({
            "topic": topic,
            "agent": event.get("agent", "unknown"),
            "timestamp": event.get("timestamp", time.time()),
            "severity": event.get("severity", "info"),
            "data": payload,
        })
        logger.info(
            "[trident_insight] Buffered event from %s (buffer size: %d)",
            topic, len(self._event_buffer),
        )

        # Reset debounce timer
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounce_then_generate())

    async def _debounce_then_generate(self):
        """Wait for debounce period, then auto-generate a report."""
        try:
            await asyncio.sleep(DEBOUNCE_SECONDS)
            logger.info(
                "[trident_insight] Debounce complete — generating report from %d buffered events",
                len(self._event_buffer),
            )
            await self._auto_generate()
        except asyncio.CancelledError:
            pass  # Timer reset — new events arrived

    async def _auto_generate(self):
        """Generate a report from buffered events + MongoDB data."""
        if not self._event_buffer:
            return

        # Snapshot and clear the buffer
        events = list(self._event_buffer)
        self._event_buffer.clear()

        # Infer scope from buffered events
        scope, entity = self._infer_scope(events)

        report = await self._build_report(scope, entity, events)
        if report:
            await self.publish(
                topic="trident:report:generated",
                payload={"report_id": report["report_id"], "scope": scope, "entity": entity},
                severity=EventSeverity.INFO,
            )

    # ── Manual run_analysis (called via /run/{agent_name} or run-all) ────

    async def run_analysis(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Manual trigger — generate a grid-scope report using MongoDB data + any buffered events."""
        events = list(self._event_buffer)
        self._event_buffer.clear()

        # Cancel any pending debounce
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        report = await self._build_report("grid", None, events)
        return report or {"status": "no_data", "summary": "No data available for report generation"}

    # ── Report builder (shared by auto + manual paths) ───────────────────

    async def _build_report(
        self,
        scope: str,
        entity: Optional[str],
        buffered_events: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Build a 4-section Trident report from MongoDB data + buffered events."""
        if self._db is None:
            logger.warning("[trident_insight] No database — cannot generate report")
            return None

        now = time.time()

        # Check cache
        cached = self._db["trident_reports"].find_one(
            {"scope": scope, "entity": entity, "stale_after": {"$gt": now}},
            {"_id": 0},
        )
        if cached:
            logger.info("[trident_insight] Returning cached report for %s/%s", scope, entity)
            return cached

        t0 = time.time()

        # Gather data from MongoDB
        from pipeline.advantage_data import (
            gather_structured_context,
            gather_kex_data,
            gather_anomaly_data,
            gather_forecast_data,
        )

        structured = gather_structured_context(self._db, scope, entity)
        kex_raw = gather_kex_data(self._db, scope, entity)
        anomaly_raw = gather_anomaly_data(self._db, scope, entity)
        forecast_raw = gather_forecast_data(self._db, scope, entity)

        # Append buffered agent event summaries to relevant sections
        event_context = self._format_buffered_events(buffered_events)
        if event_context.get("anomaly"):
            anomaly_raw += f"\n\nREAL-TIME AGENT EVENTS:\n{event_context['anomaly']}"
        if event_context.get("strategy"):
            forecast_raw += f"\n\nREAL-TIME STRATEGY EVENTS:\n{event_context['strategy']}"
        if event_context.get("fused"):
            kex_raw += f"\n\nFUSED INTELLIGENCE:\n{event_context['fused']}"

        scope_label = "the entire F1 grid" if scope == "grid" else f"{entity}"
        metrics_block = f"\n\nSTRUCTURED METRICS:\n{structured}" if structured else ""

        # Build prompts
        key_insights_prompt = (
            f"Based on the following KeX briefings and driver intelligence for {scope_label}, "
            f"write 3-5 key insights about current performance patterns. "
            f"Use the structured metrics to make specific numerical comparisons.\n\n"
            f"{kex_raw}{metrics_block}"
        )
        anomaly_prompt = (
            f"Based on the following anomaly detection data and car health metrics for {scope_label}, "
            f"identify 3-5 significant anomaly patterns and health trends. "
            f"Reference specific system health scores and compare across teams/drivers where possible.\n\n"
            f"{anomaly_raw}{metrics_block}"
        )
        forecast_prompt = (
            f"Based on the following telemetry trends and strategy data for {scope_label}, "
            f"identify 3-5 forecast signals and predicted performance trends. "
            f"Use the structured strategy and telemetry metrics to support predictions with data.\n\n"
            f"{forecast_raw}{metrics_block}"
        )

        # 3 parallel LLM calls
        key_insights, anomaly_patterns, forecast_signals = await asyncio.gather(
            self.reason(key_insights_prompt),
            self.reason(anomaly_prompt),
            self.reason(forecast_prompt),
        )

        # Recommendations depends on the 3 above
        recommendations_prompt = (
            f"Based on these three intelligence pillars for {scope_label}, "
            f"provide 3-5 actionable strategic recommendations:\n\n"
            f"KEY INSIGHTS:\n{key_insights or 'N/A'}\n\n"
            f"ANOMALY PATTERNS:\n{anomaly_patterns or 'N/A'}\n\n"
            f"FORECAST SIGNALS:\n{forecast_signals or 'N/A'}"
        )
        recommendations = await self.reason(recommendations_prompt)

        gen_time = round(time.time() - t0, 2)
        report_id = f"trident_{scope}_{entity or 'all'}_{int(now)}"

        report = {
            "report_id": report_id,
            "scope": scope,
            "entity": entity,
            "generated_at": now,
            "stale_after": now + STALE_SECONDS,
            "source": "agent",  # distinguishes from manual endpoint
            "event_count": len(buffered_events),
            "sections": {
                "key_insights": {"title": "Key Insights", "content": key_insights or "No data available"},
                "recommendations": {"title": "Recommendations", "content": recommendations or "No data available"},
                "anomaly_patterns": {"title": "Anomaly Patterns", "content": anomaly_patterns or "No data available"},
                "forecast_signals": {"title": "Forecast Signals", "content": forecast_signals or "No data available"},
            },
            "metadata": {
                "model_used": self._groq.__class__.__name__ if self._groq else "none",
                "generation_time_s": gen_time,
            },
        }

        # Persist
        self._db["trident_reports"].update_one(
            {"scope": scope, "entity": entity},
            {"$set": report},
            upsert=True,
        )
        self._db["trident_reports_history"].insert_one({**report})

        logger.info("[trident_insight] Report generated: %s (%.2fs)", report_id, gen_time)
        return report

    # ── Helpers ──────────────────────────────────────────────────────────

    def _infer_scope(self, events: List[Dict[str, Any]]) -> tuple[str, Optional[str]]:
        """Infer report scope from buffered events. Default to grid."""
        # If all events reference the same driver, scope to that driver
        driver_numbers = set()
        for e in events:
            dn = e.get("data", {}).get("driver_number")
            if dn:
                driver_numbers.add(dn)

        if len(driver_numbers) == 1:
            return "driver", str(driver_numbers.pop())

        return "grid", None

    def _format_buffered_events(self, events: List[Dict[str, Any]]) -> Dict[str, str]:
        """Format buffered events into context strings grouped by category."""
        anomaly_parts, strategy_parts, fused_parts = [], [], []

        for e in events:
            topic = e.get("topic", "")
            data = e.get("data", {})
            summary = data.get("summary") or data.get("insight") or data.get("briefing", "")
            if not summary:
                continue

            text = f"[{e.get('agent', '?')}] {str(summary)[:300]}"

            if "anomaly" in topic or "maintenance" in topic:
                anomaly_parts.append(text)
            elif "strategy" in topic or "pit" in topic:
                strategy_parts.append(text)
            elif "fused" in topic or "knowledge" in topic:
                fused_parts.append(text)

        return {
            "anomaly": "\n".join(anomaly_parts),
            "strategy": "\n".join(strategy_parts),
            "fused": "\n".join(fused_parts),
        }
```

**Step 2: Verify the module imports**

Run: `cd /home/pedrad/javier_project_folder/f1 && PYTHONPATH=omnisuitef1:pipeline:. python -c "from omniagents.agents.trident_insight import TridentInsightAgent; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add omnisuitef1/omniagents/agents/trident_insight.py
git commit -m "feat: add TridentInsightAgent — 7th omniagent with debounced auto-generation"
```

---

### Task 3: Register the agent and update execution order

**Files:**
- Modify: `pipeline/omni_agents_router.py:40-70`
- Modify: `omnisuitef1/omniagents/registry.py:62-69`

**Step 1: Add import and registration in `omni_agents_router.py`**

At line 47, add the import:

```python
    from omniagents.agents.trident_insight import TridentInsightAgent
```

At line 68, add the registration:

```python
    _registry.register(TridentInsightAgent(bus=_bus, db=_db))
```

Update the comment on line 62 from `# Register all 6 agents` to `# Register all 7 agents`.

**Step 2: Add `trident_insight` to the output collection map**

In `omni_agents_router.py` at line 224, add:

```python
        "trident_insight": "trident_reports_history",
```

**Step 3: Update execution order in `registry.py`**

At line 68 in `registry.py`, add `"trident_insight"` to the end of `execution_order`:

```python
        execution_order = [
            "telemetry_anomaly",
            "weather_adapt",
            "pit_window",
            "visual_inspect",
            "predictive_maintenance",
            "knowledge_convergence",
            "trident_insight",
        ]
```

Update the docstring on line 59 to include `→ trident_insight`.

**Step 4: Commit**

```bash
git add pipeline/omni_agents_router.py omnisuitef1/omniagents/registry.py
git commit -m "feat: register TridentInsightAgent as 7th agent in chain"
```

---

### Task 4: Add database-wide synthesis and feedback endpoints

**Files:**
- Modify: `pipeline/advantage_router.py`

**Step 1: Add the feedback model and database-synthesis request model**

After the `TridentGenerateRequest` class (line 107), add:

```python
class TridentFeedbackRequest(BaseModel):
    report_id: str
    section: str
    rating: str  # "up" or "down"
    comment: Optional[str] = None


class TridentDbSynthesisRequest(BaseModel):
    force: bool = False
```

**Step 2: Add the database-wide synthesis endpoint**

After the existing `/trident/report/{report_id}` endpoint, add:

```python
DB_STALE_SECONDS = 600  # 10 min cache for database-wide reports

@router.post("/trident/database-synthesis")
async def trident_database_synthesis(body: TridentDbSynthesisRequest):
    """Cross-collection synthesis report scanning all intelligence collections."""
    db = _get_db()
    now = time.time()

    if not body.force:
        cached = db["trident_reports"].find_one(
            {"scope": "database", "stale_after": {"$gt": now}},
            {"_id": 0},
        )
        if cached:
            cached["from_cache"] = True
            return _sanitize(cached)

    t0 = time.time()

    # Scan across all agent output collections
    agent_collections = [
        "agent_telemetry_anomalies",
        "agent_weather_alerts",
        "agent_pit_windows",
        "agent_knowledge_fused",
        "agent_visual_incidents",
        "agent_predictive_maintenance",
    ]
    cross_data_parts = []
    for coll_name in agent_collections:
        recent = list(db[coll_name].find({}, {"_id": 0}).sort("_timestamp", -1).limit(3))
        if recent:
            summaries = []
            for doc in recent:
                s = doc.get("summary") or doc.get("insight") or doc.get("briefing", "")
                if s:
                    summaries.append(str(s)[:200])
            if summaries:
                cross_data_parts.append(f"[{coll_name}]\n" + "\n".join(summaries))

    # Also pull standard Trident sources
    kex_raw = gather_kex_data(db, "grid", None)
    anomaly_raw = gather_anomaly_data(db, "grid", None)
    forecast_raw = gather_forecast_data(db, "grid", None)

    all_data = (
        f"CROSS-COLLECTION AGENT DATA:\n{chr(10).join(cross_data_parts)}\n\n"
        f"KEX BRIEFINGS:\n{kex_raw}\n\n"
        f"ANOMALY DATA:\n{anomaly_raw}\n\n"
        f"FORECAST DATA:\n{forecast_raw}"
    )

    key_insights_prompt = (
        "Synthesize the following cross-collection data from the entire McLaren intelligence platform. "
        "Identify 3-5 key insights that emerge from correlating data across multiple collections.\n\n"
        f"{all_data}"
    )
    anomaly_prompt = (
        "From the following cross-collection data, identify 3-5 anomaly patterns "
        "that span multiple data sources or correlate across collections.\n\n"
        f"{all_data}"
    )
    forecast_prompt = (
        "From the following cross-collection data, identify 3-5 forecast signals "
        "that are supported by evidence from multiple data sources.\n\n"
        f"{all_data}"
    )

    key_insights, anomaly_patterns, forecast_signals = await asyncio.gather(
        asyncio.to_thread(_llm_synthesize, key_insights_prompt, TRIDENT_SYSTEM),
        asyncio.to_thread(_llm_synthesize, anomaly_prompt, TRIDENT_SYSTEM),
        asyncio.to_thread(_llm_synthesize, forecast_prompt, TRIDENT_SYSTEM),
    )

    recommendations_prompt = (
        "Based on these cross-collection findings, provide 3-5 actionable recommendations:\n\n"
        f"KEY INSIGHTS:\n{key_insights}\n\n"
        f"ANOMALY PATTERNS:\n{anomaly_patterns}\n\n"
        f"FORECAST SIGNALS:\n{forecast_signals}"
    )
    recommendations = await asyncio.to_thread(_llm_synthesize, recommendations_prompt, TRIDENT_SYSTEM)

    gen_time = round(time.time() - t0, 2)
    report_id = f"trident_database_all_{int(now)}"

    report = {
        "report_id": report_id,
        "scope": "database",
        "entity": None,
        "generated_at": now,
        "stale_after": now + DB_STALE_SECONDS,
        "sections": {
            "key_insights": {"title": "Key Insights", "content": key_insights},
            "recommendations": {"title": "Recommendations", "content": recommendations},
            "anomaly_patterns": {"title": "Anomaly Patterns", "content": anomaly_patterns},
            "forecast_signals": {"title": "Forecast Signals", "content": forecast_signals},
        },
        "metadata": {
            "model_used": GROQ_MODEL,
            "generation_time_s": gen_time,
            "collections_scanned": len(agent_collections),
        },
    }

    db["trident_reports"].update_one(
        {"scope": "database"},
        {"$set": report},
        upsert=True,
    )
    db["trident_reports_history"].insert_one({**report})

    report.pop("_id", None)
    return _sanitize(report)
```

**Step 3: Add the feedback endpoints**

```python
@router.post("/trident/feedback")
async def trident_submit_feedback(body: TridentFeedbackRequest):
    """Submit user feedback for a Trident report section."""
    db = _get_db()
    doc = {
        "report_id": body.report_id,
        "section": body.section,
        "rating": body.rating,
        "comment": body.comment,
        "created_at": time.time(),
    }
    db["trident_report_feedback"].insert_one(doc)
    return {"status": "ok"}


@router.get("/trident/feedback/{report_id}")
async def trident_get_feedback(report_id: str):
    """Get all feedback for a specific report."""
    db = _get_db()
    docs = list(db["trident_report_feedback"].find(
        {"report_id": report_id}, {"_id": 0}
    ))
    return {"feedback": docs, "count": len(docs)}
```

**Step 4: Commit**

```bash
git add pipeline/advantage_router.py
git commit -m "feat: add database-wide synthesis and feedback endpoints for Trident"
```

---

### Task 5: Update frontend — feedback UI, SSE auto-refresh, database synthesis tab

**Files:**
- Modify: `frontend/src/app/components/AdvantageTrident.tsx`

**Step 1: Add feedback state and SSE connection**

Add new state variables after the existing state declarations (around line 83):

```typescript
  const [feedbackSent, setFeedbackSent] = useState<Record<string, string>>({});
  const [feedbackComment, setFeedbackComment] = useState('');
  const [activeFeedbackSection, setActiveFeedbackSection] = useState<string | null>(null);
```

Add `database` to SCOPE_OPTIONS (after the existing 3 options, around line 64):

```typescript
const SCOPE_OPTIONS = [
  { value: 'grid', label: 'Full Grid', icon: LayoutGrid, hint: 'Analyze all drivers and teams across the entire grid' },
  { value: 'driver', label: 'Driver', icon: Users, hint: 'Focus analysis on a single driver\'s performance profile' },
  { value: 'team', label: 'Team', icon: Shield, hint: 'Focus analysis on a single team\'s operational profile' },
  { value: 'database', label: 'Database', icon: Layers, hint: 'Cross-collection synthesis across the entire intelligence platform' },
];
```

**Step 2: Add SSE auto-refresh effect**

After the existing `useEffect` for loading latest report (around line 120), add:

```typescript
  // SSE auto-refresh: listen for trident:report:generated events
  useEffect(() => {
    let eventSource: EventSource | null = null;
    try {
      eventSource = new EventSource('/api/omni/agents/stream');
      eventSource.addEventListener('agent_event', (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.topic === 'trident:report:generated') {
            // Auto-fetch latest report
            fetch(`/api/advantage/trident/latest?scope=${scope}${entity ? `&entity=${entity}` : ''}`)
              .then(r => { if (r.ok) return r.json(); throw new Error('fetch failed'); })
              .then(setReport)
              .catch(() => {});
          }
        } catch {}
      });
    } catch {}
    return () => { eventSource?.close(); };
  }, [scope, entity]);
```

**Step 3: Update the generate function to handle database scope**

Modify the `generate` function (around line 122) to route database scope to the new endpoint:

```typescript
  const generate = async (force = false) => {
    setLoading(true);
    setError('');
    try {
      let res: Response;
      if (scope === 'database') {
        res = await fetch('/api/advantage/trident/database-synthesis', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ force }),
        });
      } else {
        res = await fetch('/api/advantage/trident/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            scope,
            entity: scope === 'grid' ? null : entity || null,
            force,
          }),
        });
      }
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setReport(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  };
```

**Step 4: Add feedback UI to section cards**

Add imports at the top: `ThumbsUp, ThumbsDown, MessageSquare` from lucide-react.

Inside the report section cards (the `SECTION_CONFIG.map` block, around line 313), add feedback buttons after the card content div:

```typescript
{/* Feedback */}
<div className="flex items-center gap-2 px-4 py-2 border-t" style={{ borderColor: `${accent}15` }}>
  {feedbackSent[`${report.report_id}:${key}`] ? (
    <span className="text-[11px] text-muted-foreground/60">
      {feedbackSent[`${report.report_id}:${key}`] === 'up' ? 'Helpful' : 'Not helpful'} — thanks
    </span>
  ) : (
    <>
      <button
        onClick={async () => {
          await fetch('/api/advantage/trident/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ report_id: report.report_id, section: key, rating: 'up' }),
          });
          setFeedbackSent(prev => ({ ...prev, [`${report.report_id}:${key}`]: 'up' }));
        }}
        className="p-1.5 rounded hover:bg-green-500/10 transition-colors"
        title="This section was helpful"
      >
        <ThumbsUp className="w-3 h-3 text-muted-foreground hover:text-green-400" />
      </button>
      <button
        onClick={async () => {
          await fetch('/api/advantage/trident/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ report_id: report.report_id, section: key, rating: 'down' }),
          });
          setFeedbackSent(prev => ({ ...prev, [`${report.report_id}:${key}`]: 'down' }));
        }}
        className="p-1.5 rounded hover:bg-red-500/10 transition-colors"
        title="This section was not helpful"
      >
        <ThumbsDown className="w-3 h-3 text-muted-foreground hover:text-red-400" />
      </button>
    </>
  )}
</div>
```

**Step 5: Verify frontend builds**

Run: `cd /home/pedrad/javier_project_folder/f1/frontend && npx tsc --noEmit`
Expected: No errors

**Step 6: Commit**

```bash
git add frontend/src/app/components/AdvantageTrident.tsx
git commit -m "feat: add feedback UI, SSE auto-refresh, and database synthesis to Trident frontend"
```

---

### Task 6: End-to-end verification

**Step 1: Start the backend**

Run: `cd /home/pedrad/javier_project_folder/f1 && PYTHONPATH=pipeline:. python -m uvicorn pipeline.chat_server:app --host 0.0.0.0 --port 8300`

**Step 2: Verify agent registration**

Run: `curl -s http://localhost:8300/api/omni/agents/status | python -m json.tool | grep trident_insight`
Expected: `trident_insight` appears in the agents list

**Step 3: Verify database synthesis endpoint**

Run: `curl -s -X POST http://localhost:8300/api/advantage/trident/database-synthesis -H 'Content-Type: application/json' -d '{"force": false}'`
Expected: JSON response with 4-section report

**Step 4: Verify feedback endpoint**

Run: `curl -s -X POST http://localhost:8300/api/advantage/trident/feedback -H 'Content-Type: application/json' -d '{"report_id": "test", "section": "key_insights", "rating": "up"}'`
Expected: `{"status": "ok"}`

**Step 5: Verify frontend builds**

Run: `cd /home/pedrad/javier_project_folder/f1/frontend && npm run build`
Expected: Build succeeds

**Step 6: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address any issues from e2e verification"
```
