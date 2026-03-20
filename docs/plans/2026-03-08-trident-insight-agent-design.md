# TridentInsightAgent — 7th OmniAgent Design

## Problem

Trident convergence reports are currently on-demand only — a user must click "Generate Report" to synthesize KeX briefings, anomaly data, and forecast signals. No proactive intelligence exists. The system has the data and the agent infrastructure but doesn't connect them.

## Solution

Add TridentInsightAgent as the 7th agent in the omniagents system. It subscribes to existing agent event topics, debounces incoming events for 60s, then auto-generates a 4-section convergence report by pulling from both MongoDB collections and buffered agent events.

## Scope (v1)

In scope:
- Event-driven auto-generation with 60s debounce
- 7th agent in omniagents chain (runs after KnowledgeConvergence)
- Database-wide cross-collection synthesis endpoint
- Staleness/caching (30-min standard, 10-min database-wide)
- User feedback ratings in frontend (thumbs up/down per section)
- SSE-driven auto-refresh in frontend

Deferred to v2:
- Periodic reports (weekly/monthly/quarterly) — needs scheduler
- Adaptive threshold hints — needs probability distribution analysis

## Architecture

### Agent: TridentInsightAgent

Extends `F1Agent` base class.

**Subscriptions:**
- `f1:telemetry:anomaly`
- `f1:strategy:recommend`
- `f1:predictive:maintenance`
- `knowledge:fused`
- `f1:pit:window:open`

**Publications:**
- `trident:report:generated`

**Debounce mechanism:**
- `on_event()` buffers incoming events and resets a 60s `asyncio.Task` timer
- When timer fires, `_auto_generate()` runs:
  1. Determine scope from buffered events (session_key, driver_number → scope/entity)
  2. Pull KeX briefings from MongoDB (`kex_briefings`, `kex_driver_briefings`)
  3. Pull anomaly snapshots from MongoDB (`anomaly_scores_snapshot`, `anomaly_briefings`)
  4. Pull forecast data from MongoDB (`telemetry_race_summaries`, `forecast_results`)
  5. Merge with buffered agent event payloads
  6. 4 parallel LLM calls → Key Insights, Anomaly Patterns, Forecast Signals, Recommendations
  7. Store to `trident_reports` (upsert latest) + `trident_reports_history` (append)
  8. Publish `trident:report:generated` event
  9. Clear buffer

### Shared Data Gathering Module

Extract `_gather_kex_data()`, `_gather_anomaly_data()`, `_gather_forecast_data()` from `advantage_router.py` into `pipeline/advantage_data.py`. Both the agent and the manual endpoint import from this shared module.

### Database-Wide Synthesis

New endpoint: `POST /api/advantage/trident/database-synthesis`

- Scans across collections: anomaly scores, forecast results, KeX briefings, all agent output collections
- Produces a cross-collection report identifying correlations
- 10-min cache TTL in `trident_reports` with `scope: "database"`
- Uses the same 4-section format

### Caching/Staleness

- Standard reports: 30-min staleness (existing `STALE_SECONDS = 1800`)
- Database-wide: 10-min TTL (`DB_STALE_SECONDS = 600`)
- `from_cache` flag on cached responses

### User Feedback

MongoDB collection: `trident_report_feedback`

```json
{
  "report_id": "trident_grid_all_1709900000",
  "section": "key_insights",
  "rating": "up",
  "comment": "Spot on about the tyre deg correlation",
  "created_at": 1709900100
}
```

New endpoints:
- `POST /api/advantage/trident/feedback` — submit rating
- `GET /api/advantage/trident/feedback/{report_id}` — get feedback for a report

### Frontend Changes (AdvantageTrident.tsx)

1. **Feedback UI**: Thumbs up/down buttons on each section card, optional comment textarea on click
2. **SSE auto-refresh**: Listen on `/api/omni/agents/stream` for `trident:report:generated` events, auto-fetch latest report when received
3. **Database synthesis tab**: New tab option alongside "Convergence Agent" and "Prompt History"

## Registration

In `omni_agents_router.py`, add as 7th agent:

```python
from omniagents.agents.trident_insight import TridentInsightAgent
_registry.register(TridentInsightAgent(bus=_bus, db=_db))
```

Registry execution order in `run_all()`:
```
1. telemetry_anomaly
2. weather_adapt
3. pit_window
4. visual_inspect
5. predictive_maintenance
6. knowledge_convergence
7. trident_insight
```

## Files

| File | Action |
|------|--------|
| `omnisuitef1/omniagents/agents/trident_insight.py` | Create |
| `pipeline/advantage_data.py` | Create |
| `pipeline/advantage_router.py` | Modify — import shared module, add database-synthesis + feedback endpoints |
| `pipeline/omni_agents_router.py` | Modify — register 7th agent |
| `omnisuitef1/omniagents/registry.py` | Modify — add trident_insight to execution order |
| `frontend/src/app/components/AdvantageTrident.tsx` | Modify — feedback UI, SSE auto-refresh, database synthesis tab |

## Collections

| Collection | Purpose |
|------------|---------|
| `trident_reports` | Latest report per scope (upsert) — exists |
| `trident_reports_history` | Full report history (append) — exists |
| `trident_report_feedback` | User feedback ratings — new |
