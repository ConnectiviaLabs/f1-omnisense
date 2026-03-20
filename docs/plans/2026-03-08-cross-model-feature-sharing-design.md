# Cross-Model Feature Sharing — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate 30-50% compute waste by consolidating duplicate ML code and adding a session-pinned feature store cache in MongoDB.

**Architecture:** Single `feature_store` MongoDB collection caches all intermediate ML results keyed by `(session_key, driver_number, computation)`. Duplicate anomaly ensemble code consolidated into `omnianalytics/anomaly.py`. Shared telemetry loader replaces 5+ independent MongoDB queries. Cache-first logic added to anomaly, SHAP, forecast, and health report pipelines.

**Tech Stack:** MongoDB (feature_store collection), Python (omnianalytics, omniagents), pandas, scikit-learn

---

## Problem

Three independent systems compute the same anomaly scores, SHAP features, and forecasts:

1. `pipeline/anomaly/ensemble.py` — batch pipeline + telemetry agent
2. `omnisuitef1/omnianalytics/anomaly.py` — API routes + omnihealth (95% identical to #1)
3. Agent chain — each agent loads telemetry and runs ML independently

Telemetry is loaded 5+ times for the same session+driver. Forecasts are never cached. SHAP is computed twice. A single `run-all` wastes ~20-40s on redundant computation.

## Solution

### 1. Feature Store Collection

Single `feature_store` collection in MongoDB. Each document is one cached computation result.

```json
{
  "_id": ObjectId,
  "session_key": 9573,
  "driver_number": 4,
  "computation": "anomaly_scores",
  "created_at": ISODate,
  "result": { ... }
}
```

**Computation types:**

| `computation` | What's cached | Producer | Consumers |
|---|---|---|---|
| `anomaly_scores` | 4-model ensemble scores + severity per component | `omnianalytics/anomaly.py` | agents, omnihealth, API routes |
| `shap_features` | Top-K SHAP features per component | `omnianalytics/anomaly.py` | omnihealth, predictive maintenance agent |
| `forecasts` | ETS/ARIMA results per feature+horizon | `omnianalytics/forecast.py` | omnihealth/risk.py, API routes |
| `health_report` | Full health assessment per driver | `omnihealth/__init__.py` | API routes, agents |
| `telemetry_scaled` | Scaled numeric telemetry DataFrame (as BSON) | shared loader | all anomaly/forecast consumers |

**Index:** Compound unique index on `(session_key, driver_number, computation)`.

**Cache strategy:** Session-pinned. Historical race data is immutable — once computed for a session+driver, results are valid forever. No TTL, no expiry.

### 2. Shared Telemetry Loader

New file: `omnisuitef1/omnianalytics/telemetry_loader.py`

Single function all agents, routers, and analytics call:

```python
async def load_telemetry(db, session_key: int, driver_number: int) -> pd.DataFrame:
    # 1. Check feature_store for pre-scaled data
    # 2. If miss: query car_data -> fallback openf1_car_data
    # 3. Scale with StandardScaler
    # 4. Cache scaled result to feature_store
    # 5. Return DataFrame
```

In-memory LRU (`maxsize=16`) on top for back-to-back agent calls in the same process.

**Replaces:**
- `telemetry_anomaly.py::_load_telemetry()`
- `predictive_maintenance.py::_load_telemetry()`
- `omni_analytics_router.py::_build_dataset()`
- `omni_health_router.py::load_driver_race_telemetry()`

### 3. Consolidated Anomaly Ensemble

`omnianalytics/anomaly.py` becomes the single source of truth for the 4-model ensemble + SHAP.

`pipeline/anomaly/ensemble.py` becomes a thin wrapper that imports from `omnianalytics/anomaly.py` — preserves backward compat for batch scripts.

Cache-first logic added:

```python
async def run(self, data, ..., session_key, driver_number):
    # 1. Check feature_store for cached anomaly_scores
    # 2. If hit: return cached result
    # 3. If miss: run 4-model ensemble + SHAP
    # 4. Cache to feature_store
    # 5. Return
```

Same pattern for `omnianalytics/forecast.py`:

```python
def forecast(data, column, horizon, ..., session_key, driver_number):
    # 1. Check feature_store for cached forecast (session+driver+column+horizon)
    # 2. If hit: return cached ForecastResult
    # 3. If miss: compute, cache, return
```

### 4. Cache-First Agent Chain Flow

**Before:**
```
Agent 1: load telemetry -> run ensemble -> publish
Agent 5: load telemetry AGAIN -> run ensemble AGAIN -> SHAP AGAIN -> forecast AGAIN
```

**After:**
```
Agent 1: load telemetry -> cache -> run ensemble -> cache -> publish
Agent 5: cache HIT telemetry -> cache HIT anomaly -> cache HIT SHAP -> forecast -> cache
```

`session_key` and `driver_number` already exist in every agent's `run_analysis()` signature. Thread them through to anomaly/forecast/loader functions.

`omnihealth.assess()` gets optional `session_key` and `driver_number` kwargs. If provided, checks feature store. If not, computes normally (no breaking change).

### 5. Estimated Savings

Per `run-all` invocation:
- Telemetry loading: 1 query instead of 3 (~2-4s saved)
- Anomaly ensemble: 1 run instead of 2-3 (~5-10s saved)
- SHAP: 1 run instead of 2 (~3-5s saved)
- Forecasts: cached across agents (~10-20s per shared feature)

**Total: ~20-40s per run-all, 30-50% compute reduction.**

## Files Changed

### New (2)
- `omnisuitef1/omnianalytics/telemetry_loader.py` — shared loader
- `omnisuitef1/omnianalytics/feature_store.py` — cache read/write helpers

### Modified (8)
- `omnisuitef1/omnianalytics/anomaly.py` — add cache-first logic
- `omnisuitef1/omnianalytics/forecast.py` — add cache-first logic
- `omnisuitef1/omnihealth/__init__.py` — accept optional session_key/driver_number, use cache
- `omnisuitef1/omnihealth/risk.py` — use cached forecasts
- `omnisuitef1/omniagents/agents/telemetry_anomaly.py` — use shared loader + consolidated ensemble
- `omnisuitef1/omniagents/agents/predictive_maintenance.py` — use shared loader + cached results
- `pipeline/anomaly/ensemble.py` — thin wrapper importing from omnianalytics
- `pipeline/omni_analytics_router.py` — use shared loader

## New MongoDB Collection

| Collection | Index | Purpose |
|---|---|---|
| `feature_store` | `(session_key, driver_number, computation)` unique | Cached ML intermediate results |
