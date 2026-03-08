# Forecast Backtest & Validation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a walk-forward forecast backtesting system that validates 4 forecast methods against real F1 telemetry, surfacing accuracy metrics in a new BacktestView tab.

**Architecture:** Pure computation module (`forecast_backtest.py`) takes a pandas Series, runs walk-forward cross-validation across methods and horizons, computes 6 metrics per combination. API endpoints store/serve results from MongoDB. Frontend renders a "Forecast Validation" tab with method comparison, horizon decay, and feature heatmap.

**Tech Stack:** Python (numpy, pandas), existing `forecast_*()` functions from `omnianalytics/forecast.py`, FastAPI endpoints, React/Recharts frontend, MongoDB `backtest_forecast_results` collection.

**Design doc:** `docs/plans/2026-03-08-forecast-backtest-design.md`

---

### Task 1: Create `forecast_backtest.py` — Metrics + Walk-Forward Engine

**Files:**
- Create: `omnisuitef1/omnianalytics/forecast_backtest.py`

**Step 1: Write the metrics module**

Create `omnisuitef1/omnianalytics/forecast_backtest.py` with the metrics computation and walk-forward engine.

```python
"""Walk-forward forecast backtesting for F1 telemetry.

Validates forecast methods against held-out data using expanding-window
cross-validation. Computes MAE, RMSE, MAPE, directional accuracy,
CI coverage, and RMSSE per (method, feature, horizon).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from omnianalytics.forecast import (
    forecast_arima,
    forecast_ets,
    forecast_linear,
    forecast_sf,
)

logger = logging.getLogger(__name__)

METHODS = ["linear", "ets", "arima", "sf"]
DEFAULT_HORIZONS = [5, 10, 30]
DEFAULT_FEATURES = ["speed", "rpm", "throttle", "brake", "n_gear", "drs"]
MIN_TRAIN_SIZE = 200


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    naive_errors: np.ndarray,
) -> Dict[str, float]:
    """Compute 6 accuracy metrics for a single forecast window.

    Parameters
    ----------
    actual : true values for the horizon window
    predicted : forecasted values
    lower, upper : 95% confidence bounds (or None)
    naive_errors : absolute errors from naive (last-value) baseline,
                   computed across the full training set for RMSSE scaling

    Returns
    -------
    dict with mae, rmse, mape, directional_acc, ci_coverage, rmsse
    """
    n = len(actual)
    errors = actual - predicted
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE — guard against division by zero
    nonzero = np.abs(actual) > 1e-8
    if nonzero.any():
        mape = float(np.mean(abs_errors[nonzero] / np.abs(actual[nonzero])) * 100)
    else:
        mape = 0.0

    # Directional accuracy — did forecast predict direction of change?
    if n >= 2:
        actual_dir = np.diff(actual) > 0
        pred_dir = np.diff(predicted) > 0
        directional_acc = float(np.mean(actual_dir == pred_dir))
    else:
        directional_acc = 1.0

    # CI coverage — proportion of actuals within bounds
    if lower is not None and upper is not None:
        within = (actual >= lower) & (actual <= upper)
        ci_coverage = float(np.mean(within))
    else:
        ci_coverage = None

    # RMSSE — relative to naive baseline (random walk)
    naive_mse = float(np.mean(naive_errors ** 2)) if len(naive_errors) > 0 else 1.0
    if naive_mse > 1e-10:
        rmsse = float(np.sqrt(np.mean(errors ** 2) / naive_mse))
    else:
        rmsse = 1.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "directional_acc": round(directional_acc, 4),
        "ci_coverage": round(ci_coverage, 4) if ci_coverage is not None else None,
        "rmsse": round(rmsse, 4),
    }


# ── Walk-forward engine ──────────────────────────────────────────────────

def _run_method(method: str, train_series: pd.Series, horizon: int):
    """Call the appropriate forecast function. Returns ForecastResult."""
    dispatch = {
        "linear": forecast_linear,
        "ets": forecast_ets,
        "arima": forecast_arima,
        "sf": forecast_sf,
    }
    fn = dispatch.get(method)
    if fn is None:
        raise ValueError(f"Unknown method: {method}")
    return fn(train_series, horizon=horizon)


def backtest_feature(
    series: np.ndarray,
    feature_name: str,
    *,
    horizons: List[int] = None,
    methods: List[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Walk-forward backtest for a single feature.

    Returns
    -------
    Nested dict: {method: {str(horizon): {metric: value}}}
    """
    horizons = horizons or DEFAULT_HORIZONS
    methods = methods or METHODS

    # Naive baseline errors for RMSSE (diff of consecutive values over full series)
    naive_errors = np.abs(np.diff(series))

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method in methods:
        results[method] = {}
        for horizon in horizons:
            window_metrics: List[Dict[str, float]] = []

            # Walk-forward: expand training, step by horizon (non-overlapping)
            origin = MIN_TRAIN_SIZE
            while origin + horizon <= len(series):
                train = series[:origin]
                actual = series[origin : origin + horizon]

                try:
                    pd_train = pd.Series(train, name=feature_name)
                    fc = _run_method(method, pd_train, horizon)

                    predicted = np.array(fc.values[:len(actual)])
                    lower = np.array(fc.lower_bound[:len(actual)]) if fc.lower_bound else None
                    upper = np.array(fc.upper_bound[:len(actual)]) if fc.upper_bound else None

                    if len(predicted) == len(actual):
                        m = compute_metrics(actual, predicted, lower, upper, naive_errors)
                        window_metrics.append(m)
                except Exception:
                    logger.debug(
                        "backtest window failed: method=%s feature=%s origin=%d",
                        method, feature_name, origin,
                    )

                origin += horizon  # non-overlapping step

            # Aggregate across windows
            if window_metrics:
                agg: Dict[str, float] = {}
                for key in window_metrics[0]:
                    vals = [wm[key] for wm in window_metrics if wm[key] is not None]
                    agg[key] = round(float(np.mean(vals)), 4) if vals else None
                agg["n_windows"] = len(window_metrics)
                results[method][str(horizon)] = agg
            else:
                results[method][str(horizon)] = {"n_windows": 0}

    return results


def backtest_session(
    telemetry_df: pd.DataFrame,
    *,
    features: List[str] = None,
    horizons: List[int] = None,
    methods: List[str] = None,
) -> Dict[str, Any]:
    """Run walk-forward backtest across all features for one session.

    Parameters
    ----------
    telemetry_df : DataFrame with telemetry columns (speed, rpm, etc.)
    features : which columns to test (defaults to DEFAULT_FEATURES)
    horizons : forecast horizons to test (defaults to [5, 10, 30])
    methods : forecast methods to test (defaults to METHODS)

    Returns
    -------
    dict with per-method/feature/horizon metrics + best_method per feature
    """
    features = features or DEFAULT_FEATURES
    horizons = horizons or DEFAULT_HORIZONS
    methods = methods or METHODS

    available = [f for f in features if f in telemetry_df.columns]
    if not available:
        return {"status": "no_features", "features_tested": []}

    # Results structure: {method: {feature: {horizon: metrics}}}
    results: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in methods}
    total_windows = 0

    for feat in available:
        series = telemetry_df[feat].dropna().values.astype(float)
        if len(series) < MIN_TRAIN_SIZE + min(horizons):
            logger.info("Skipping %s: only %d samples (need %d)", feat, len(series), MIN_TRAIN_SIZE + min(horizons))
            continue

        feat_results = backtest_feature(series, feat, horizons=horizons, methods=methods)

        for method in methods:
            results[method][feat] = feat_results.get(method, {})
            for h_key, metrics in feat_results.get(method, {}).items():
                if isinstance(metrics, dict):
                    total_windows += metrics.get("n_windows", 0)

    # Best method per feature (lowest RMSSE at horizon=10)
    best_method: Dict[str, str] = {}
    ref_horizon = "10"
    for feat in available:
        best_rmsse = float("inf")
        best = methods[0]
        for method in methods:
            h_metrics = results[method].get(feat, {}).get(ref_horizon, {})
            rmsse = h_metrics.get("rmsse")
            if rmsse is not None and rmsse < best_rmsse:
                best_rmsse = rmsse
                best = method
        best_method[feat] = best

    return {
        "features_tested": available,
        "methods_tested": methods,
        "horizons_tested": horizons,
        "total_windows": total_windows,
        "series_length": len(telemetry_df),
        "results": results,
        "best_method": best_method,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
```

**Step 2: Verify the module imports cleanly**

Run: `cd /home/pedrad/javier_project_folder/f1 && PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from omnianalytics.forecast_backtest import backtest_session, compute_metrics; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add omnisuitef1/omnianalytics/forecast_backtest.py
git commit -m "feat: add walk-forward forecast backtest engine with 6 metrics"
```

---

### Task 2: Add API Endpoints to `chat_server.py`

**Files:**
- Modify: `pipeline/chat_server.py` — add 3 endpoints after the existing backtest section (~line 5029)

**Step 1: Add the forecast backtest endpoints**

Add the following after the existing `@app.post("/api/local/backtest/kex")` handler (after line ~5060 in `chat_server.py`):

```python
# ── Forecast Backtest ────────────────────────────────────────────────

@app.post("/api/local/backtest/forecast/run")
async def run_forecast_backtest(
    session_key: int,
    driver_number: int,
    force: bool = False,
):
    """Run walk-forward forecast backtest for a single session+driver."""
    import asyncio
    from omnianalytics.forecast_backtest import backtest_session
    from omnianalytics.telemetry_loader import load_session_telemetry

    db = get_data_db()

    # Return cached unless forced
    if not force:
        stored = db["backtest_forecast_results"].find_one(
            {"session_key": session_key, "driver_number": driver_number}, {"_id": 0}
        )
        if stored and stored.get("results"):
            return {**stored, "from_cache": True}

    # Load telemetry
    telemetry_df = await asyncio.to_thread(load_session_telemetry, db, session_key, driver_number)
    if telemetry_df.empty:
        raise HTTPException(404, f"No telemetry for session {session_key} driver {driver_number}")

    # Run backtest
    result = await asyncio.to_thread(backtest_session, telemetry_df)
    if result.get("status") == "no_features":
        raise HTTPException(404, "No valid telemetry features found")

    # Store
    doc = {"session_key": session_key, "driver_number": driver_number, **result}
    db["backtest_forecast_results"].update_one(
        {"session_key": session_key, "driver_number": driver_number},
        {"$set": doc, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
        upsert=True,
    )

    return doc


@app.get("/api/local/backtest/forecast/results")
async def get_forecast_backtest_results(
    session_key: int,
    driver_number: int,
):
    """Get stored forecast backtest results."""
    db = get_data_db()
    doc = db["backtest_forecast_results"].find_one(
        {"session_key": session_key, "driver_number": driver_number}, {"_id": 0}
    )
    if not doc:
        raise HTTPException(404, "No forecast backtest results found")
    return doc


@app.post("/api/local/backtest/forecast/run-multi")
async def run_forecast_backtest_multi(body: dict):
    """Run forecast backtest across multiple sessions, aggregate results."""
    import asyncio
    from omnianalytics.forecast_backtest import backtest_session, METHODS, DEFAULT_HORIZONS, DEFAULT_FEATURES
    from omnianalytics.telemetry_loader import load_session_telemetry

    session_keys = body.get("session_keys", [])
    driver_number = body.get("driver_number")
    if not session_keys or not driver_number:
        raise HTTPException(400, "session_keys and driver_number required")

    db = get_data_db()
    all_results = []

    for sk in session_keys:
        # Check cache first
        stored = db["backtest_forecast_results"].find_one(
            {"session_key": sk, "driver_number": driver_number}, {"_id": 0}
        )
        if stored and stored.get("results"):
            all_results.append(stored)
            continue

        telemetry_df = await asyncio.to_thread(load_session_telemetry, db, sk, driver_number)
        if telemetry_df.empty:
            continue

        result = await asyncio.to_thread(backtest_session, telemetry_df)
        if result.get("status") == "no_features":
            continue

        doc = {"session_key": sk, "driver_number": driver_number, **result}
        db["backtest_forecast_results"].update_one(
            {"session_key": sk, "driver_number": driver_number},
            {"$set": doc, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
        all_results.append(doc)

    if not all_results:
        raise HTTPException(404, "No results generated for any session")

    # Aggregate: average metrics across sessions per (method, feature, horizon)
    import numpy as np
    agg_results = {m: {} for m in METHODS}
    for method in METHODS:
        for feat in DEFAULT_FEATURES:
            agg_results[method][feat] = {}
            for h in DEFAULT_HORIZONS:
                h_key = str(h)
                metric_lists: Dict[str, list] = {}
                for sr in all_results:
                    h_metrics = sr.get("results", {}).get(method, {}).get(feat, {}).get(h_key, {})
                    for mk, mv in h_metrics.items():
                        if mk == "n_windows" or mv is None:
                            continue
                        metric_lists.setdefault(mk, []).append(mv)
                agg = {k: round(float(np.mean(v)), 4) for k, v in metric_lists.items() if v}
                agg["n_sessions"] = len([
                    sr for sr in all_results
                    if sr.get("results", {}).get(method, {}).get(feat, {}).get(h_key, {}).get("n_windows", 0) > 0
                ])
                agg_results[method][feat][h_key] = agg

    # Best method per feature across sessions
    best_method = {}
    for feat in DEFAULT_FEATURES:
        best_rmsse = float("inf")
        best = METHODS[0]
        for method in METHODS:
            rmsse = agg_results[method].get(feat, {}).get("10", {}).get("rmsse")
            if rmsse is not None and rmsse < best_rmsse:
                best_rmsse = rmsse
                best = method
        best_method[feat] = best

    return {
        "sessions_evaluated": len(all_results),
        "session_keys": [r["session_key"] for r in all_results],
        "driver_number": driver_number,
        "results": agg_results,
        "best_method": best_method,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
```

**Step 2: Verify the server starts**

Run: `cd /home/pedrad/javier_project_folder/f1 && PYTHONPATH=omnisuitef1:pipeline:. timeout 5 python3 -c "from pipeline.chat_server import app; print('OK')" 2>&1 | head -5`
Expected: `OK` (or server startup logs)

**Step 3: Commit**

```bash
git add pipeline/chat_server.py
git commit -m "feat: add forecast backtest API endpoints (run, results, run-multi)"
```

---

### Task 3: Add "Forecast Validation" Tab to `BacktestView.tsx`

**Files:**
- Modify: `frontend/src/app/components/BacktestView.tsx`

**Step 1: Add TypeScript types and state**

Add after the existing `CaseStudy` interface (~line 100):

```typescript
/* ─── Forecast Backtest Types ──────────────────────────────────── */

interface ForecastMetrics {
  mae: number;
  rmse: number;
  mape: number;
  directional_acc: number;
  ci_coverage: number | null;
  rmsse: number;
  n_windows: number;
}

interface ForecastBacktestData {
  session_key: number;
  driver_number: number;
  features_tested: string[];
  methods_tested: string[];
  horizons_tested: number[];
  total_windows: number;
  series_length: number;
  results: Record<string, Record<string, Record<string, ForecastMetrics>>>;
  best_method: Record<string, string>;
  generated_at: string;
  from_cache?: boolean;
}
```

**Step 2: Add tab state + forecast state to main component**

In the `BacktestView` component, add state variables after the existing ones (~line 462):

```typescript
const [activeTab, setActiveTab] = useState<'race' | 'forecast'>('race');
const [forecastData, setForecastData] = useState<ForecastBacktestData | null>(null);
const [forecastLoading, setForecastLoading] = useState(false);
const [forecastError, setForecastError] = useState<string | null>(null);
```

**Step 3: Add tab selector in the header card**

After the header controls div and before the `{!data && !running && (` block, add a tab row:

```tsx
{/* ── Tab Selector ── */}
<div className="flex gap-1 mt-3 border-t border-border pt-3">
  <button
    onClick={() => setActiveTab('race')}
    className={`px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors ${
      activeTab === 'race'
        ? 'bg-primary/15 text-primary'
        : 'text-muted-foreground hover:text-foreground hover:bg-zinc-800'
    }`}
  >
    <div className="flex items-center gap-1.5">
      <Target className="w-3.5 h-3.5" />
      Race Outcome
    </div>
  </button>
  <button
    onClick={() => setActiveTab('forecast')}
    className={`px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors ${
      activeTab === 'forecast'
        ? 'bg-primary/15 text-primary'
        : 'text-muted-foreground hover:text-foreground hover:bg-zinc-800'
    }`}
  >
    <div className="flex items-center gap-1.5">
      <TrendingDown className="w-3.5 h-3.5" />
      Forecast Validation
    </div>
  </button>
</div>
```

**Step 4: Create the ForecastValidationTab component**

Add a new component before the main `BacktestView` export. This renders:
- Session/driver selector with run button
- Method comparison table (RMSSE color-coded, best highlighted)
- Horizon decay chart (line chart: accuracy vs horizon per method)
- Feature × Method heatmap (RMSSE as background color intensity)
- Best method recommendation cards per feature

Key details:
- Use `LineChart` from recharts (add to imports: `LineChart, Line, Legend`)
- Method colors: `{ linear: '#6b7280', ets: '#3b82f6', arima: '#8b5cf6', sf: '#f59e0b' }`
- RMSSE color scale: `<0.7` green, `0.7-1.0` yellow, `>1.0` red
- Table columns: Method | MAE | RMSE | MAPE | Dir. Acc | CI Coverage | RMSSE
- Feature selector dropdown (speed, rpm, throttle, etc.)
- Horizon selector: 5 | 10 | 30

**Step 5: Wrap existing content in tab conditional**

Wrap the existing `{m && data && ( ... )}` block in `{activeTab === 'race' && m && data && ( ... )}` and add `{activeTab === 'forecast' && <ForecastValidationTab />}`.

**Step 6: Verify frontend compiles**

Run: `cd /home/pedrad/javier_project_folder/f1/frontend && npx tsc --noEmit 2>&1 | head -20`
Expected: No errors (or only pre-existing ones)

**Step 7: Commit**

```bash
git add frontend/src/app/components/BacktestView.tsx
git commit -m "feat: add Forecast Validation tab to BacktestView"
```

---

### Task 4: Update `data_tracker.html`

**Files:**
- Modify: `data_tracker.html`

**Step 1: Add the new collection entry**

Find the backtest_results row in `data_tracker.html` and add a new row after it for `backtest_forecast_results`:

```html
<tr>
  <td>backtest_forecast_results</td>
  <td><span class="category derived">Derived</span></td>
  <td>-</td>
  <td>-</td>
  <td>session_key + driver_number (unique)</td>
  <td>chat_server forecast backtest endpoints</td>
  <td>chat_server GET endpoint</td>
  <td>BacktestView (Forecast Validation tab)</td>
  <td><span class="status active">Active</span></td>
</tr>
```

**Step 2: Update the total collection count** in the page header if present.

**Step 3: Commit**

```bash
git add data_tracker.html
git commit -m "docs: add backtest_forecast_results to data tracker"
```

---

## Verification

1. **Import test:** `PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from omnianalytics.forecast_backtest import backtest_session; print('OK')"`
2. **Server start:** `PYTHONPATH=omnisuitef1:pipeline:. python3 -m uvicorn pipeline.chat_server:app --host 0.0.0.0 --port 8300`
3. **Single session:** `curl -X POST 'http://localhost:8300/api/local/backtest/forecast/run?session_key=9573&driver_number=4'`
4. **Cached results:** `curl 'http://localhost:8300/api/local/backtest/forecast/results?session_key=9573&driver_number=4'`
5. **Frontend:** Navigate to Backtest view, click "Forecast Validation" tab, enter session/driver, run
6. **Multi-session:** POST to `/api/local/backtest/forecast/run-multi` with `{"session_keys": [9573, 9574], "driver_number": 4}`
