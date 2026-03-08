# Cross-Model Feature Sharing — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate 30-50% compute waste by adding a MongoDB `feature_store` cache, consolidating duplicate anomaly ensemble code, and creating a shared telemetry loader.

**Architecture:** New `feature_store` MongoDB collection caches intermediate ML results keyed by `(session_key, driver_number, computation)`. Duplicate anomaly code consolidated into `omnianalytics/anomaly.py`. Shared telemetry loader replaces 5+ independent MongoDB queries. Cache-first logic wired into anomaly, forecast, and health pipelines.

**Tech Stack:** MongoDB, Python, pandas, scikit-learn, FastAPI

---

### Task 1: Create Feature Store Helper Module

**Files:**
- Create: `omnisuitef1/omnianalytics/feature_store.py`

**Context:** This module provides `get()` and `put()` helpers that read/write cached computation results from the `feature_store` MongoDB collection. Every downstream task depends on this module existing.

**Step 1: Create the feature store module**

```python
"""Feature store — session-pinned cache for intermediate ML results.

Keyed by (session_key, driver_number, computation). Historical race data
is immutable, so cached results never expire.

Collections cached:
  - anomaly_scores: 4-model ensemble results + severity per component
  - shap_features: Top-K SHAP features per component
  - forecasts:{column}:{horizon}: ETS/ARIMA/linear results
  - health_report: Full omnihealth assessment
  - telemetry_raw: Raw telemetry DataFrame (as records)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _make_key(session_key: int, driver_number: int, computation: str) -> Dict[str, Any]:
    """Build the unique compound key for a feature store document."""
    return {
        "session_key": session_key,
        "driver_number": driver_number,
        "computation": computation,
    }


def get(
    db,
    session_key: int,
    driver_number: int,
    computation: str,
) -> Optional[Dict[str, Any]]:
    """Read a cached result from the feature store.

    Returns the 'result' field if found, None otherwise.
    """
    if db is None:
        return None
    try:
        doc = db["feature_store"].find_one(
            _make_key(session_key, driver_number, computation),
            {"_id": 0, "result": 1},
        )
        return doc["result"] if doc else None
    except Exception as e:
        logger.debug("feature_store get failed: %s", e)
        return None


def put(
    db,
    session_key: int,
    driver_number: int,
    computation: str,
    result: Any,
) -> bool:
    """Write a computation result to the feature store.

    Uses upsert so re-running the same computation just overwrites.
    Returns True on success.
    """
    if db is None:
        return False
    try:
        key = _make_key(session_key, driver_number, computation)
        db["feature_store"].update_one(
            key,
            {"$set": {
                **key,
                "result": result,
                "created_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )
        return True
    except Exception as e:
        logger.warning("feature_store put failed: %s", e)
        return False


def ensure_indexes(db) -> None:
    """Create the compound unique index if it doesn't exist."""
    if db is None:
        return
    try:
        db["feature_store"].create_index(
            [("session_key", 1), ("driver_number", 1), ("computation", 1)],
            unique=True,
            name="feature_store_key",
        )
    except Exception as e:
        logger.debug("feature_store index creation: %s", e)
```

**Step 2: Verify the module imports**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from omnianalytics.feature_store import get, put, ensure_indexes; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add omnisuitef1/omnianalytics/feature_store.py
git commit -m "feat: add feature_store cache module for cross-model sharing"
```

---

### Task 2: Create Shared Telemetry Loader

**Files:**
- Create: `omnisuitef1/omnianalytics/telemetry_loader.py`

**Context:** Currently 5+ locations independently query MongoDB for car telemetry. This module provides a single `load_telemetry()` function that all agents and routers call. It caches raw telemetry in the feature store so repeated loads for the same session+driver hit cache.

**Step 1: Create the shared telemetry loader**

```python
"""Shared telemetry loader — single source of truth for car telemetry.

Replaces independent MongoDB queries in:
  - telemetry_anomaly.py::_load_telemetry()
  - predictive_maintenance.py::_load_telemetry()
  - omni_analytics_router.py::_build_dataset()
  - omni_health_router.py::_load_telemetry()

Caches raw telemetry in feature_store so repeated loads are free.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, Optional

import pandas as pd

from omnianalytics import feature_store

logger = logging.getLogger(__name__)


def load_session_telemetry(
    db,
    session_key: int,
    driver_number: Optional[int] = None,
) -> pd.DataFrame:
    """Load car telemetry for a session+driver from MongoDB.

    Check order:
    1. feature_store cache (fast)
    2. car_data collection (backfilled OpenF1)
    3. openf1_car_data collection (raw OpenF1 fallback)

    Caches the result in feature_store for future calls.
    """
    if db is None:
        return pd.DataFrame()

    cache_driver = driver_number or 0

    # 1. Check feature store cache
    cached = feature_store.get(db, session_key, cache_driver, "telemetry_raw")
    if cached is not None:
        logger.info("feature_store HIT: telemetry for session=%s driver=%s", session_key, driver_number)
        return pd.DataFrame(cached)

    # 2. Query MongoDB
    query: Dict[str, Any] = {"session_key": session_key}
    if driver_number:
        query["driver_number"] = driver_number

    # Try car_data first
    docs = list(db["car_data"].find(query, {"_id": 0}).sort("date", 1))

    if not docs:
        # Fallback to openf1_car_data
        docs = list(db["openf1_car_data"].find(query, {"_id": 0}).sort("date", 1))

    if not docs:
        logger.warning("No telemetry for session_key=%s driver=%s", session_key, driver_number)
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    logger.info("Loaded %d telemetry samples for session %s driver %s", len(df), session_key, driver_number)

    # 3. Cache in feature store (convert datetime columns to strings for BSON)
    try:
        records = df.copy()
        for col in records.select_dtypes(include=["datetime64", "datetimetz"]).columns:
            records[col] = records[col].astype(str)
        feature_store.put(db, session_key, cache_driver, "telemetry_raw", records.to_dict("records"))
    except Exception as e:
        logger.debug("Failed to cache telemetry: %s", e)

    return df
```

**Step 2: Verify the module imports**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from omnianalytics.telemetry_loader import load_session_telemetry; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add omnisuitef1/omnianalytics/telemetry_loader.py
git commit -m "feat: add shared telemetry loader with feature_store caching"
```

---

### Task 3: Make `pipeline/anomaly/ensemble.py` a Thin Wrapper

**Files:**
- Modify: `pipeline/anomaly/ensemble.py` (full file, lines 1-265)

**Context:** This file is 95% identical to `omnianalytics/anomaly.py`. We make it import from `omnianalytics` so there's one source of truth. The `AnomalyDetectionEnsemble` and `AnomalyStatistics` classes are kept as re-exports for backward compat (the telemetry agent and batch scripts import from here).

**Step 1: Replace the file contents**

Replace the entire file with:

```python
"""Anomaly Detection Ensemble for F1 Telemetry.

Thin wrapper — delegates to omnianalytics.anomaly for the actual implementation.
Kept for backward compatibility with pipeline scripts and agents that import
AnomalyDetectionEnsemble / AnomalyStatistics from this location.

The canonical implementation lives in omnisuitef1/omnianalytics/anomaly.py.
"""

from omnianalytics.anomaly import (  # noqa: F401
    AnomalyEnsemble,
    estimate_contamination,
    statistical_threshold,
    MODEL_WEIGHTS,
)

# ── Backward-compat aliases ─────────────────────────────────────────────
# Old code imports AnomalyDetectionEnsemble and AnomalyStatistics.
# Map them to the omnianalytics equivalents.


class AnomalyDetectionEnsemble:
    """Backward-compatible wrapper around omnianalytics.AnomalyEnsemble.

    Exposes the old DataFrame-based API (sklearn_models, run_autoencoder,
    run_anomaly_detection_models) by delegating to the new ensemble internally.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._ensemble = AnomalyEnsemble(random_state=random_state)

    def sklearn_models(self, raw_data, scaled_data):
        """Run IsolationForest + OneClassSVM + KNN (legacy API)."""
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.neighbors import NearestNeighbors

        scaled_data = scaled_data.fillna(scaled_data.median()).fillna(0)
        est_contam = estimate_contamination(scaled_data.values)

        algorithms = {
            "IsolationForest": IsolationForest(
                random_state=self.random_state,
                contamination=est_contam,
                n_jobs=-1,
                max_samples=min(256, len(scaled_data)),
            ),
            "OneClassSVM": SGDOneClassSVM(
                nu=min(0.5, max(0.01, est_contam)),
                random_state=self.random_state,
            ),
        }

        for name, algo in algorithms.items():
            try:
                preds = algo.fit_predict(scaled_data)
                scores = -algo.decision_function(scaled_data)
                raw_data[f"{name}_Anomaly"] = np.where(preds == -1, 1, 0)
                raw_data[f"{name}_AnomalyScore"] = scores
            except Exception:
                raw_data[f"{name}_Anomaly"] = 0
                raw_data[f"{name}_AnomalyScore"] = 0.0

        # KNN
        n = scaled_data.shape[0]
        k = max(2, min(5, n - 1))
        try:
            nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
            nn.fit(scaled_data)
            distances, _ = nn.kneighbors(scaled_data)
            avg_dist = distances[:, 1:].mean(axis=1)
            raw_data["KNN_AnomalyScore"] = avg_dist
            thr = statistical_threshold(avg_dist)
            raw_data["KNN_Anomaly"] = (avg_dist >= thr).astype(int)
        except Exception:
            raw_data["KNN_AnomalyScore"] = 0.0
            raw_data["KNN_Anomaly"] = 0

        # Normalize
        score_cols = [c for c in raw_data.columns if c.endswith("_AnomalyScore")]
        for col in score_cols:
            scores = raw_data[col].values
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            if p95 - p5 > 1e-10:
                raw_data[col] = np.clip(0.1 + 0.8 * (scores - p5) / (p95 - p5), 0, 1)
            else:
                raw_data[col] = 0.2

        return scaled_data, raw_data

    def run_autoencoder(self, scaled_data, raw_data):
        """PCA reconstruction (legacy API)."""
        import numpy as np
        from sklearn.decomposition import PCA

        try:
            n_components = max(2, min(scaled_data.shape[1] // 2, 10))
            pca = PCA(n_components=n_components, random_state=self.random_state)
            transformed = pca.fit_transform(scaled_data)
            reconstructed = pca.inverse_transform(transformed)
            error = np.mean((scaled_data.values - reconstructed) ** 2, axis=1)
            import pandas as pd
            threshold = float((error <= statistical_threshold(error)).mean())
            normalized = pd.Series(error).rank(pct=True).values
            normalized = np.clip(normalized, 0.0, 1.0)
            raw_data["Autoencoder_Anomaly"] = np.where(normalized > threshold, 1, 0)
            raw_data["Autoencoder_AnomalyScore"] = normalized
        except Exception:
            raw_data["Autoencoder_Anomaly"] = 0
            raw_data["Autoencoder_AnomalyScore"] = 0.0

        return scaled_data, raw_data

    def run_anomaly_detection_models(self, raw_data, scaled_data):
        """Full ensemble pipeline (legacy API)."""
        scaled_data, raw_data = self.sklearn_models(raw_data, scaled_data)
        scaled_data, raw_data = self.run_autoencoder(scaled_data, raw_data)
        return scaled_data, raw_data


class AnomalyStatistics:
    """Backward-compatible severity classifier."""

    def anomaly_insights(self, df):
        """Compute voting, weighted scores, severity levels (legacy API)."""
        import numpy as np

        anoms = [a for a in ["IsolationForest_Anomaly", "OneClassSVM_Anomaly", "Autoencoder_Anomaly"] if a in df.columns]
        scores = [s for s in ["IsolationForest_AnomalyScore", "OneClassSVM_AnomalyScore", "Autoencoder_AnomalyScore"] if s in df.columns]

        if not anoms or not scores:
            df["Anomaly_Level"] = "normal"
            return df

        anomaly_sum = df[anoms].sum(axis=1)
        score_mean = df[scores].mean(axis=1)
        score_std = df[scores].std(axis=1)

        if "KNN_Anomaly" in df.columns:
            anomaly_sum += df["KNN_Anomaly"]
        if "KNN_AnomalyScore" in df.columns:
            score_mean = (score_mean * len(scores) + df["KNN_AnomalyScore"]) / (len(scores) + 1)

        df["Voted_Anomaly"] = np.where(anomaly_sum >= 2, 1, 0)
        df["Anomaly_Score_Mean"] = score_mean
        df["Anomaly_Score_STD"] = score_std

        model_names = [a.replace("_Anomaly", "") for a in anoms]
        if "KNN_Anomaly" in df.columns:
            model_names.append("KNN")
        weights = np.array([MODEL_WEIGHTS.get(m, 0.5) for m in model_names])
        weights = weights / weights.sum()

        all_scores_cols = scores + (["KNN_AnomalyScore"] if "KNN_AnomalyScore" in df.columns else [])
        df["Reliability_Weighted_Score"] = df[all_scores_cols].values.dot(weights)
        df["Enhanced_Anomaly_Score"] = df["Reliability_Weighted_Score"] - (score_std * 0.1)
        df["Enhanced_Anomaly_Score"] += np.where(anomaly_sum == len(model_names), 0.05, 0)

        sq = df["Enhanced_Anomaly_Score"].quantile(q=[0.25, 0.50, 0.75, 0.85, 0.95])
        std_q = score_std.quantile(q=[0.25, 0.50, 0.75])

        df["Anomaly_Level"] = df.apply(
            lambda r: self._classify(r["Enhanced_Anomaly_Score"], r["Anomaly_Score_STD"], sq, std_q),
            axis=1,
        )
        return df

    @staticmethod
    def _classify(score, std_dev, sq, std_q):
        adj = 0.1 if std_dev >= std_q[0.75] else (-0.1 if std_dev <= std_q[0.25] else 0)
        if score >= sq[0.95] + adj:
            return "critical"
        elif score >= sq[0.85] + adj:
            return "high"
        elif score >= sq[0.75]:
            return "medium"
        elif score >= sq[0.50]:
            return "low"
        return "normal"


def severity_from_votes(vote_count: int, total_models: int = 4) -> str:
    """Data-driven severity based on model voting consensus."""
    if vote_count == 0:
        return "normal"
    elif vote_count == 1:
        return "low"
    elif vote_count == 2:
        return "medium"
    elif vote_count == 3:
        return "high"
    else:
        return "critical"
```

**Step 2: Verify backward-compat imports still work**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "
from pipeline.anomaly.ensemble import AnomalyDetectionEnsemble, AnomalyStatistics, severity_from_votes
print('AnomalyDetectionEnsemble:', AnomalyDetectionEnsemble)
print('AnomalyStatistics:', AnomalyStatistics)
print('severity_from_votes:', severity_from_votes('critical'))
print('OK')
"
```
Expected: All imports succeed, `OK` printed.

**Step 3: Commit**

```bash
git add pipeline/anomaly/ensemble.py
git commit -m "refactor: make pipeline/anomaly/ensemble.py a thin wrapper over omnianalytics"
```

---

### Task 4: Wire Shared Loader into Telemetry Anomaly Agent

**Files:**
- Modify: `omnisuitef1/omniagents/agents/telemetry_anomaly.py`

**Context:** Replace the agent's private `_load_telemetry()` with the shared loader. The agent currently imports `AnomalyDetectionEnsemble` from `pipeline.anomaly.ensemble` — this still works (thin wrapper), but the telemetry loading now goes through the cached shared loader.

**Step 1: Add shared loader import and remove private method**

At the top of the file (line 17, after the existing imports), add:
```python
from omnianalytics.telemetry_loader import load_session_telemetry
```

**Step 2: Replace `_load_telemetry` call in `run_analysis`**

Change lines 47-50 from:
```python
        telemetry_df = await asyncio.to_thread(
            self._load_telemetry, session_key, driver_number
        )
```
to:
```python
        telemetry_df = await asyncio.to_thread(
            load_session_telemetry, self._db, session_key, driver_number
        )
```

**Step 3: Delete the `_load_telemetry` method**

Remove the entire `_load_telemetry` method (lines 109-135).

**Step 4: Verify import**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from omniagents.agents.telemetry_anomaly import TelemetryAnomalyAgent; print('OK')"
```
Expected: `OK`

**Step 5: Commit**

```bash
git add omnisuitef1/omniagents/agents/telemetry_anomaly.py
git commit -m "refactor: telemetry agent uses shared loader instead of private _load_telemetry"
```

---

### Task 5: Wire Shared Loader into Predictive Maintenance Agent

**Files:**
- Modify: `omnisuitef1/omniagents/agents/predictive_maintenance.py`

**Context:** Same pattern as Task 4 — replace private `_load_telemetry()` with shared loader.

**Step 1: Add shared loader import**

At the top of the file (line 17, after existing imports), add:
```python
from omnianalytics.telemetry_loader import load_session_telemetry
```

**Step 2: Replace `_load_telemetry` call in `run_analysis`**

Change lines 73-76 from:
```python
        telemetry_df = await asyncio.to_thread(
            self._load_telemetry, session_key, driver_number
        )
```
to:
```python
        telemetry_df = await asyncio.to_thread(
            load_session_telemetry, self._db, session_key, driver_number
        )
```

**Step 3: Delete the `_load_telemetry` method**

Remove the entire `_load_telemetry` method (lines 195-221).

**Step 4: Verify import**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from omniagents.agents.predictive_maintenance import PredictiveMaintenanceAgent; print('OK')"
```
Expected: `OK`

**Step 5: Commit**

```bash
git add omnisuitef1/omniagents/agents/predictive_maintenance.py
git commit -m "refactor: predictive maintenance agent uses shared loader"
```

---

### Task 6: Add Cache-First Logic to Anomaly Ensemble

**Files:**
- Modify: `omnisuitef1/omnianalytics/anomaly.py` (lines 71-184)

**Context:** Add optional `session_key` and `driver_number` params to `AnomalyEnsemble.run()`. When provided, check feature store before computing. Cache results after computing.

**Step 1: Add feature_store import at the top**

After line 8 (`from typing import ...`), add:
```python
from omnianalytics import feature_store
```

**Step 2: Add cache params to `run()` method signature**

Change the method signature (lines 71-79) from:
```python
    def run(
        self,
        dataset: TabularDataset,
        columns: Optional[List[str]] = None,
        *,
        weights: Optional[Dict[str, float]] = None,
        explain_critical: bool = True,
        top_k_features: int = 3,
    ) -> AnomalyResult:
```
to:
```python
    def run(
        self,
        dataset: TabularDataset,
        columns: Optional[List[str]] = None,
        *,
        weights: Optional[Dict[str, float]] = None,
        explain_critical: bool = True,
        top_k_features: int = 3,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None,
        db=None,
    ) -> AnomalyResult:
```

**Step 3: Add cache-check at the start of `run()` body**

After the docstring (line 84), before `from omnidata.profiler ...` (line 85), add:
```python
        # ── Feature store cache check ──
        if session_key is not None and driver_number is not None and db is not None:
            cached = feature_store.get(db, session_key, driver_number, "anomaly_scores")
            if cached is not None:
                logger.info("feature_store HIT: anomaly_scores for session=%s driver=%s", session_key, driver_number)
                return AnomalyResult.from_dict(cached)
```

**Step 4: Add cache-write at the end of `run()`, before the return**

Just before the final `return AnomalyResult(...)` (line 176), add:
```python
        # ── Cache result in feature store ──
        if session_key is not None and driver_number is not None and db is not None:
            try:
                result_obj = AnomalyResult(
                    scores=scores,
                    contamination_estimate=round(contam, 4),
                    threshold=round(float(threshold), 4),
                    anomaly_count=int(anomaly_flags.sum()),
                    total_rows=n_rows,
                    severity_distribution=severity_dist,
                    model_weights={m: round(w.get(m, 0.5), 2) for m in model_names},
                )
                feature_store.put(db, session_key, driver_number, "anomaly_scores", result_obj.to_dict())
            except Exception as e:
                logger.debug("Failed to cache anomaly_scores: %s", e)
```

Then return the already-constructed `result_obj` if caching succeeded, or build a new one:

Actually, to keep it simple, just build the result once and cache it:

Replace the existing return block (lines 176-184) with:
```python
        result = AnomalyResult(
            scores=scores,
            contamination_estimate=round(contam, 4),
            threshold=round(float(threshold), 4),
            anomaly_count=int(anomaly_flags.sum()),
            total_rows=n_rows,
            severity_distribution=severity_dist,
            model_weights={m: round(w.get(m, 0.5), 2) for m in model_names},
        )

        # ── Cache result in feature store ──
        if session_key is not None and driver_number is not None and db is not None:
            try:
                feature_store.put(db, session_key, driver_number, "anomaly_scores", result.to_dict())
            except Exception as e:
                logger.debug("Failed to cache anomaly_scores: %s", e)

        return result
```

**Step 5: Add `from_dict` classmethod to AnomalyResult if it doesn't exist**

Check `omnisuitef1/omnianalytics/_types.py` for `AnomalyResult`. If it doesn't have `from_dict()`, add one. If it already has `to_dict()`, model `from_dict()` as its inverse.

**Step 6: Verify**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "
from omnianalytics.anomaly import AnomalyEnsemble
import inspect
sig = inspect.signature(AnomalyEnsemble.run)
assert 'session_key' in sig.parameters
assert 'db' in sig.parameters
print('OK')
"
```
Expected: `OK`

**Step 7: Commit**

```bash
git add omnisuitef1/omnianalytics/anomaly.py omnisuitef1/omnianalytics/_types.py
git commit -m "feat: add cache-first logic to AnomalyEnsemble.run()"
```

---

### Task 7: Add Cache-First Logic to Forecast Functions

**Files:**
- Modify: `omnisuitef1/omnianalytics/forecast.py` (lines 367-414)

**Context:** The `forecast()` dispatcher function gets optional `session_key`, `driver_number`, and `db` params. When provided, it checks the feature store before computing. Cache key includes column name and horizon for granularity.

**Step 1: Add feature_store import**

After line 7 (`import pandas as pd`), add:
```python
from omnianalytics import feature_store
```

**Step 2: Modify `forecast()` signature**

Change lines 367-373 from:
```python
def forecast(
    dataset: TabularDataset,
    column: str,
    *,
    horizon: int = 10,
    method: str = "auto",
) -> ForecastResult:
```
to:
```python
def forecast(
    dataset: TabularDataset,
    column: str,
    *,
    horizon: int = 10,
    method: str = "auto",
    session_key: Optional[int] = None,
    driver_number: Optional[int] = None,
    db=None,
) -> ForecastResult:
```

**Step 3: Add cache check at the start of `forecast()` body**

After the docstring (line 379), before `if column not in dataset.df.columns:` (line 380), add:
```python
    # ── Feature store cache check ──
    cache_computation = f"forecast:{column}:{horizon}:{method}"
    if session_key is not None and driver_number is not None and db is not None:
        cached = feature_store.get(db, session_key, driver_number, cache_computation)
        if cached is not None:
            logger.info("feature_store HIT: %s for session=%s driver=%s", cache_computation, session_key, driver_number)
            return ForecastResult.from_dict(cached)
```

**Step 4: Add cache write before each return**

At the end of the function, after computing the result but before returning, wrap the return. The simplest approach: capture the result variable, cache it, then return. Change the function to:

After the existing dispatch logic (the if/elif chain), the function returns `forecast_sf(...)`, `forecast_arima(...)`, etc. Instead of returning directly, capture and cache:

Replace the entire dispatch section (lines 397-414) with:
```python
    if method == "arima":
        result = forecast_arima(series, horizon=horizon, timestamps=timestamps)
    elif method == "linear":
        result = forecast_linear(series, horizon=horizon, timestamps=timestamps)
    elif method == "ets":
        result = forecast_ets(series, horizon=horizon, timestamps=timestamps)
    elif method == "statsforecast" or method == "sf":
        result = forecast_sf(series, horizon=horizon, timestamps=timestamps)
    elif method == "lightgbm":
        feature_cols = [c for c in dataset.profile.metric_cols if c != column]
        if feature_cols:
            result = forecast_lightgbm(
                dataset.df, column, feature_cols, horizon=horizon, timestamps=timestamps,
            )
        else:
            result = forecast_linear(series, horizon=horizon, timestamps=timestamps)
    else:
        # auto: StatsForecast ensemble
        result = forecast_sf(series, horizon=horizon, timestamps=timestamps)

    # ── Cache result in feature store ──
    if session_key is not None and driver_number is not None and db is not None:
        try:
            feature_store.put(db, session_key, driver_number, cache_computation, result.to_dict())
        except Exception as e:
            logger.debug("Failed to cache %s: %s", cache_computation, e)

    return result
```

**Step 5: Add `from_dict` to `ForecastResult` if needed**

Check `omnisuitef1/omnianalytics/_types.py` for `ForecastResult`. Add `from_dict()` if missing.

**Step 6: Verify**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "
from omnianalytics.forecast import forecast
import inspect
sig = inspect.signature(forecast)
assert 'session_key' in sig.parameters
assert 'db' in sig.parameters
print('OK')
"
```
Expected: `OK`

**Step 7: Commit**

```bash
git add omnisuitef1/omnianalytics/forecast.py omnisuitef1/omnianalytics/_types.py
git commit -m "feat: add cache-first logic to forecast() dispatcher"
```

---

### Task 8: Wire Cache Into omnihealth.assess()

**Files:**
- Modify: `omnisuitef1/omnihealth/__init__.py` (lines 65-220)

**Context:** `assess()` is the one-call pipeline (health scoring → risk → scheduling). Add optional `session_key`, `driver_number`, `db` params. When provided, cache the full `HealthReport` and also pass cache params down to `assess_feature_risk()`.

**Step 1: Modify `assess()` signature**

Change lines 65-73 from:
```python
def assess(
    data: pd.DataFrame,
    component_map: Dict[str, List[str]],
    *,
    horizon: int = 10,
    forecast_method: str = "auto",
    include_schedule: bool = True,
    include_timeseries: bool = False,
    anomaly_weights: Optional[Dict[str, float]] = None,
) -> HealthReport:
```
to:
```python
def assess(
    data: pd.DataFrame,
    component_map: Dict[str, List[str]],
    *,
    horizon: int = 10,
    forecast_method: str = "auto",
    include_schedule: bool = True,
    include_timeseries: bool = False,
    anomaly_weights: Optional[Dict[str, float]] = None,
    session_key: Optional[int] = None,
    driver_number: Optional[int] = None,
    db=None,
) -> HealthReport:
```

**Step 2: Add cache check at the start**

After the docstring (line 90), before `# Step 1: Health scoring` (line 92), add:
```python
    # ── Feature store cache check ──
    if session_key is not None and driver_number is not None and db is not None:
        from omnianalytics import feature_store
        cached = feature_store.get(db, session_key, driver_number, "health_report")
        if cached is not None:
            import logging
            logging.getLogger(__name__).info(
                "feature_store HIT: health_report for session=%s driver=%s", session_key, driver_number
            )
            return HealthReport.from_dict(cached)
```

**Step 3: Add cache write before the final return**

Just before `return HealthReport(...)` (line 213), add:
```python
    report = HealthReport(
        components=health_scores,
        risk_assessments=risk_assessments,
        schedule=schedule,
        overall_health=round(overall_health, 1),
        overall_risk=overall_risk,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    # ── Cache result in feature store ──
    if session_key is not None and driver_number is not None and db is not None:
        from omnianalytics import feature_store
        try:
            feature_store.put(db, session_key, driver_number, "health_report", report.to_dict())
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Failed to cache health_report: %s", e)

    return report
```

(And remove the old `return HealthReport(...)` block.)

**Step 4: Verify**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "
from omnihealth import assess
import inspect
sig = inspect.signature(assess)
assert 'session_key' in sig.parameters
print('OK')
"
```
Expected: `OK`

**Step 5: Commit**

```bash
git add omnisuitef1/omnihealth/__init__.py
git commit -m "feat: add cache-first logic to omnihealth.assess()"
```

---

### Task 9: Wire Cache Into Predictive Maintenance Agent's omnihealth Call

**Files:**
- Modify: `omnisuitef1/omniagents/agents/predictive_maintenance.py` (lines 96-106)

**Context:** The predictive maintenance agent calls `omnihealth.assess()`. Now pass `session_key`, `driver_number`, and `db` so it uses the cache.

**Step 1: Pass cache params to `run_health_pipeline`**

Change lines 99-106 from:
```python
            health_report = await asyncio.to_thread(
                run_health_pipeline,
                telemetry_df[available].fillna(0),
                filtered_map,
                horizon=30,
                include_schedule=True,
                include_timeseries=True,
            )
```
to:
```python
            health_report = await asyncio.to_thread(
                run_health_pipeline,
                telemetry_df[available].fillna(0),
                filtered_map,
                horizon=30,
                include_schedule=True,
                include_timeseries=True,
                session_key=session_key,
                driver_number=driver_number,
                db=self._db,
            )
```

**Step 2: Verify**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from omniagents.agents.predictive_maintenance import PredictiveMaintenanceAgent; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add omnisuitef1/omniagents/agents/predictive_maintenance.py
git commit -m "feat: predictive maintenance agent passes cache params to omnihealth"
```

---

### Task 10: Ensure Feature Store Index on Server Startup

**Files:**
- Modify: `pipeline/chat_server.py`

**Context:** When the server starts, ensure the `feature_store` compound index exists so lookups are fast.

**Step 1: Find the startup event in `chat_server.py`**

Look for the `@app.on_event("startup")` handler or the place where MongoDB is initialized. Add the index creation there.

**Step 2: Add index creation**

After the MongoDB client is initialized (look for `get_db()` or similar), add:
```python
# Ensure feature_store index
try:
    from omnianalytics.feature_store import ensure_indexes
    ensure_indexes(db)
except Exception:
    pass
```

**Step 3: Verify server starts**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "
from omnianalytics.feature_store import ensure_indexes
print('ensure_indexes imported OK')
"
```
Expected: `OK`

**Step 4: Commit**

```bash
git add pipeline/chat_server.py
git commit -m "feat: ensure feature_store index on server startup"
```

---

### Task 11: Add `from_dict` Methods to Types

**Files:**
- Modify: `omnisuitef1/omnianalytics/_types.py`
- Modify: `omnisuitef1/omnihealth/_types.py`

**Context:** The cache stores results as dicts (via `to_dict()`). When reading from cache, we need `from_dict()` class methods to reconstruct the typed objects. Check if `to_dict()` exists on `AnomalyResult`, `ForecastResult`, and `HealthReport`. If so, add `from_dict()` as its inverse.

**Step 1: Read the types files to understand the current structure**

Check what `to_dict()` returns for each type, then write `from_dict()` that reconstructs the object.

**Step 2: Add `from_dict()` to `AnomalyResult`**

```python
@classmethod
def from_dict(cls, d: dict) -> "AnomalyResult":
    scores = [AnomalyScore(**s) if isinstance(s, dict) else s for s in d.get("scores", [])]
    return cls(
        scores=scores,
        contamination_estimate=d.get("contamination_estimate", 0),
        threshold=d.get("threshold", 0),
        anomaly_count=d.get("anomaly_count", 0),
        total_rows=d.get("total_rows", 0),
        severity_distribution=d.get("severity_distribution", {}),
        model_weights=d.get("model_weights", {}),
    )
```

**Step 3: Add `from_dict()` to `ForecastResult`**

Model after the existing fields in ForecastResult's `__init__`.

**Step 4: Add `from_dict()` to `HealthReport`**

Model after HealthReport's `__init__` — will need to reconstruct nested `HealthScore`, `RiskAssessment`, `MaintenanceSchedule` objects.

**Step 5: Verify**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "
from omnianalytics._types import AnomalyResult, ForecastResult
from omnihealth._types import HealthReport
print('AnomalyResult.from_dict:', hasattr(AnomalyResult, 'from_dict'))
print('ForecastResult.from_dict:', hasattr(ForecastResult, 'from_dict'))
print('HealthReport.from_dict:', hasattr(HealthReport, 'from_dict'))
"
```
Expected: All `True`

**Step 6: Commit**

```bash
git add omnisuitef1/omnianalytics/_types.py omnisuitef1/omnihealth/_types.py
git commit -m "feat: add from_dict() to AnomalyResult, ForecastResult, HealthReport"
```

---

### Task 12: Update data_tracker.html with New Collection

**Files:**
- Modify: `data_tracker.html`

**Context:** The project tracks all MongoDB collections in `data_tracker.html`. Add `feature_store` to the inventory.

**Step 1: Add `feature_store` entry**

Find the collection table in `data_tracker.html` and add a row:
- Collection: `feature_store`
- Category: System
- Writers: `omnianalytics/anomaly.py`, `omnianalytics/forecast.py`, `omnihealth/__init__.py`, `omnianalytics/telemetry_loader.py`
- Readers: Same (cache consumers)
- Status: Active

**Step 2: Commit**

```bash
git add data_tracker.html
git commit -m "docs: add feature_store collection to data tracker"
```

---

## Verification

After all tasks are complete:

1. **Import chain:**
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "
from omnianalytics.feature_store import get, put, ensure_indexes
from omnianalytics.telemetry_loader import load_session_telemetry
from omnianalytics.anomaly import AnomalyEnsemble
from omnianalytics.forecast import forecast
from omnihealth import assess
from pipeline.anomaly.ensemble import AnomalyDetectionEnsemble, AnomalyStatistics
from omniagents.agents.telemetry_anomaly import TelemetryAnomalyAgent
from omniagents.agents.predictive_maintenance import PredictiveMaintenanceAgent
print('ALL IMPORTS OK')
"
```

2. **Server starts:**
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -m uvicorn pipeline.chat_server:app --host 0.0.0.0 --port 8300
```

3. **Run-all agents** — verify the second run is faster (cache hits):
```bash
# First run: compute + cache
curl -X POST http://localhost:8300/api/omni/agents/run-all -H 'Content-Type: application/json' -d '{"session_key": 9573, "driver_number": 4}'

# Second run: should see "feature_store HIT" in logs
curl -X POST http://localhost:8300/api/omni/agents/run-all -H 'Content-Type: application/json' -d '{"session_key": 9573, "driver_number": 4}'
```

4. **Check feature_store collection:**
```bash
mongosh marip_f1 --eval "db.feature_store.find({}, {result: 0}).pretty()"
```
