# Forecast Backtest & Validation System — Design

## Goal

Build a walk-forward forecast backtesting system that validates all 5 forecast methods (linear, ETS, ARIMA, StatsForecast, LightGBM) against real F1 telemetry data, producing per-method, per-feature, per-horizon accuracy metrics. Results surface in a new "Forecast Validation" tab in the existing BacktestView.

## Architecture

**Core module:** `omnisuitef1/omnianalytics/forecast_backtest.py`

Pure computation — takes a pandas Series, runs walk-forward validation across methods and horizons, returns structured results. No MongoDB dependency in the core.

**Data flow:**

```
MongoDB (car_data) → telemetry_loader → DataFrame
    → forecast_backtest.backtest_feature(series, horizons, methods)
        → for each window origin:
            train = series[:origin]
            actual = series[origin:origin+horizon]
            → for each method: forecast(train, horizon) → compare to actual
        → aggregate metrics across all windows
    → return per-method/feature/horizon metrics
→ Store in backtest_forecast_results collection
→ Serve via /api/local/backtest/forecast/*
→ Render in BacktestView "Forecast Validation" tab
```

**Key principle:** The backtest calls `forecast_linear()`, `forecast_ets()`, `forecast_arima()`, `forecast_sf()` directly — the exact same functions the production pipeline uses.

## Walk-Forward Engine

**Parameters:**
- Minimum training size: 200 samples
- Step size: `horizon` (non-overlapping windows)
- Horizons: `[5, 10, 30]`
- Methods: `["linear", "ets", "arima", "sf"]` (LightGBM skipped — needs multivariate input)
- Features: `["speed", "rpm", "throttle", "brake", "n_gear", "drs"]`

**Metrics per (method, feature, horizon):**

| Metric | Description |
|---|---|
| MAE | Mean absolute error in original units |
| RMSE | Root mean squared error |
| MAPE | Mean absolute percentage error (scale-independent) |
| Directional accuracy | % of windows where forecast correctly predicted direction |
| CI coverage | % of actual values within 95% confidence bounds |
| RMSSE | Root mean squared scaled error vs naive baseline. <1 = beats naive |

**Aggregation:** Per-session results averaged across walk-forward windows. Multi-session results averaged across sessions.

**Best method selection:** Rank by RMSSE per feature. Lowest RMSSE wins.

## API Endpoints

```
POST /api/local/backtest/forecast/run
  ?session_key=9573&driver_number=4&force=false
  Runs walk-forward backtest, stores and returns results.

GET /api/local/backtest/forecast/results
  ?session_key=9573&driver_number=4
  Returns stored results (no recompute).

POST /api/local/backtest/forecast/run-multi
  Body: {"session_keys": [9573, 9574, ...], "driver_number": 4}
  Cross-session aggregate with robust method rankings.
```

## MongoDB Collection

**`backtest_forecast_results`** — keyed by `(session_key, driver_number)`.

```json
{
    "session_key": 9573,
    "driver_number": 4,
    "features_tested": ["speed", "rpm", "throttle", "brake", "n_gear", "drs"],
    "methods_tested": ["linear", "ets", "arima", "sf"],
    "horizons_tested": [5, 10, 30],
    "total_windows": 42,
    "series_length": 2100,
    "results": {
        "ets": {
            "speed": {
                "5":  {"mae": 2.1, "rmse": 3.4, "mape": 1.2, "directional_acc": 0.78, "ci_coverage": 0.93, "rmsse": 0.65},
                "10": {},
                "30": {}
            }
        }
    },
    "best_method": {
        "speed": "sf",
        "rpm": "ets"
    },
    "generated_at": "2026-03-08T..."
}
```

## Frontend

New "Forecast Validation" tab in `BacktestView.tsx`:

- **Method comparison table** — MAE/RMSE/RMSSE per method, color-coded winners
- **Horizon decay chart** — accuracy vs forecast distance, line per method
- **Feature heatmap** — features × methods, RMSSE as color intensity
- **CI coverage gauge** — calibration check per method
- **Best method recommendation** — card per feature showing winner + metrics

## Files

### New
- `omnisuitef1/omnianalytics/forecast_backtest.py` — walk-forward engine + metrics
- (Frontend tab added inline to existing BacktestView.tsx)

### Modified
- `pipeline/chat_server.py` — 3 new endpoints under `/api/local/backtest/forecast/`
- `frontend/src/app/components/BacktestView.tsx` — new tab with forecast validation charts
- `data_tracker.html` — add `backtest_forecast_results` collection