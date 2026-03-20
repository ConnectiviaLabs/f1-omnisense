"""Walk-forward (expanding window) forecast backtest engine.

Evaluates forecast_linear, forecast_ets, forecast_arima, and forecast_sf
on historical telemetry data to measure real out-of-sample accuracy.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from omnianalytics.forecast import (
    forecast_arima,
    forecast_ets,
    forecast_linear,
    forecast_sf,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

METHODS = ["linear", "ets", "arima", "sf"]
DEFAULT_HORIZONS = [5, 10, 30]
DEFAULT_FEATURES = ["speed", "rpm", "throttle", "brake", "n_gear", "drs"]
MIN_TRAIN_SIZE = 200

_METHOD_DISPATCH = {
    "linear": forecast_linear,
    "ets": forecast_ets,
    "arima": forecast_arima,
    "sf": forecast_sf,
}


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    naive_errors: np.ndarray,
) -> dict:
    """Compute 6 evaluation metrics for a single forecast window.

    Parameters
    ----------
    actual : array of true values
    predicted : array of forecast values (same length as actual)
    lower : array of lower confidence bounds (or None)
    upper : array of upper confidence bounds (or None)
    naive_errors : array of |diff(series)| from full training set for RMSSE

    Returns
    -------
    dict with keys: mae, rmse, mape, directional_acc, ci_coverage, rmsse
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted

    # MAE
    mae = float(np.mean(np.abs(errors)))

    # RMSE
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE — guard against division by zero
    abs_actual = np.abs(actual)
    nonzero = abs_actual > 1e-10
    if nonzero.any():
        mape = float(np.mean(np.abs(errors[nonzero]) / abs_actual[nonzero]) * 100)
    else:
        mape = None

    # Directional accuracy — % of steps with correct direction of change
    if len(actual) >= 2:
        actual_dir = np.diff(actual)
        pred_dir = np.diff(predicted)
        correct = np.sign(actual_dir) == np.sign(pred_dir)
        directional_acc = float(np.mean(correct))
    else:
        directional_acc = None

    # CI coverage — % of actual values within [lower, upper]
    if lower is not None and upper is not None:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        inside = (actual >= lower) & (actual <= upper)
        ci_coverage = float(np.mean(inside))
    else:
        ci_coverage = None

    # RMSSE — root mean squared scaled error vs naive baseline
    naive_errors = np.asarray(naive_errors, dtype=float)
    naive_mse = np.mean(naive_errors ** 2)
    if naive_mse > 1e-10:
        rmsse = float(np.sqrt(np.mean(errors ** 2) / naive_mse))
    else:
        rmsse = None

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 4) if mape is not None else None,
        "directional_acc": round(directional_acc, 4) if directional_acc is not None else None,
        "ci_coverage": round(ci_coverage, 4) if ci_coverage is not None else None,
        "rmsse": round(rmsse, 6) if rmsse is not None else None,
    }


# ── Single-feature backtest ──────────────────────────────────────────────

def backtest_feature(
    series: pd.Series,
    feature_name: str,
    *,
    horizons: List[int] = DEFAULT_HORIZONS,
    methods: List[str] = METHODS,
) -> dict:
    """Walk-forward backtest for one feature across methods and horizons.

    Parameters
    ----------
    series : pd.Series of numeric values (the full telemetry column)
    feature_name : name used for logging
    horizons : list of forecast horizons to evaluate
    methods : list of method keys (subset of METHODS)

    Returns
    -------
    {method: {str(horizon): {metrics + n_windows}}}
    """
    values = series.dropna().values.astype(float)
    n = len(values)

    # Naive errors computed per window (scoped to training set only)

    results: Dict[str, Dict[str, dict]] = {}

    for method in methods:
        if method not in _METHOD_DISPATCH:
            logger.warning("Unknown method '%s', skipping", method)
            continue

        fn = _METHOD_DISPATCH[method]
        results[method] = {}

        for horizon in horizons:
            window_metrics: List[dict] = []
            origin = MIN_TRAIN_SIZE

            while origin + horizon <= n:
                train = pd.Series(values[:origin], name=feature_name)
                actual = values[origin : origin + horizon]

                try:
                    result = fn(train, horizon=horizon)
                    predicted = np.array(result.values[:len(actual)])
                    lower = np.array(result.lower_bound[:len(actual)]) if len(result.lower_bound) > 0 else None
                    upper = np.array(result.upper_bound[:len(actual)]) if len(result.upper_bound) > 0 else None

                    # Naive errors scoped to training window only (no future leakage)
                    naive_errors = np.abs(np.diff(values[:origin]))
                    metrics = compute_metrics(actual, predicted, lower, upper, naive_errors)
                    window_metrics.append(metrics)
                except Exception as e:
                    logger.warning(
                        "Method %s failed at origin=%d horizon=%d for %s: %s",
                        method, origin, horizon, feature_name, e,
                    )

                origin += horizon

            if window_metrics:
                # Average metrics across windows
                avg: dict = {}
                for key in window_metrics[0]:
                    vals = [m[key] for m in window_metrics if m[key] is not None]
                    avg[key] = round(float(np.mean(vals)), 6) if vals else None
                avg["n_windows"] = len(window_metrics)
                results[method][str(horizon)] = avg
            else:
                results[method][str(horizon)] = {"n_windows": 0}

    return results


# ── Session-level backtest ───────────────────────────────────────────────

def backtest_session(
    telemetry_df: pd.DataFrame,
    *,
    features: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,
) -> dict:
    """Run walk-forward backtest across features for a telemetry session.

    Parameters
    ----------
    telemetry_df : DataFrame with telemetry columns
    features : columns to backtest (defaults to DEFAULT_FEATURES)
    horizons : forecast horizons (defaults to DEFAULT_HORIZONS)
    methods : forecast methods (defaults to METHODS)

    Returns
    -------
    dict with keys: features_tested, methods_tested, horizons_tested,
                    total_windows, series_length, results, best_method,
                    generated_at
    """
    features = features or DEFAULT_FEATURES
    horizons = horizons or DEFAULT_HORIZONS
    methods = methods or METHODS

    available = [f for f in features if f in telemetry_df.columns]
    if not available:
        return {
            "features_tested": [],
            "methods_tested": methods,
            "horizons_tested": horizons,
            "total_windows": 0,
            "series_length": len(telemetry_df),
            "results": {},
            "best_method": {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    all_results: Dict[str, dict] = {}
    total_windows = 0

    for feat in available:
        series = pd.to_numeric(telemetry_df[feat], errors="coerce").dropna()
        series.name = feat

        if len(series) < MIN_TRAIN_SIZE + min(horizons):
            logger.info(
                "Skipping %s: only %d points (need %d + %d)",
                feat, len(series), MIN_TRAIN_SIZE, min(horizons),
            )
            continue

        feat_results = backtest_feature(series, feat, horizons=horizons, methods=methods)
        all_results[feat] = feat_results

        # Count windows — once per (feature, horizon), not per method
        first_method = next(iter(feat_results.values()), {})
        for h_results in first_method.values():
            if isinstance(h_results, dict):
                total_windows += h_results.get("n_windows", 0)

    # Select best method per feature by lowest RMSSE at horizon=10 (or max if 10 absent)
    best_method: Dict[str, str] = {}
    ref_horizon = str(10 if 10 in horizons else max(horizons))
    if ref_horizon != "10":
        logger.warning("Horizon 10 not in horizons=%s; using %s for best_method selection", horizons, ref_horizon)
    for feat, feat_results in all_results.items():
        best_rmsse = float("inf")
        best_m = None
        for method, horizon_results in feat_results.items():
            h_data = horizon_results.get(ref_horizon, {})
            rmsse = h_data.get("rmsse")
            if rmsse is not None and rmsse < best_rmsse:
                best_rmsse = rmsse
                best_m = method
        if best_m is not None:
            best_method[feat] = best_m

    features_tested = [f for f in available if f in all_results]

    return {
        "features_tested": features_tested,
        "methods_tested": methods,
        "horizons_tested": horizons,
        "total_windows": total_windows,
        "series_length": len(telemetry_df),
        "results": all_results,
        "best_method": best_method,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
