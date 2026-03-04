"""Risk assessment from forecasts + degradation detection.

Ports DataSense _assess_risk() tolerance formula and adds degradation detection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from omnihealth._types import (
    RiskAssessment,
    RiskLevel,
    TimeSeriesAnalysis,
    TrendDirection,
)

logger = logging.getLogger(__name__)


# ── Core risk formula ────────────────────────────────────────────────────────

def assess_risk(
    current_value: float,
    forecast_value: float,
    std_dev: float,
    series: Optional[pd.Series] = None,
) -> RiskLevel:
    """Data-driven risk assessment using historical delta percentiles.

    If series is provided, thresholds are derived from the actual distribution
    of period-over-period changes (90th and 99th percentiles). This handles
    skewed, heavy-tailed, and non-normal distributions correctly.

    Falls back to std_dev-based tolerance when series is unavailable or too short.
    """
    delta = abs(forecast_value - current_value)

    # Data-driven: use actual delta distribution percentiles
    if series is not None and len(series) >= 10:
        hist_deltas = np.abs(np.diff(series.values))
        if len(hist_deltas) > 0:
            p90 = float(np.percentile(hist_deltas, 90))
            p99 = float(np.percentile(hist_deltas, 99))
            # Ensure minimums to avoid zero thresholds on constant data
            p90 = max(p90, 1e-6)
            p99 = max(p99, p90 * 1.5)

            if delta > p99:
                return RiskLevel.HIGH
            elif delta > p90:
                return RiskLevel.MEDIUM
            return RiskLevel.LOW

    # Fallback: std_dev-based tolerance
    if std_dev > 0:
        tolerance = std_dev
    else:
        tolerance = max(abs(current_value) * 0.05, 1e-3)

    if delta > 2 * tolerance:
        return RiskLevel.HIGH
    elif delta > tolerance:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# ── Auto horizon selection ───────────────────────────────────────────────────

def _auto_horizons(series: pd.Series) -> List[int]:
    """Derive 3 forecast horizons from the data's temporal structure.

    - Short: autocorrelation decay length (how far the signal is predictable)
    - Medium: detected seasonal period or 3x short
    - Long: 2x medium, capped at len//3

    Returns sorted [short, medium, long] with duplicates removed.
    """
    values = series.values
    n = len(values)
    max_horizon = max(n // 3, 3)

    # Short: autocorrelation decay
    short = _auto_window(values)  # reuse from degradation — ACF < 0.5

    # Medium: try to detect dominant period from ACF peaks
    medium = short * 3
    if n >= 20:
        centered = values - np.mean(values)
        var = np.var(values)
        if var > 1e-12:
            max_lag = min(n // 2, 100)
            acf_vals = []
            for lag in range(1, max_lag):
                acf = float(np.mean(centered[:n - lag] * centered[lag:])) / var
                acf_vals.append(acf)
            # Find first ACF peak after decay (seasonal period)
            if len(acf_vals) >= 3:
                for i in range(1, len(acf_vals) - 1):
                    if acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1]:
                        if acf_vals[i] > 0.2:  # significant peak
                            medium = i + 1
                            break

    # Long: 2x medium
    long = medium * 2

    # Clamp and deduplicate
    horizons = sorted(set([
        max(2, min(short, max_horizon)),
        max(3, min(medium, max_horizon)),
        max(5, min(long, max_horizon)),
    ]))

    # Ensure exactly 3 — spread if duplicates collapsed
    if len(horizons) == 1:
        h = horizons[0]
        horizons = sorted(set([h, min(h * 2, max_horizon), min(h * 4, max_horizon)]))
    if len(horizons) == 2:
        horizons.append(min(horizons[-1] * 2, max_horizon))
        horizons = sorted(set(horizons))

    return horizons[:3]


# ── Feature-level risk assessment ────────────────────────────────────────────

def assess_feature_risk(
    dataset: Any,
    column: str,
    *,
    horizon: int = 10,
    method: str = "auto",
    include_analysis: bool = True,
) -> RiskAssessment:
    """Full risk pipeline for one feature with multi-horizon forecasting.

    Forecasts at 3 data-driven horizons (short/medium/long derived from
    autocorrelation and seasonality). The worst-case risk across horizons
    determines the final risk level. All horizon results are stored.

    Parameters
    ----------
    dataset : TabularDataset from omnidata.
    column : Column name to assess.
    horizon : Base forecast horizon (used as fallback if auto fails).
    method : Forecasting method ("auto", "arima", "linear", "lightgbm").
    include_analysis : If True, include full time series analysis.

    Returns
    -------
    RiskAssessment with worst-case risk across horizons.
    """
    from omnianalytics.forecast import forecast as run_forecast

    series = pd.to_numeric(dataset.df[column], errors="coerce").dropna()
    current = float(series.iloc[-1]) if len(series) > 0 else 0.0

    # Determine 3 horizons from data
    if len(series) >= 10:
        horizons = _auto_horizons(series)
    else:
        horizons = [max(2, horizon // 3), horizon, min(horizon * 2, max(horizon, 5))]

    # Run forecast at each horizon
    _RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
    horizon_results: List[Dict[str, Any]] = []
    worst_risk = RiskLevel.LOW
    worst_result = None  # track the horizon with worst risk

    for h in horizons:
        try:
            forecast_result = run_forecast(dataset, column, horizon=h, method=method)
            fval = float(forecast_result.values[-1]) if forecast_result.values else current
            lo = float(forecast_result.lower_bound[-1]) if forecast_result.lower_bound else fval
            hi = float(forecast_result.upper_bound[-1]) if forecast_result.upper_bound else fval
            sd = (hi - lo) / (2 * 1.96) if hi != lo else float(series.std())
            risk = assess_risk(current, fval, sd, series=series)

            if abs(current) > 1e-10:
                t_pct = ((fval - current) / abs(current)) * 100
            else:
                t_pct = 0.0

            hr = {
                "horizon": h,
                "forecast_value": round(fval, 4),
                "risk_level": risk.value,
                "trend_pct": round(t_pct, 2),
                "confidence_lower": round(lo, 4),
                "confidence_upper": round(hi, 4),
            }
            horizon_results.append(hr)

            if _RISK_ORDER.get(risk, 0) > _RISK_ORDER.get(worst_risk, 0):
                worst_risk = risk
                worst_result = hr
            elif worst_result is None:
                worst_result = hr
        except Exception:
            pass

    # Use worst horizon for the primary fields, fall back to last horizon
    if worst_result is None:
        # All horizons failed — run single forecast as fallback
        forecast_result = run_forecast(dataset, column, horizon=horizon, method=method)
        fval = float(forecast_result.values[-1]) if forecast_result.values else current
        lo = float(forecast_result.lower_bound[-1]) if forecast_result.lower_bound else fval
        hi = float(forecast_result.upper_bound[-1]) if forecast_result.upper_bound else fval
        sd = (hi - lo) / (2 * 1.96) if hi != lo else float(series.std())
        worst_risk = assess_risk(current, fval, sd, series=series)
        worst_result = {
            "horizon": horizon, "forecast_value": round(fval, 4),
            "risk_level": worst_risk.value, "trend_pct": 0.0,
            "confidence_lower": round(lo, 4), "confidence_upper": round(hi, 4),
        }
        horizon_results = [worst_result]

    forecast_val = worst_result["forecast_value"]
    trend_pct = worst_result["trend_pct"]

    if abs(trend_pct) < 1.0:
        trend = TrendDirection.STABLE
    elif trend_pct > 0:
        trend = TrendDirection.INCREASING
    else:
        trend = TrendDirection.DECREASING

    # Optional time series analysis
    ts_analysis: Optional[TimeSeriesAnalysis] = None
    if include_analysis and len(series) >= 10:
        from omnihealth.timeseries import analyze
        try:
            ts_analysis = analyze(series)
        except Exception as e:
            logger.warning("Time series analysis failed for '%s': %s", column, e)

    # Degradation rate
    deg = detect_degradation(series)
    degradation_rate = deg.get("rate") if deg.get("is_degrading") else None

    return RiskAssessment(
        feature=column,
        current_value=round(current, 4),
        forecast_value=round(forecast_val, 4),
        risk_level=worst_risk,
        trend=trend,
        trend_pct=round(trend_pct, 2),
        confidence_lower=worst_result["confidence_lower"],
        confidence_upper=worst_result["confidence_upper"],
        degradation_rate=round(degradation_rate, 6) if degradation_rate is not None else None,
        time_series_analysis=ts_analysis,
        horizon_results=horizon_results,
    )


def assess_all_risks(
    dataset: Any,
    columns: Optional[List[str]] = None,
    *,
    horizon: int = 10,
    method: str = "auto",
    include_analysis: bool = False,
) -> List[RiskAssessment]:
    """Assess risk for multiple columns.

    Parameters
    ----------
    dataset : TabularDataset.
    columns : Columns to assess (default: all metric columns).
    horizon : Forecast horizon.
    method : Forecasting method.
    include_analysis : Include time series analysis per column.

    Returns
    -------
    List of RiskAssessment.
    """
    cols = columns or dataset.profile.metric_cols
    results = []
    for col in cols:
        try:
            ra = assess_feature_risk(
                dataset, col,
                horizon=horizon, method=method,
                include_analysis=include_analysis,
            )
            results.append(ra)
        except Exception as e:
            logger.warning("Risk assessment failed for '%s': %s", col, e)
    return results


# ── Degradation detection ────────────────────────────────────────────────────

def _auto_window(series: np.ndarray) -> int:
    """Auto-calibrate rolling window from autocorrelation decay.

    Uses the lag at which autocorrelation drops below 0.5 (or 1/e),
    clamped to [3, len//4]. This adapts to the signal's memory length.
    """
    n = len(series)
    if n < 8:
        return 3
    centered = series - np.mean(series)
    var = np.var(series)
    if var < 1e-12:
        return 3
    max_lag = min(n // 4, 50)
    for lag in range(1, max_lag):
        acf = float(np.mean(centered[:n - lag] * centered[lag:])) / var
        if acf < 0.5:
            return max(3, lag)
    return max_lag


def _auto_threshold(series: np.ndarray) -> float:
    """Auto-calibrate degradation threshold from delta noise level.

    Uses the std of period-over-period deltas (not raw values) so that
    a trending signal doesn't inflate the threshold. The threshold is set
    at 2x the delta noise — a sustained mean delta above this indicates
    real degradation, not random walk.
    """
    if len(series) < 3:
        return 1e-3
    deltas = np.diff(series)
    delta_std = float(np.std(deltas))
    # Threshold = 2x noise in deltas — sustained signal above this is real
    return max(delta_std * 2.0, 1e-6)


def detect_degradation(
    series: pd.Series,
    *,
    window: Optional[int] = None,
    threshold_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect sustained degradation patterns in a time series.

    Parameters auto-calibrate from the data when not provided:
    - window: derived from autocorrelation decay length
    - threshold: derived from coefficient of variation

    Parameters
    ----------
    series : Numeric time series.
    window : Rolling window (default: auto from autocorrelation).
    threshold_pct : Percentage threshold (default: auto from CV).

    Returns
    -------
    Dict with: is_degrading, rate, direction, confidence, window, threshold.
    """
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 4:
        return {"is_degrading": False, "rate": 0.0, "direction": "stable", "confidence": 0.0}

    values = clean.values

    # Auto-calibrate from data
    w = window if window is not None else _auto_window(values)
    if threshold_pct is not None:
        mean_val = abs(np.mean(values))
        thresh = mean_val * (threshold_pct / 100) if mean_val > 0 else 1e-3
    else:
        thresh = _auto_threshold(values)

    if len(values) < w + 1:
        return {"is_degrading": False, "rate": 0.0, "direction": "stable", "confidence": 0.0}

    deltas = np.diff(values)
    rolling_delta = pd.Series(deltas).rolling(w, min_periods=1).mean().values

    # Check last window of rolling deltas
    recent_deltas = rolling_delta[-w:]
    mean_recent = float(np.mean(recent_deltas))
    is_degrading = abs(mean_recent) > thresh

    # Direction
    if mean_recent > thresh:
        direction = "increasing"
    elif mean_recent < -thresh:
        direction = "decreasing"
    else:
        direction = "stable"

    # Confidence: what fraction of recent deltas exceed threshold
    if is_degrading:
        confidence = float(np.mean(np.abs(recent_deltas) > thresh))
    else:
        confidence = 0.0

    return {
        "is_degrading": bool(is_degrading),
        "rate": round(float(mean_recent), 6),
        "direction": direction,
        "confidence": round(confidence, 4),
        "window": w,
        "threshold": round(thresh, 6),
    }
