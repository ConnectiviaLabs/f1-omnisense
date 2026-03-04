"""Time series analysis — trend, stationarity, seasonality, forecastability.

Ports DataSense ts_analysis.py (pure computation, no MongoDB).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import IsolationForest

from omnihealth._types import TimeSeriesAnalysis, TrendDirection

logger = logging.getLogger(__name__)

_HAS_STATSMODELS = True
try:
    from statsmodels.tsa.stattools import acf, adfuller, kpss
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
except ImportError:
    _HAS_STATSMODELS = False
    logger.info("statsmodels not installed — using polyfit/ACF fallbacks")


# ── Main entry ───────────────────────────────────────────────────────────────

def analyze(
    data: pd.Series,
    *,
    seasonal_periods: Optional[List[int]] = None,
) -> TimeSeriesAnalysis:
    """Full time series analysis: trend + seasonality + stationarity + anomalies + forecastability.

    Parameters
    ----------
    data : Numeric time series (pd.Series).
    seasonal_periods : Periods to test for seasonality (default [7, 24]).

    Returns
    -------
    TimeSeriesAnalysis dataclass.
    """
    data = _clean(data)
    if len(data) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(data)}")

    periods = seasonal_periods or [7, 24]

    trend_info = analyze_trend(data)
    stationarity_info = test_stationarity(data)
    seasonality_info = detect_seasonality(data, periods=periods)
    anomaly_info = detect_anomalies_ts(data)
    forecast_info = score_forecastability(data, anomaly_count=anomaly_info["count"])

    # Operational zones
    zones = _compute_operational_zones(data)

    # Recommendations
    recommendations = _generate_recommendations(
        trend_info, seasonality_info, stationarity_info, anomaly_info,
    )

    return TimeSeriesAnalysis(
        trend=TrendDirection(trend_info["direction"]),
        trend_strength=trend_info["strength"],
        slope=trend_info["slope"],
        drift_rate=trend_info["drift_rate"],
        is_stationary=stationarity_info["is_stationary"],
        seasonality_detected=seasonality_info["detected"],
        seasonal_period=seasonality_info.get("primary_period"),
        seasonal_strength=seasonality_info["strength"],
        anomaly_count=anomaly_info["count"],
        anomaly_pct=anomaly_info["percentage"],
        forecastability_score=forecast_info["score"],
        forecastability_rating=forecast_info["rating"],
        operational_zones=zones,
        recommendations=recommendations,
    )


# ── Trend analysis ───────────────────────────────────────────────────────────

def analyze_trend(data: pd.Series) -> Dict[str, Any]:
    """Trend analysis via STL decomposition (or polyfit fallback).

    Returns dict with: direction, strength, slope, drift_rate, change_points.
    """
    clean = _clean(data)
    n = len(clean)
    values = clean.values

    if _HAS_STATSMODELS and n >= 6:
        # Sample for performance
        sample_rate = max(1, n // 500)
        sampled = values[::sample_rate]

        seasonal_period = min(7, len(sampled) // 4)
        if seasonal_period < 3:
            seasonal_period = 3

        try:
            stl = STL(sampled, seasonal=seasonal_period)
            decomp = stl.fit()
            trend_vals = decomp.trend

            # Trend strength
            combined_var = np.var(trend_vals + decomp.resid)
            if combined_var > 0:
                strength = max(0.0, 1.0 - (np.var(decomp.resid) / combined_var))
            else:
                strength = 0.0

            # Linear fit on trend component
            x = np.arange(len(trend_vals))
            slope, _ = np.polyfit(x, trend_vals, 1)
            drift_rate = slope * sample_rate

        except Exception:
            # Fall back to polyfit
            slope, strength, drift_rate = _polyfit_trend(values)
    else:
        slope, strength, drift_rate = _polyfit_trend(values)

    if abs(slope) < 1e-10:
        direction = "stable"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    change_points = _find_change_points(values)

    return {
        "direction": direction,
        "strength": round(float(strength), 4),
        "slope": round(float(slope), 6),
        "drift_rate": round(float(drift_rate), 6),
        "change_points": change_points,
    }


def _polyfit_trend(values: np.ndarray):
    """Simple polyfit fallback for trend extraction."""
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    residuals = values - (slope * x + intercept)
    total_var = np.var(values)
    if total_var > 0:
        strength = max(0.0, 1.0 - (np.var(residuals) / total_var))
    else:
        strength = 0.0
    return slope, strength, slope


def _find_change_points(values: np.ndarray, min_segment: int = 10) -> List[int]:
    """Simple change point detection via rolling variance shift."""
    if len(values) < min_segment * 2:
        return []

    window = max(min_segment, len(values) // 10)
    rolling_std = pd.Series(values).rolling(window, min_periods=window // 2).std()
    if rolling_std.dropna().empty:
        return []

    baseline_std = float(rolling_std.dropna().median())
    if baseline_std < 1e-10:
        return []

    change_points = []
    above = rolling_std > baseline_std * 2
    transitions = above.astype(int).diff().fillna(0)
    for idx in transitions.index:
        if abs(transitions[idx]) > 0:
            change_points.append(int(idx))

    return change_points


# ── Stationarity tests ───────────────────────────────────────────────────────

def test_stationarity(data: pd.Series) -> Dict[str, Any]:
    """ADF + KPSS stationarity tests (polyfit + rolling-stats fallback).

    Returns dict with: is_stationary, adf_pvalue, kpss_pvalue, mean_stable, variance_stable.
    """
    clean = _clean(data)
    values = clean.values

    # Sample for large data
    sample_rate = max(1, len(values) // 500)
    sampled = values[::sample_rate]

    adf_p = None
    kpss_p = None

    if _HAS_STATSMODELS and len(sampled) >= 8:
        max_lag = min(12, len(sampled) // 10)
        max_lag = max(1, max_lag)

        try:
            adf_result = adfuller(sampled, maxlag=max_lag, autolag="AIC")
            adf_p = float(adf_result[1])
        except Exception:
            pass

        try:
            kpss_result = kpss(sampled, regression="c", nlags=max_lag)
            kpss_p = float(kpss_result[1])
        except Exception:
            pass

    # Rolling stability check
    window = max(3, min(len(clean) // 4, 30))
    rolling_mean = clean.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = clean.rolling(window=window, min_periods=window // 2).std()

    mean_stable = False
    std_stable = False
    if len(rolling_mean.dropna()) > 0:
        mean_stable = bool(np.std(rolling_mean.dropna()) < 0.1 * abs(np.mean(values)))
    if len(rolling_std.dropna()) > 0:
        std_val = np.std(values)
        std_stable = bool(np.std(rolling_std.dropna()) < 0.1 * std_val) if std_val > 0 else True

    # Combine
    if adf_p is not None and kpss_p is not None:
        is_stationary = (adf_p < 0.05) and (kpss_p > 0.05)
    elif adf_p is not None:
        is_stationary = adf_p < 0.05
    else:
        is_stationary = mean_stable and std_stable

    return {
        "is_stationary": bool(is_stationary),
        "adf_pvalue": round(adf_p, 4) if adf_p is not None else None,
        "kpss_pvalue": round(kpss_p, 4) if kpss_p is not None else None,
        "mean_stable": bool(mean_stable),
        "variance_stable": bool(std_stable),
    }


# ── Seasonality detection ────────────────────────────────────────────────────

def detect_seasonality(
    data: pd.Series,
    periods: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Multi-period seasonal decomposition + ACF analysis.

    Returns dict with: detected, primary_period, strength, significant_lags, type.
    """
    clean = _clean(data)
    periods = periods or [7, 24]

    best_strength = 0.0
    best_period = None

    if _HAS_STATSMODELS and len(clean) >= 6:
        # Sample for performance
        sample_rate = max(1, len(clean) // 500)
        sampled = clean.values[::sample_rate]

        for period in periods:
            if len(sampled) < 2 * period:
                continue
            try:
                decomp = seasonal_decompose(sampled, model="additive", period=period)
                seasonal = decomp.seasonal
                resid = decomp.resid
                resid_clean = resid[~np.isnan(resid)]
                seasonal_clean = seasonal[~np.isnan(seasonal)]

                if len(resid_clean) == 0 or len(seasonal_clean) == 0:
                    continue

                resid_var = np.var(resid_clean)
                total_var = np.var(np.concatenate([seasonal_clean, resid_clean]))
                strength = max(0.0, 1.0 - (resid_var / total_var)) if total_var > 0 else 0.0

                if strength > best_strength:
                    best_strength = strength
                    best_period = period

                if best_strength > 0.7:
                    break
            except Exception:
                continue

    # ACF analysis for significant lags
    significant_lags = []
    if _HAS_STATSMODELS and len(clean) >= 12:
        try:
            max_lags = min(30, len(clean) // 6)
            acf_values = acf(clean.values, nlags=max_lags)
            significant_lags = [
                int(i) for i in range(1, len(acf_values))
                if abs(acf_values[i]) > 0.2
            ]
        except Exception:
            pass

    detected = best_strength > 0.3

    # Classify type
    season_type = "unknown"
    if best_period is not None:
        if best_period <= 12:
            season_type = "sub-daily-cycle"
        elif best_period <= 24:
            season_type = "daily-cycle"
        elif best_period <= 168:
            season_type = "weekly-cycle"
        else:
            season_type = "long-period"

    return {
        "detected": detected,
        "primary_period": best_period,
        "strength": round(float(best_strength), 4),
        "significant_lags": significant_lags[:10],
        "type": season_type if detected else None,
    }


# ── Anomaly detection (time series) ─────────────────────────────────────────

def detect_anomalies_ts(data: pd.Series) -> Dict[str, Any]:
    """Triple method: Z-score + IQR + IsolationForest → union of anomalies.

    Returns dict with: count, percentage, anomaly_indices, operational_thresholds.
    """
    clean = _clean(data)
    values = clean.values
    n = len(values)

    all_anomalies = set()

    # 1. Z-score (3 STD)
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val > 0:
        z_scores = np.abs((values - mean_val) / std_val)
        z_anomalies = np.where(z_scores > 3)[0]
        all_anomalies.update(z_anomalies.tolist())

    # 2. IQR (1.5 * IQR)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    if iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_anomalies = np.where((values < lower) | (values > upper))[0]
        all_anomalies.update(iqr_anomalies.tolist())
    else:
        lower = float(q1)
        upper = float(q3)

    # 3. Isolation Forest
    if n >= 10:
        sample_size = min(500, n)
        if n > 500:
            sample_idx = np.random.choice(n, sample_size, replace=False)
            sample = values[sample_idx].reshape(-1, 1)
        else:
            sample = values.reshape(-1, 1)

        iso = IsolationForest(
            contamination=0.1, random_state=42,
            n_estimators=50, max_samples=min(256, sample_size), n_jobs=1,
        )
        if n > 500:
            iso.fit(sample)
            preds = iso.predict(values.reshape(-1, 1))
        else:
            preds = iso.fit_predict(values.reshape(-1, 1))
        iso_anomalies = np.where(preds == -1)[0]
        all_anomalies.update(iso_anomalies.tolist())

    sorted_anomalies = sorted(all_anomalies)
    count = len(sorted_anomalies)
    pct = (count / n * 100) if n > 0 else 0.0

    return {
        "count": count,
        "percentage": round(pct, 2),
        "anomaly_indices": sorted_anomalies,
        "operational_thresholds": {
            "upper_limit": float(q3 + 1.5 * iqr) if iqr > 0 else float(q3),
            "lower_limit": float(q1 - 1.5 * iqr) if iqr > 0 else float(q1),
            "normal_range": [float(q1), float(q3)],
        },
    }


# ── Forecastability scoring ──────────────────────────────────────────────────

def score_forecastability(
    data: pd.Series,
    anomaly_count: int = 0,
) -> Dict[str, Any]:
    """Score how forecastable a time series is (0-100).

    Combines: data completeness, signal clarity, pattern strength, sensor reliability.
    Returns dict with: score, rating ("High"/"Medium"/"Low"), components.
    """
    clean = _clean(data)
    values = clean.values
    n = len(values)

    scores: Dict[str, float] = {}

    # Data completeness
    missing_pct = data.isna().sum() / len(data) * 100 if len(data) > 0 else 100
    scores["data_completeness"] = max(0, 100 - missing_pct * 2)

    # Signal clarity (signal-to-noise via Savitzky-Golay)
    if n >= 7:
        win = min(21, n if n % 2 == 1 else n - 1)
        win = max(5, win)
        poly = min(3, win - 1)
        try:
            smooth = savgol_filter(values, window_length=win, polyorder=poly)
            noise = values - smooth
            noise_std = np.std(noise)
            signal_std = np.std(values)
            snr = signal_std / noise_std if noise_std > 0 else 100
            scores["signal_clarity"] = min(100, snr * 10)
        except Exception:
            scores["signal_clarity"] = 50.0
    else:
        scores["signal_clarity"] = 50.0

    # Pattern strength (autocorrelation)
    if _HAS_STATSMODELS and n >= 12:
        try:
            max_lags = min(10, n // 6)
            acf_vals = acf(values, nlags=max_lags)
            scores["pattern_strength"] = float(np.mean(np.abs(acf_vals[1:])) * 100)
        except Exception:
            scores["pattern_strength"] = 30.0
    else:
        # Simple lag-1 autocorrelation fallback
        if n >= 3:
            lag1_corr = np.corrcoef(values[:-1], values[1:])[0, 1]
            scores["pattern_strength"] = abs(lag1_corr) * 100
        else:
            scores["pattern_strength"] = 30.0

    # Sensor reliability
    scores["sensor_reliability"] = max(0, 100 - (anomaly_count / max(n, 1) * 100))

    overall = float(np.mean(list(scores.values())))
    if overall > 70:
        rating = "High"
    elif overall > 50:
        rating = "Medium"
    else:
        rating = "Low"

    return {
        "score": round(overall, 1),
        "rating": rating,
        "components": {k: round(v, 1) for k, v in scores.items()},
    }


# ── Operational zone classification ──────────────────────────────────────────

def classify_operational_zone(value: float, data: pd.Series) -> str:
    """Classify a value into operational zone based on data percentiles.

    Zones: critical_low / warning_low / optimal / warning_high / critical_high.
    """
    clean = _clean(data)
    values = clean.values
    thresholds = np.percentile(values, [5, 10, 25, 75, 90, 95])

    if value <= thresholds[0]:
        return "critical_low"
    elif value <= thresholds[1]:
        return "warning_low"
    elif value <= thresholds[3]:
        return "optimal"
    elif value <= thresholds[4]:
        return "warning_high"
    else:
        return "critical_high"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _clean(data: pd.Series) -> pd.Series:
    """Drop NaN and infinite values."""
    clean = data.replace([np.inf, -np.inf], np.nan).dropna()
    return clean


def _compute_operational_zones(data: pd.Series) -> Dict[str, Any]:
    """Compute operational zone boundaries from percentiles."""
    clean = _clean(data)
    values = clean.values
    if len(values) == 0:
        return {}
    p = np.percentile(values, [10, 25, 75, 90])
    return {
        "warning_lower": float(p[0]),
        "optimal_low": float(p[1]),
        "optimal_high": float(p[2]),
        "warning_upper": float(p[3]),
        "current_value": float(values[-1]),
    }


def _generate_recommendations(
    trend: Dict, seasonality: Dict, stationarity: Dict, anomalies: Dict,
) -> List[str]:
    """Generate maintenance-relevant recommendations from analysis."""
    recs = []

    direction = trend.get("direction", "stable")
    strength = trend.get("strength", 0)
    if direction == "increasing" and strength > 0.5:
        recs.append("Strong increasing trend detected — monitor for threshold breaches.")
    elif direction == "decreasing" and strength > 0.5:
        recs.append("Strong decreasing trend — investigate potential degradation.")

    if not stationarity.get("is_stationary", True):
        recs.append("Non-stationary data — consider differencing for forecasting.")

    if seasonality.get("detected", False):
        period = seasonality.get("primary_period", "?")
        recs.append(f"Seasonal pattern (period={period}) — schedule maintenance around cycles.")

    anom_pct = anomalies.get("percentage", 0)
    if anom_pct > 10:
        recs.append(f"High anomaly rate ({anom_pct:.1f}%) — immediate inspection recommended.")
    elif anom_pct > 5:
        recs.append(f"Moderate anomaly rate ({anom_pct:.1f}%) — schedule preventive maintenance.")

    if not recs:
        recs.append("System operating within normal parameters.")

    return recs
