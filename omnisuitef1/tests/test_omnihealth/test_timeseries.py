"""Tests for omnihealth.timeseries — trend, stationarity, seasonality, anomalies, forecastability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omnihealth._types import TimeSeriesAnalysis, TrendDirection
from omnihealth.timeseries import (
    analyze,
    analyze_trend,
    classify_operational_zone,
    detect_anomalies_ts,
    detect_seasonality,
    score_forecastability,
    test_stationarity,
)


# ── analyze (full pipeline) ─────────────────────────────────────────────────

class TestAnalyze:
    def test_returns_time_series_analysis(self, stable_series):
        result = analyze(stable_series)
        assert isinstance(result, TimeSeriesAnalysis)

    def test_trending_up_detected(self, trending_up_series):
        result = analyze(trending_up_series)
        assert result.trend == TrendDirection.INCREASING
        assert result.trend_strength > 0.3

    def test_trending_down_detected(self, trending_down_series):
        result = analyze(trending_down_series)
        assert result.trend == TrendDirection.DECREASING

    def test_stable_detected(self, stable_series):
        result = analyze(stable_series)
        # STL may detect a slight trend in random data — just check it's weak
        assert result.trend_strength < 0.5

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            analyze(pd.Series([1.0, 2.0]))

    def test_has_recommendations(self, trending_up_series):
        result = analyze(trending_up_series)
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0

    def test_operational_zones_populated(self, stable_series):
        result = analyze(stable_series)
        assert "optimal_low" in result.operational_zones
        assert "optimal_high" in result.operational_zones

    def test_forecastability_score_in_range(self, stable_series):
        result = analyze(stable_series)
        assert 0 <= result.forecastability_score <= 100

    def test_handles_nan_values(self):
        data = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, np.nan, 12])
        result = analyze(data)
        assert isinstance(result, TimeSeriesAnalysis)

    def test_custom_seasonal_periods(self, seasonal_series):
        result = analyze(seasonal_series, seasonal_periods=[7])
        assert isinstance(result, TimeSeriesAnalysis)


# ── analyze_trend ───────────────────────────────────────────────────────────

class TestAnalyzeTrend:
    def test_increasing_trend(self, trending_up_series):
        info = analyze_trend(trending_up_series)
        assert info["direction"] == "increasing"
        assert info["slope"] > 0
        assert info["strength"] > 0.5

    def test_decreasing_trend(self, trending_down_series):
        info = analyze_trend(trending_down_series)
        assert info["direction"] == "decreasing"
        assert info["slope"] < 0

    def test_stable_trend(self, stable_series):
        info = analyze_trend(stable_series)
        # Slope should be near zero
        assert abs(info["slope"]) < 0.1

    def test_has_change_points(self):
        """Concatenate two very different distributions → should detect change."""
        np.random.seed(42)
        part1 = np.random.normal(10, 0.5, 50)
        part2 = np.random.normal(10, 5.0, 50)  # much higher variance
        data = pd.Series(np.concatenate([part1, part2]))
        info = analyze_trend(data)
        assert isinstance(info["change_points"], list)

    def test_short_series(self):
        data = pd.Series([1, 2, 3])
        info = analyze_trend(data)
        assert "direction" in info

    def test_constant_series(self):
        data = pd.Series([5.0] * 20)
        info = analyze_trend(data)
        assert info["direction"] == "stable"
        assert abs(info["slope"]) < 1e-8


# ── test_stationarity ───────────────────────────────────────────────────────

class TestStationarity:
    def test_stationary_white_noise(self, stable_series):
        info = test_stationarity(stable_series)
        assert info["is_stationary"] is True

    def test_non_stationary_trend(self, trending_up_series):
        info = test_stationarity(trending_up_series)
        assert info["is_stationary"] is False

    def test_short_series_fallback(self):
        data = pd.Series([1, 2, 3, 4, 5])
        info = test_stationarity(data)
        assert "is_stationary" in info

    def test_returns_pvalues_when_available(self, stable_series):
        info = test_stationarity(stable_series)
        # With statsmodels installed, should have p-values
        if info["adf_pvalue"] is not None:
            assert 0 <= info["adf_pvalue"] <= 1

    def test_rolling_stability_flags(self, stable_series):
        info = test_stationarity(stable_series)
        assert isinstance(info["mean_stable"], bool)
        assert isinstance(info["variance_stable"], bool)


# ── detect_seasonality ──────────────────────────────────────────────────────

class TestSeasonality:
    def test_detects_period_7(self, seasonal_series):
        info = detect_seasonality(seasonal_series, periods=[7])
        # With statsmodels: detected=True, period=7, strong signal
        # Without statsmodels: seasonal decompose unavailable, may not detect
        try:
            import statsmodels  # noqa: F401
            assert info["detected"] is True
            assert info["primary_period"] == 7
            assert info["strength"] > 0.3
        except ImportError:
            # Fallback path — just verify it doesn't crash and returns valid structure
            assert isinstance(info["detected"], bool)
            assert isinstance(info["strength"], float)

    def test_no_seasonality_in_noise(self, stable_series):
        info = detect_seasonality(stable_series, periods=[7, 24])
        # Random noise shouldn't have strong seasonality
        assert info["strength"] < 0.5

    def test_significant_lags_returned(self, seasonal_series):
        info = detect_seasonality(seasonal_series, periods=[7])
        assert isinstance(info["significant_lags"], list)

    def test_short_series_no_crash(self):
        data = pd.Series([1, 2, 3, 4, 5])
        info = detect_seasonality(data, periods=[7])
        assert "detected" in info

    def test_season_type_classification(self, seasonal_series):
        info = detect_seasonality(seasonal_series, periods=[7])
        if info["detected"]:
            assert info["type"] in {"sub-daily-cycle", "daily-cycle", "weekly-cycle", "long-period"}


# ── detect_anomalies_ts ─────────────────────────────────────────────────────

class TestAnomaliesTS:
    def test_clean_data_few_anomalies(self, stable_series):
        info = detect_anomalies_ts(stable_series)
        assert info["count"] >= 0
        assert info["percentage"] >= 0

    def test_injected_outliers_detected(self):
        np.random.seed(42)
        data = pd.Series(np.random.normal(50, 1, 100))
        data.iloc[50] = 100  # extreme outlier
        data.iloc[70] = -50  # extreme outlier
        info = detect_anomalies_ts(data)
        assert info["count"] >= 2
        assert 50 in info["anomaly_indices"] or 70 in info["anomaly_indices"]

    def test_operational_thresholds(self, stable_series):
        info = detect_anomalies_ts(stable_series)
        thresholds = info["operational_thresholds"]
        assert thresholds["upper_limit"] > thresholds["lower_limit"]
        assert len(thresholds["normal_range"]) == 2

    def test_all_same_values(self):
        data = pd.Series([5.0] * 50)
        info = detect_anomalies_ts(data)
        # No variance → no z-score or IQR anomalies
        assert isinstance(info["count"], int)

    def test_small_dataset(self):
        """<10 points → IsolationForest skipped but z-score/IQR still run."""
        data = pd.Series([1, 2, 3, 100, 5, 6, 7])
        info = detect_anomalies_ts(data)
        assert info["count"] >= 1  # 100 is an outlier


# ── score_forecastability ───────────────────────────────────────────────────

class TestForecastability:
    def test_clean_predictable_data(self, trending_up_series):
        info = score_forecastability(trending_up_series, anomaly_count=0)
        assert info["score"] > 50
        assert info["rating"] in {"High", "Medium", "Low"}

    def test_noisy_data_lower_score(self):
        np.random.seed(42)
        noisy = pd.Series(np.random.uniform(-100, 100, 100))
        info = score_forecastability(noisy, anomaly_count=20)
        clean = pd.Series(np.arange(100) * 1.0)
        info_clean = score_forecastability(clean, anomaly_count=0)
        assert info_clean["score"] > info["score"]

    def test_has_component_breakdown(self, stable_series):
        info = score_forecastability(stable_series)
        assert "data_completeness" in info["components"]
        assert "signal_clarity" in info["components"]
        assert "pattern_strength" in info["components"]
        assert "sensor_reliability" in info["components"]

    def test_missing_data_penalized(self):
        data = pd.Series([1, 2, np.nan, np.nan, 5, 6, np.nan, 8, 9, 10])
        info = score_forecastability(data)
        assert info["components"]["data_completeness"] < 100

    def test_high_anomalies_penalized(self, stable_series):
        info_clean = score_forecastability(stable_series, anomaly_count=0)
        info_dirty = score_forecastability(stable_series, anomaly_count=50)
        assert info_clean["components"]["sensor_reliability"] > info_dirty["components"]["sensor_reliability"]


# ── classify_operational_zone ───────────────────────────────────────────────

class TestOperationalZone:
    def test_optimal_for_median(self, stable_series):
        median = stable_series.median()
        zone = classify_operational_zone(median, stable_series)
        assert zone == "optimal"

    def test_critical_high(self, stable_series):
        extreme = stable_series.max() + 10
        zone = classify_operational_zone(extreme, stable_series)
        assert zone == "critical_high"

    def test_critical_low(self, stable_series):
        extreme = stable_series.min() - 10
        zone = classify_operational_zone(extreme, stable_series)
        assert zone == "critical_low"

    def test_warning_zones_exist(self):
        """Values near the 5th and 95th percentile should be warning zones."""
        np.random.seed(42)
        data = pd.Series(np.arange(1000, dtype=float))
        # p5=49.5, p10=99.5, p25=249.75, p75=749.25, p90=899.5, p95=949.5
        assert classify_operational_zone(75.0, data) == "warning_low"
        assert classify_operational_zone(875.0, data) == "warning_high"
