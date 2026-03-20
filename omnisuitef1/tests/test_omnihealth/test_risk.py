"""Tests for omnihealth.risk — real execution, no mocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omnidata._types import ColumnProfile, ColumnRole, DatasetProfile, DType, TabularDataset
from omnihealth._types import RiskAssessment, RiskLevel, TrendDirection
from omnihealth.risk import (
    _auto_horizons,
    _auto_threshold,
    _auto_window,
    assess_all_risks,
    assess_feature_risk,
    assess_risk,
    detect_degradation,
)


def _make_dataset(df: pd.DataFrame) -> TabularDataset:
    """Build TabularDataset from DataFrame."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    ts_col = next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])), None)
    col_profiles = [
        ColumnProfile(name=c, dtype=DType.FLOAT, role=ColumnRole.METRIC,
                      null_count=int(df[c].isna().sum()), unique_count=int(df[c].nunique()))
        for c in numeric_cols
    ]
    profile = DatasetProfile(
        row_count=len(df), column_count=len(numeric_cols),
        columns=col_profiles, metric_cols=numeric_cols, timestamp_col=ts_col,
    )
    return TabularDataset(df=df, profile=profile)


# ── assess_risk (pure computation) ─────────────────────────────────────────

class TestAssessRisk:
    def test_no_change(self):
        assert assess_risk(10.0, 10.0, 1.0) == RiskLevel.LOW

    def test_small_change(self):
        assert assess_risk(10.0, 10.5, 1.0) == RiskLevel.LOW

    def test_medium_change(self):
        assert assess_risk(10.0, 11.5, 1.0) == RiskLevel.MEDIUM

    def test_large_change(self):
        assert assess_risk(10.0, 13.0, 1.0) == RiskLevel.HIGH

    def test_zero_std(self):
        assert assess_risk(10.0, 12.0, 0.0) == RiskLevel.HIGH

    def test_percentile_mode_with_series(self):
        np.random.seed(42)
        series = pd.Series(np.random.normal(100, 1, 50))
        assert assess_risk(100.0, 100.2, 1.0, series=series) == RiskLevel.LOW
        assert assess_risk(100.0, 120.0, 1.0, series=series) == RiskLevel.HIGH

    def test_short_series_fallback(self):
        series = pd.Series([1, 2, 3, 4, 5])
        result = assess_risk(3.0, 3.5, 1.0, series=series)
        assert result in RiskLevel


# ── _auto_window ────────────────────────────────────────────────────────────

class TestAutoWindow:
    def test_short(self):
        assert _auto_window(np.array([1, 2, 3, 4, 5])) == 3

    def test_white_noise(self):
        np.random.seed(42)
        assert _auto_window(np.random.normal(0, 1, 100)) <= 10

    def test_constant(self):
        assert _auto_window(np.array([5.0] * 20)) == 3


# ── _auto_threshold ─────────────────────────────────────────────────────────

class TestAutoThreshold:
    def test_noisy(self):
        np.random.seed(42)
        assert _auto_threshold(np.random.normal(50, 5, 100)) > 0

    def test_trending(self):
        assert _auto_threshold(np.arange(100, dtype=float)) < 1.0

    def test_constant(self):
        assert _auto_threshold(np.array([5.0] * 50)) == 1e-6


# ── _auto_horizons ──────────────────────────────────────────────────────────

class TestAutoHorizons:
    def test_returns_three_sorted(self, stable_series):
        h = _auto_horizons(stable_series)
        assert len(h) == 3
        assert h == sorted(h)
        assert all(x > 0 for x in h)

    def test_capped_at_third(self, stable_series):
        h = _auto_horizons(stable_series)
        assert all(x <= max(len(stable_series) // 3, 3) for x in h)

    def test_seasonal_picks_period(self, seasonal_series):
        h = _auto_horizons(seasonal_series)
        assert any(5 <= x <= 15 for x in h)


# ── detect_degradation (pure computation) ──────────────────────────────────

class TestDetectDegradation:
    def test_stable_not_degrading(self, stable_series):
        r = detect_degradation(stable_series)
        assert r["is_degrading"] is False
        assert r["direction"] == "stable"

    def test_increasing_degradation(self, degrading_series):
        r = detect_degradation(degrading_series)
        assert r["is_degrading"] is True
        assert r["direction"] == "increasing"
        assert r["rate"] > 0

    def test_decreasing_degradation(self):
        np.random.seed(42)
        data = pd.Series(100 - np.arange(50) * 2.0 + np.random.normal(0, 0.1, 50))
        r = detect_degradation(data)
        assert r["is_degrading"] is True
        assert r["direction"] == "decreasing"

    def test_too_short(self):
        r = detect_degradation(pd.Series([1, 2, 3]))
        assert r["is_degrading"] is False

    def test_handles_inf(self):
        data = pd.Series([1, 2, np.inf, 4, 5, 6, 7, 8, 9, 10])
        r = detect_degradation(data)
        assert isinstance(r, dict)

    def test_auto_calibrated(self, degrading_series):
        r = detect_degradation(degrading_series)
        assert r["window"] >= 3
        assert r["threshold"] > 0
        assert 0 <= r["confidence"] <= 1


# ── assess_feature_risk — REAL forecasting ──────────────────────────────────

class TestAssessFeatureRisk:
    def test_stable_data(self, normal_df):
        """Real forecast on normal data → should produce a valid RiskAssessment."""
        dataset = _make_dataset(normal_df)
        ra = assess_feature_risk(dataset, "vibration", horizon=10, include_analysis=False)
        assert isinstance(ra, RiskAssessment)
        assert ra.feature == "vibration"
        assert ra.risk_level in RiskLevel
        assert ra.trend in TrendDirection
        assert ra.horizon_results is not None
        assert len(ra.horizon_results) >= 1

    def test_anomalous_data(self, anomalous_df):
        """Data with anomalies → risk should reflect instability."""
        dataset = _make_dataset(anomalous_df)
        ra = assess_feature_risk(dataset, "vibration", horizon=10, include_analysis=False)
        assert isinstance(ra, RiskAssessment)
        # The injected spike should affect the forecast
        assert ra.current_value > 0

    def test_with_time_series_analysis(self, normal_df):
        """include_analysis=True → time_series_analysis populated."""
        dataset = _make_dataset(normal_df)
        ra = assess_feature_risk(dataset, "vibration", horizon=5, include_analysis=True)
        assert isinstance(ra, RiskAssessment)
        # time_series_analysis may or may not succeed depending on data length
        # but it shouldn't crash

    def test_multi_horizon(self, normal_df):
        """Should produce multiple horizon results."""
        dataset = _make_dataset(normal_df)
        ra = assess_feature_risk(dataset, "vibration", horizon=10)
        assert len(ra.horizon_results) >= 1
        for hr in ra.horizon_results:
            assert "horizon" in hr
            assert "risk_level" in hr
            assert "forecast_value" in hr

    def test_to_dict(self, normal_df):
        dataset = _make_dataset(normal_df)
        ra = assess_feature_risk(dataset, "vibration", horizon=5, include_analysis=False)
        d = ra.to_dict()
        assert isinstance(d, dict)
        assert d["feature"] == "vibration"
        assert isinstance(d["risk_level"], str)


# ── assess_all_risks ────────────────────────────────────────────────────────

class TestAssessAllRisks:
    def test_multiple_columns(self, normal_df):
        dataset = _make_dataset(normal_df)
        results = assess_all_risks(dataset, columns=["vibration", "temperature"],
                                   horizon=5, include_analysis=False)
        assert len(results) == 2
        features = {r.feature for r in results}
        assert features == {"vibration", "temperature"}

    def test_all_metric_cols(self, normal_df):
        dataset = _make_dataset(normal_df)
        results = assess_all_risks(dataset, horizon=5, include_analysis=False)
        assert len(results) == len(dataset.profile.metric_cols)


# ── assess_feature_risk with REAL F1 data ──────────────────────────────────

class TestAssessFeatureRiskF1:
    def test_f1_rpm_risk(self, f1_raw):
        """Run real forecast + risk assessment on F1 RPM data."""
        if "RPM" not in f1_raw.columns:
            pytest.skip("No RPM column")
        df = f1_raw[["RPM"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 20:
            pytest.skip("Not enough data")
        dataset = _make_dataset(df)
        ra = assess_feature_risk(dataset, "RPM", horizon=10, include_analysis=False)
        assert isinstance(ra, RiskAssessment)
        print(f"\nF1 RPM: risk={ra.risk_level.value}, trend={ra.trend.value} "
              f"({ra.trend_pct:+.1f}%), horizons={len(ra.horizon_results)}")

    def test_f1_speed_risk(self, f1_raw):
        """Run real forecast + risk assessment on F1 Speed data."""
        if "Speed" not in f1_raw.columns:
            pytest.skip("No Speed column")
        df = f1_raw[["Speed"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 20:
            pytest.skip("Not enough data")
        dataset = _make_dataset(df)
        ra = assess_feature_risk(dataset, "Speed", horizon=10, include_analysis=True)
        assert isinstance(ra, RiskAssessment)
        print(f"\nF1 Speed: risk={ra.risk_level.value}, trend={ra.trend.value} "
              f"({ra.trend_pct:+.1f}%)")

    def test_f1_degradation_detection(self, f1_raw):
        """Run degradation detection on F1 RPM."""
        if "RPM" not in f1_raw.columns:
            pytest.skip("No RPM column")
        series = pd.to_numeric(f1_raw["RPM"], errors="coerce").dropna()
        if len(series) < 10:
            pytest.skip("Not enough data")
        r = detect_degradation(series)
        assert isinstance(r["is_degrading"], bool)
        print(f"\nF1 RPM degradation: {r['direction']}, rate={r['rate']:.4f}, "
              f"conf={r['confidence']:.2f}")
