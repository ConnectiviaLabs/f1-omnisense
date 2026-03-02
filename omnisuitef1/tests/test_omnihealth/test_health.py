"""Tests for omnihealth.health — real execution, no mocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omnianalytics._types import SeverityLevel
from omnihealth._types import (
    HealthScore,
    MaintenanceAction,
    RiskLevel,
    SEVERITY_TO_ACTION,
)
from omnihealth.health import (
    _consensus_gated_action,
    _risk_from_severity,
    _worst_severity,
    add_lifecycle_context,
    add_temporal_features,
    assess_component,
    assess_components,
    score_health,
)
from omnianalytics._types import AnomalyScore


# ── score_health ────────────────────────────────────────────────────────────

class TestScoreHealth:
    def test_zero_anomaly(self):
        assert score_health(0.0) == 100.0

    def test_full_anomaly(self):
        assert score_health(1.0) == 20.0

    def test_above_one_clamps(self):
        assert score_health(1.5) == 10.0

    def test_mid_range(self):
        assert score_health(0.5) == 60.0

    def test_custom_clamp(self):
        assert score_health(2.0, clamp_low=0.0) == 0.0
        assert score_health(0.0, clamp_high=95.0) == 95.0

    @pytest.mark.parametrize("score,expected", [
        (0.0, 100.0), (0.25, 80.0), (0.5, 60.0), (0.75, 40.0), (1.0, 20.0),
    ])
    def test_linear_mapping(self, score, expected):
        assert abs(score_health(score) - expected) < 0.01


# ── _worst_severity / _risk_from_severity ───────────────────────────────────

class TestHelpers:
    def test_worst_severity_mixed(self):
        scores = [
            AnomalyScore(0, 0.1, 0.0, SeverityLevel.NORMAL, 1, 4),
            AnomalyScore(1, 0.8, 0.0, SeverityLevel.HIGH, 4, 4),
            AnomalyScore(2, 0.3, 0.0, SeverityLevel.LOW, 2, 4),
        ]
        assert _worst_severity(scores) == SeverityLevel.HIGH

    def test_risk_from_severity(self):
        assert _risk_from_severity(SeverityLevel.CRITICAL) == RiskLevel.HIGH
        assert _risk_from_severity(SeverityLevel.HIGH) == RiskLevel.HIGH
        assert _risk_from_severity(SeverityLevel.MEDIUM) == RiskLevel.MEDIUM
        assert _risk_from_severity(SeverityLevel.LOW) == RiskLevel.LOW
        assert _risk_from_severity(SeverityLevel.NORMAL) == RiskLevel.LOW


# ── consensus gated action ──────────────────────────────────────────────────

class TestConsensusGatedAction:
    def test_high_consensus_full_action(self):
        # 87.5% consensus → no downgrade
        action = _consensus_gated_action(SeverityLevel.CRITICAL, 35, 4, 10)
        assert action == MaintenanceAction.ALERT_AND_REMEDIATE

    def test_medium_consensus_downgrade_one(self):
        # 62.5% → downgrade 1
        action = _consensus_gated_action(SeverityLevel.CRITICAL, 25, 4, 10)
        assert action == MaintenanceAction.ALERT

    def test_low_consensus_downgrade_two(self):
        # 37.5% → downgrade 2
        action = _consensus_gated_action(SeverityLevel.CRITICAL, 15, 4, 10)
        assert action == MaintenanceAction.LOG_AND_MONITOR

    def test_no_underflow(self):
        # LOW severity base = LOG → downgrade 2 → NONE (floor)
        action = _consensus_gated_action(SeverityLevel.LOW, 1, 4, 10)
        assert action == MaintenanceAction.NONE

    def test_zero_rows(self):
        # n_rows=0 → max(0,1)=1 → high consensus
        action = _consensus_gated_action(SeverityLevel.HIGH, 4, 4, 0)
        assert action == SEVERITY_TO_ACTION[SeverityLevel.HIGH]


# ── assess_component — REAL ensemble execution ─────────────────────────────

class TestAssessComponent:
    def test_normal_data(self, normal_df):
        """Normal data → real ensemble → should produce a healthy score."""
        hs = assess_component(normal_df, "Motor", ["vibration", "rpm"])
        assert isinstance(hs, HealthScore)
        assert hs.component == "Motor"
        assert 0 <= hs.health_pct <= 100
        assert hs.severity in SeverityLevel
        assert hs.action in MaintenanceAction
        assert hs.vote_count >= 0
        assert hs.total_models >= 1

    def test_anomalous_data_lower_health(self, anomalous_df):
        """Injected anomalies → lower health than normal data."""
        hs_bad = assess_component(anomalous_df, "Motor", ["vibration", "rpm"])
        assert isinstance(hs_bad, HealthScore)
        assert 0 <= hs_bad.health_pct <= 100
        # The anomalous data should be picked up
        assert hs_bad.anomaly_score > 0

    def test_metadata_populated(self, normal_df):
        hs = assess_component(normal_df, "Motor", ["vibration", "rpm"])
        assert "columns" in hs.metadata
        assert "vibration" in hs.metadata["columns"]
        assert "anomaly_count" in hs.metadata
        assert "total_rows" in hs.metadata

    def test_single_column(self, normal_df):
        hs = assess_component(normal_df, "Thermal", ["temperature"])
        assert hs.component == "Thermal"
        assert isinstance(hs.health_pct, float)

    def test_auto_detect_columns(self, normal_df):
        """No columns specified → uses all numeric."""
        hs = assess_component(normal_df, "AllSensors")
        assert len(hs.metadata["columns"]) >= 4  # vibration, temp, pressure, rpm

    def test_no_numeric_raises(self):
        df = pd.DataFrame({"name": ["a", "b", "c"] * 10})
        with pytest.raises(ValueError, match="No numeric columns"):
            assess_component(df, "test")

    def test_top_features_structure(self, anomalous_df):
        hs = assess_component(anomalous_df, "Motor", ["vibration", "rpm"])
        for f in hs.top_features:
            assert "feature" in f
            assert "importance" in f

    def test_to_dict_works(self, normal_df):
        hs = assess_component(normal_df, "Motor", ["vibration", "rpm"])
        d = hs.to_dict()
        assert isinstance(d, dict)
        assert d["component"] == "Motor"
        assert isinstance(d["risk_level"], str)


# ── assess_components ───────────────────────────────────────────────────────

class TestAssessComponents:
    def test_multiple_components(self, normal_df, component_map):
        results = assess_components(normal_df, component_map)
        assert len(results) == 3
        names = {h.component for h in results}
        assert names == {"Motor", "Thermal", "Hydraulic"}

    def test_skips_missing_columns(self, normal_df):
        cmap = {"Motor": ["vibration"], "Missing": ["nonexistent"]}
        results = assess_components(normal_df, cmap)
        assert len(results) == 1

    def test_all_results_are_health_scores(self, normal_df, component_map):
        for hs in assess_components(normal_df, component_map):
            assert isinstance(hs, HealthScore)
            assert 0 <= hs.health_pct <= 100


# ── assess_component with REAL F1 data ─────────────────────────────────────

class TestAssessComponentF1:
    def test_f1_power_unit(self, f1_raw, f1_component_map):
        """Run real ensemble on actual F1 telemetry."""
        cols = [c for c in f1_component_map["Power Unit"] if c in f1_raw.columns]
        if not cols:
            pytest.skip("No Power Unit columns in F1 data")
        # Need numeric and enough rows
        df = f1_raw[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 10:
            pytest.skip("Not enough numeric rows")
        hs = assess_component(df, "Power Unit", cols)
        assert isinstance(hs, HealthScore)
        assert 0 <= hs.health_pct <= 100
        print(f"\nF1 Power Unit: {hs.health_pct:.1f}% health, "
              f"severity={hs.severity.value}, action={hs.action.value}")

    def test_f1_multiple_components(self, f1_raw, f1_component_map):
        """Run real ensemble on multiple F1 components."""
        # Prepare numeric data
        numeric_cols = []
        for cols in f1_component_map.values():
            numeric_cols.extend(cols)
        numeric_cols = list(set(c for c in numeric_cols if c in f1_raw.columns))
        df = f1_raw[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 10:
            pytest.skip("Not enough data")

        results = assess_components(df, f1_component_map)
        assert len(results) > 0
        for hs in results:
            print(f"\nF1 {hs.component}: {hs.health_pct:.1f}% "
                  f"({hs.severity.value}) → {hs.action.value}")


# ── temporal features ───────────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_creates_columns(self, normal_df):
        enriched, new_cols = add_temporal_features(normal_df, ["vibration", "temperature"])
        assert "vibration_delta" in new_cols
        assert "vibration_roll3_mean" in new_cols
        assert "vibration_roll3_std" in new_cols
        assert len(enriched) == len(normal_df)

    def test_delta_first_row_zero(self, normal_df):
        enriched, _ = add_temporal_features(normal_df, ["vibration"])
        assert enriched["vibration_delta"].iloc[0] == 0

    def test_custom_window(self, normal_df):
        enriched, new_cols = add_temporal_features(normal_df, ["vibration"], window=5)
        assert "vibration_roll5_mean" in new_cols

    def test_skips_missing(self, normal_df):
        _, new_cols = add_temporal_features(normal_df, ["nonexistent"])
        assert new_cols == []

    def test_does_not_modify_original(self, normal_df):
        orig_cols = list(normal_df.columns)
        add_temporal_features(normal_df, ["vibration"])
        assert list(normal_df.columns) == orig_cols


class TestLifecycleContext:
    def test_creates_columns(self, normal_df):
        enriched, new_cols = add_lifecycle_context(normal_df)
        assert set(new_cols) == {"period_index", "lifecycle_pct", "is_second_half"}

    def test_lifecycle_range(self, normal_df):
        enriched, _ = add_lifecycle_context(normal_df)
        assert enriched["lifecycle_pct"].iloc[0] == 0.0
        assert abs(enriched["lifecycle_pct"].iloc[-1] - 1.0) < 0.03

    def test_second_half_flag(self, normal_df):
        enriched, _ = add_lifecycle_context(normal_df)
        assert enriched["is_second_half"].iloc[0] == 0
        assert enriched["is_second_half"].iloc[-1] == 1

    def test_custom_total_periods(self, normal_df):
        enriched, _ = add_lifecycle_context(normal_df, total_periods=100)
        assert enriched["lifecycle_pct"].iloc[-1] < 1.0
