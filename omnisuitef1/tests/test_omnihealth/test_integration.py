"""Integration tests — full assess() pipeline, real execution, no mocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omnianalytics._types import SeverityLevel
from omnihealth import assess
from omnihealth._types import (
    HealthReport,
    HealthScore,
    MaintenanceAction,
    MaintenancePriority,
    MaintenanceSchedule,
    RiskAssessment,
    RiskLevel,
)


# ── Full pipeline on synthetic data ─────────────────────────────────────────

class TestAssessPipeline:
    def test_normal_data_full_pipeline(self, normal_df, component_map):
        """Normal data → full pipeline → healthy report."""
        report = assess(normal_df, component_map)

        assert isinstance(report, HealthReport)
        assert len(report.components) == 3
        assert 0 <= report.overall_health <= 100
        assert report.overall_risk in RiskLevel
        assert isinstance(report.schedule, MaintenanceSchedule)
        assert "T" in report.generated_at  # ISO format

    def test_anomalous_data_full_pipeline(self, anomalous_df, component_map):
        """Anomalous data → should detect issues and generate tasks."""
        report = assess(anomalous_df, component_map)

        assert isinstance(report, HealthReport)
        assert len(report.components) == 3
        # At least Motor should show some anomaly
        motor = next((c for c in report.components if c.component == "Motor"), None)
        assert motor is not None
        assert motor.anomaly_score > 0

    def test_skip_schedule(self, normal_df, component_map):
        """include_schedule=False → empty schedule."""
        report = assess(normal_df, component_map, include_schedule=False)
        assert report.schedule.total_tasks == 0
        assert "skipped" in report.schedule.summary.lower()

    def test_report_to_dict(self, normal_df, component_map):
        """Full report serializes without error."""
        report = assess(normal_df, component_map)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "components" in d
        assert "risk_assessments" in d
        assert "schedule" in d
        assert "overall_health" in d
        assert "overall_risk" in d
        assert isinstance(d["overall_risk"], str)

    def test_single_component(self, normal_df):
        report = assess(normal_df, {"Motor": ["vibration", "rpm"]})
        assert len(report.components) == 1

    def test_no_matching_columns(self, normal_df):
        """No matching columns → empty report, 100% health."""
        report = assess(normal_df, {"Missing": ["nonexistent"]})
        assert len(report.components) == 0
        assert report.overall_health == 100.0
        assert report.overall_risk == RiskLevel.LOW

    def test_no_timestamp(self):
        """Data without timestamp → still works."""
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.normal(10, 1, 50),
            "b": np.random.normal(20, 2, 50),
        })
        report = assess(df, {"System": ["a", "b"]})
        assert isinstance(report, HealthReport)
        assert len(report.components) == 1

    def test_include_timeseries(self, normal_df, component_map):
        """include_timeseries=True shouldn't crash."""
        report = assess(normal_df, component_map, include_timeseries=True)
        assert isinstance(report, HealthReport)


# ── Full pipeline on REAL F1 data ──────────────────────────────────────────

class TestAssessF1:
    def test_f1_full_pipeline(self, f1_raw, f1_component_map):
        """Run the ENTIRE pipeline on real F1 telemetry."""
        # Prepare numeric data
        all_cols = set()
        for cols in f1_component_map.values():
            all_cols.update(cols)
        available = [c for c in all_cols if c in f1_raw.columns]
        if not available:
            pytest.skip("No F1 columns available")

        df = f1_raw[available].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 20:
            pytest.skip("Not enough numeric rows")

        report = assess(df, f1_component_map, horizon=5)

        assert isinstance(report, HealthReport)
        assert len(report.components) > 0

        print(f"\n{'='*60}")
        print(f"F1 Full Pipeline Results")
        print(f"{'='*60}")
        print(f"Overall health: {report.overall_health:.1f}%")
        print(f"Overall risk: {report.overall_risk.value}")
        print(f"Components assessed: {len(report.components)}")
        for h in report.components:
            print(f"  {h.component}: {h.health_pct:.1f}% "
                  f"(severity={h.severity.value}, action={h.action.value})")
            if h.top_features:
                for f in h.top_features[:3]:
                    print(f"    → {f['feature']}: importance={f['importance']:.3f}")
        print(f"Risk assessments: {len(report.risk_assessments)}")
        for r in report.risk_assessments:
            print(f"  {r.feature}: risk={r.risk_level.value}, "
                  f"trend={r.trend.value} ({r.trend_pct:+.1f}%)")
        print(f"Schedule: {report.schedule.total_tasks} tasks")
        for t in report.schedule.tasks:
            print(f"  [{t.priority.value}] {t.component}/{t.feature}: {t.description}")
        print(f"Summary: {report.schedule.summary}")

    def test_f1_report_serializable(self, f1_raw, f1_component_map):
        """F1 report → to_dict() → valid JSON-like structure."""
        all_cols = set()
        for cols in f1_component_map.values():
            all_cols.update(cols)
        available = [c for c in all_cols if c in f1_raw.columns]
        if not available:
            pytest.skip("No F1 columns")

        df = f1_raw[available].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 20:
            pytest.skip("Not enough data")

        report = assess(df, f1_component_map, horizon=5)
        d = report.to_dict()

        # Validate structure
        assert isinstance(d["components"], list)
        assert isinstance(d["risk_assessments"], list)
        assert isinstance(d["schedule"], dict)
        assert isinstance(d["schedule"]["tasks"], list)

        # All enum values should be strings
        for comp in d["components"]:
            assert isinstance(comp["risk_level"], str)
            assert isinstance(comp["severity"], str)
            assert isinstance(comp["action"], str)


# ── Edge cases with real execution ──────────────────────────────────────────

class TestEdgeCases:
    def test_constant_data(self):
        """All-constant data → should handle gracefully."""
        df = pd.DataFrame({
            "x": [5.0] * 50,
            "y": [10.0] * 50,
        })
        report = assess(df, {"System": ["x", "y"]})
        assert isinstance(report, HealthReport)

    def test_single_row(self):
        """Single row → ensemble should handle it (may fail gracefully)."""
        df = pd.DataFrame({"x": [5.0], "y": [10.0]})
        # This will likely fail inside AnomalyEnsemble (needs >=10 rows)
        # but assess() should handle the exception
        report = assess(df, {"System": ["x", "y"]})
        # Either 0 components (failed gracefully) or 1
        assert isinstance(report, HealthReport)

    def test_mixed_dtypes(self):
        """DataFrame with mixed types → only numeric used."""
        np.random.seed(42)
        df = pd.DataFrame({
            "name": ["sensor_" + str(i) for i in range(50)],
            "value": np.random.normal(10, 1, 50),
            "status": ["ok"] * 50,
        })
        report = assess(df, {"System": ["value"]})
        assert len(report.components) == 1

    def test_large_synthetic(self):
        """1000-row dataset → smoke test for performance."""
        np.random.seed(42)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=1000, freq="h"),
            "a": np.random.normal(0, 1, 1000),
            "b": np.random.normal(0, 1, 1000),
            "c": np.random.normal(0, 1, 1000),
        })
        report = assess(df, {"AB": ["a", "b"], "C": ["c"]}, horizon=10)
        assert isinstance(report, HealthReport)
        assert len(report.components) == 2
