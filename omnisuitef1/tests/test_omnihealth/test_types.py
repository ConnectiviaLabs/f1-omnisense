"""Tests for omnihealth._types — enums, dataclasses, serialization roundtrips."""

import time

import pytest
from omnianalytics._types import SeverityLevel
from omnihealth._types import (
    PRIORITY_HOURS,
    SENSOR_UNITS,
    SEVERITY_TO_ACTION,
    HealthReport,
    HealthScore,
    MaintenanceAction,
    MaintenancePriority,
    MaintenanceSchedule,
    MaintenanceTask,
    RiskAssessment,
    RiskLevel,
    TimeSeriesAnalysis,
    TrendDirection,
)


# ── Enum basics ─────────────────────────────────────────────────────────────

class TestEnums:
    def test_risk_level_values(self):
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.LOW.value == "low"

    def test_risk_level_from_string(self):
        assert RiskLevel("high") == RiskLevel.HIGH

    def test_maintenance_priority_all_values(self):
        expected = {"critical", "high", "medium", "low", "routine"}
        assert {p.value for p in MaintenancePriority} == expected

    def test_maintenance_action_all_values(self):
        expected = {"alert_and_remediate", "alert", "log_and_monitor", "log", "none"}
        assert {a.value for a in MaintenanceAction} == expected

    def test_trend_direction_values(self):
        assert TrendDirection.INCREASING.value == "increasing"
        assert TrendDirection.DECREASING.value == "decreasing"
        assert TrendDirection.STABLE.value == "stable"

    def test_enums_are_str(self):
        """All enums inherit from str, so they work as dict keys and JSON values."""
        assert isinstance(RiskLevel.HIGH, str)
        assert isinstance(MaintenancePriority.CRITICAL, str)
        assert isinstance(MaintenanceAction.ALERT, str)
        assert isinstance(TrendDirection.STABLE, str)


# ── Constants ───────────────────────────────────────────────────────────────

class TestConstants:
    def test_severity_to_action_covers_all(self):
        for sev in SeverityLevel:
            assert sev in SEVERITY_TO_ACTION

    def test_priority_hours_covers_all(self):
        for p in MaintenancePriority:
            assert p in PRIORITY_HOURS
            assert PRIORITY_HOURS[p] > 0

    def test_priority_hours_ordering(self):
        """CRITICAL < HIGH < MEDIUM < LOW < ROUTINE."""
        assert PRIORITY_HOURS[MaintenancePriority.CRITICAL] < PRIORITY_HOURS[MaintenancePriority.HIGH]
        assert PRIORITY_HOURS[MaintenancePriority.HIGH] < PRIORITY_HOURS[MaintenancePriority.MEDIUM]
        assert PRIORITY_HOURS[MaintenancePriority.MEDIUM] < PRIORITY_HOURS[MaintenancePriority.LOW]
        assert PRIORITY_HOURS[MaintenancePriority.LOW] < PRIORITY_HOURS[MaintenancePriority.ROUTINE]

    def test_sensor_units_structure(self):
        for key, val in SENSOR_UNITS.items():
            assert isinstance(key, str)
            assert isinstance(val, tuple)
            assert len(val) == 2


# ── HealthScore ─────────────────────────────────────────────────────────────

class TestHealthScore:
    def _make(self, **overrides):
        defaults = dict(
            component="Motor",
            health_pct=85.0,
            risk_level=RiskLevel.LOW,
            severity=SeverityLevel.LOW,
            action=MaintenanceAction.LOG,
            anomaly_score=0.19,
            vote_count=10,
            total_models=4,
            top_features=[{"feature": "vibration", "importance": 0.8}],
            metadata={"columns": ["vibration", "rpm"]},
        )
        defaults.update(overrides)
        return HealthScore(**defaults)

    def test_to_dict_serializes_enums(self):
        hs = self._make()
        d = hs.to_dict()
        assert d["risk_level"] == "low"
        assert d["severity"] == "low"
        assert d["action"] == "log"

    def test_from_dict_roundtrip(self):
        hs = self._make()
        d = hs.to_dict()
        restored = HealthScore.from_dict(d)
        assert restored.component == hs.component
        assert restored.health_pct == hs.health_pct
        assert restored.risk_level == hs.risk_level
        assert restored.severity == hs.severity
        assert restored.action == hs.action

    def test_to_dict_includes_top_features(self):
        hs = self._make()
        d = hs.to_dict()
        assert len(d["top_features"]) == 1
        assert d["top_features"][0]["feature"] == "vibration"

    def test_defaults(self):
        hs = HealthScore(
            component="X", health_pct=50, risk_level=RiskLevel.LOW,
            severity=SeverityLevel.NORMAL, action=MaintenanceAction.NONE,
            anomaly_score=0.5, vote_count=0, total_models=4,
        )
        assert hs.top_features == []
        assert hs.metadata == {}


# ── TimeSeriesAnalysis ──────────────────────────────────────────────────────

class TestTimeSeriesAnalysis:
    def _make(self, **overrides):
        defaults = dict(
            trend=TrendDirection.INCREASING,
            trend_strength=0.8,
            slope=0.05,
            drift_rate=0.05,
            is_stationary=False,
            seasonality_detected=True,
            seasonal_period=7,
            seasonal_strength=0.6,
            anomaly_count=3,
            anomaly_pct=3.0,
            forecastability_score=75.0,
            forecastability_rating="High",
            operational_zones={"optimal_low": 40, "optimal_high": 60},
            recommendations=["Monitor trend"],
        )
        defaults.update(overrides)
        return TimeSeriesAnalysis(**defaults)

    def test_to_dict_serializes_trend(self):
        ts = self._make()
        d = ts.to_dict()
        assert d["trend"] == "increasing"

    def test_from_dict_roundtrip(self):
        ts = self._make()
        d = ts.to_dict()
        restored = TimeSeriesAnalysis.from_dict(d)
        assert restored.trend == TrendDirection.INCREASING
        assert restored.seasonal_period == 7

    def test_optional_seasonal_period_none(self):
        ts = self._make(seasonal_period=None, seasonality_detected=False)
        d = ts.to_dict()
        assert d["seasonal_period"] is None


# ── RiskAssessment ──────────────────────────────────────────────────────────

class TestRiskAssessment:
    def _make(self, **overrides):
        defaults = dict(
            feature="vibration",
            current_value=5.0,
            forecast_value=6.5,
            risk_level=RiskLevel.MEDIUM,
            trend=TrendDirection.INCREASING,
            trend_pct=30.0,
            confidence_lower=5.5,
            confidence_upper=7.5,
        )
        defaults.update(overrides)
        return RiskAssessment(**defaults)

    def test_to_dict_basic(self):
        ra = self._make()
        d = ra.to_dict()
        assert d["risk_level"] == "medium"
        assert d["trend"] == "increasing"
        assert d["trend_pct"] == 30.0

    def test_from_dict_roundtrip(self):
        ra = self._make()
        d = ra.to_dict()
        restored = RiskAssessment.from_dict(d)
        assert restored.feature == "vibration"
        assert restored.risk_level == RiskLevel.MEDIUM

    def test_with_time_series_analysis(self):
        tsa = TimeSeriesAnalysis(
            trend=TrendDirection.INCREASING, trend_strength=0.8, slope=0.05,
            drift_rate=0.05, is_stationary=False, seasonality_detected=False,
            seasonal_period=None, seasonal_strength=0.0, anomaly_count=0,
            anomaly_pct=0.0, forecastability_score=70.0, forecastability_rating="Medium",
        )
        ra = self._make(time_series_analysis=tsa)
        d = ra.to_dict()
        assert "time_series_analysis" in d
        assert d["time_series_analysis"]["trend"] == "increasing"

    def test_from_dict_with_nested_tsa(self):
        tsa = TimeSeriesAnalysis(
            trend=TrendDirection.STABLE, trend_strength=0.1, slope=0.0,
            drift_rate=0.0, is_stationary=True, seasonality_detected=False,
            seasonal_period=None, seasonal_strength=0.0, anomaly_count=0,
            anomaly_pct=0.0, forecastability_score=50.0, forecastability_rating="Medium",
        )
        ra = self._make(time_series_analysis=tsa)
        d = ra.to_dict()
        restored = RiskAssessment.from_dict(d)
        assert restored.time_series_analysis is not None
        assert restored.time_series_analysis.trend == TrendDirection.STABLE

    def test_with_horizon_results(self):
        hr = [{"horizon": 5, "risk_level": "low"}, {"horizon": 10, "risk_level": "medium"}]
        ra = self._make(horizon_results=hr)
        d = ra.to_dict()
        assert d["horizon_results"] == hr

    def test_degradation_rate_optional(self):
        ra = self._make(degradation_rate=None)
        d = ra.to_dict()
        assert d["degradation_rate"] is None

        ra2 = self._make(degradation_rate=0.003)
        d2 = ra2.to_dict()
        assert d2["degradation_rate"] == 0.003


# ── MaintenanceTask ─────────────────────────────────────────────────────────

class TestMaintenanceTask:
    def test_to_dict(self):
        task = MaintenanceTask(
            task_id="abc123",
            component="Motor",
            feature="vibration",
            priority=MaintenancePriority.HIGH,
            action=MaintenanceAction.ALERT,
            description="Motor needs review",
            reason="Severity: high; Health: 60%",
            estimated_hours=24,
        )
        d = task.to_dict()
        assert d["priority"] == "high"
        assert d["action"] == "alert"
        assert d["task_id"] == "abc123"
        assert "created_at" in d

    def test_created_at_default(self):
        before = time.time()
        task = MaintenanceTask(
            task_id="x", component="C", feature="f",
            priority=MaintenancePriority.LOW, action=MaintenanceAction.LOG,
            description="", reason="", estimated_hours=168,
        )
        after = time.time()
        assert before <= task.created_at <= after

    def test_to_dict_with_nested_risk_and_health(self):
        from omnihealth._types import HealthScore, RiskAssessment
        hs = HealthScore(
            component="M", health_pct=70, risk_level=RiskLevel.MEDIUM,
            severity=SeverityLevel.MEDIUM, action=MaintenanceAction.LOG_AND_MONITOR,
            anomaly_score=0.38, vote_count=5, total_models=4,
        )
        ra = RiskAssessment(
            feature="v", current_value=5.0, forecast_value=7.0,
            risk_level=RiskLevel.MEDIUM, trend=TrendDirection.INCREASING,
            trend_pct=40.0, confidence_lower=4.0, confidence_upper=8.0,
        )
        task = MaintenanceTask(
            task_id="t1", component="M", feature="v",
            priority=MaintenancePriority.MEDIUM, action=MaintenanceAction.LOG_AND_MONITOR,
            description="test", reason="test", estimated_hours=72,
            risk_assessment=ra, health_score=hs,
        )
        d = task.to_dict()
        assert "risk_assessment" in d
        assert "health_score" in d
        assert d["risk_assessment"]["risk_level"] == "medium"


# ── MaintenanceSchedule ─────────────────────────────────────────────────────

class TestMaintenanceSchedule:
    def test_to_dict(self):
        task = MaintenanceTask(
            task_id="t1", component="A", feature="x",
            priority=MaintenancePriority.CRITICAL, action=MaintenanceAction.ALERT_AND_REMEDIATE,
            description="test", reason="test", estimated_hours=2,
        )
        schedule = MaintenanceSchedule(
            generated_at="2026-01-01T00:00:00Z",
            total_tasks=1,
            priority_breakdown={"critical": 1},
            tasks=[task],
            summary="1 action item(s) generated.",
        )
        d = schedule.to_dict()
        assert d["total_tasks"] == 1
        assert len(d["tasks"]) == 1
        assert d["tasks"][0]["priority"] == "critical"

    def test_empty_schedule(self):
        schedule = MaintenanceSchedule(
            generated_at="2026-01-01T00:00:00Z",
            total_tasks=0,
            priority_breakdown={},
            tasks=[],
            summary="No action items.",
        )
        d = schedule.to_dict()
        assert d["total_tasks"] == 0
        assert d["tasks"] == []


# ── HealthReport ────────────────────────────────────────────────────────────

class TestHealthReport:
    def test_to_dict_full(self):
        hs = HealthScore(
            component="Motor", health_pct=90.0, risk_level=RiskLevel.LOW,
            severity=SeverityLevel.NORMAL, action=MaintenanceAction.NONE,
            anomaly_score=0.12, vote_count=5, total_models=4,
        )
        schedule = MaintenanceSchedule(
            generated_at="2026-01-01T00:00:00Z", total_tasks=0,
            priority_breakdown={}, tasks=[], summary="All clear.",
        )
        report = HealthReport(
            components=[hs],
            risk_assessments=[],
            schedule=schedule,
            overall_health=90.0,
            overall_risk=RiskLevel.LOW,
            generated_at="2026-01-01T00:00:00Z",
        )
        d = report.to_dict()
        assert d["overall_health"] == 90.0
        assert d["overall_risk"] == "low"
        assert len(d["components"]) == 1
        assert d["components"][0]["component"] == "Motor"
