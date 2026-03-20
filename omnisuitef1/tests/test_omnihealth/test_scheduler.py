"""Tests for omnihealth.scheduler — priority, sampling inference, scaling, task/schedule generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omnianalytics._types import SeverityLevel
from omnihealth._types import (
    HealthScore,
    MaintenanceAction,
    MaintenancePriority,
    MaintenanceSchedule,
    MaintenanceTask,
    RiskAssessment,
    RiskLevel,
    TrendDirection,
    PRIORITY_HOURS,
)
from omnihealth.scheduler import (
    _scale_priority_hours,
    determine_priority,
    estimate_completion_hours,
    generate_schedule,
    generate_task,
    infer_sampling_hours,
)


# ── determine_priority ──────────────────────────────────────────────────────

class TestDeterminePriority:
    def test_critical_risk_high_severity(self):
        """risk=HIGH + severity>=HIGH → CRITICAL."""
        assert determine_priority(RiskLevel.HIGH, SeverityLevel.HIGH) == MaintenancePriority.CRITICAL
        assert determine_priority(RiskLevel.HIGH, SeverityLevel.CRITICAL) == MaintenancePriority.CRITICAL

    def test_high_risk_alone(self):
        """risk=HIGH + lower severity → HIGH."""
        assert determine_priority(RiskLevel.HIGH, SeverityLevel.LOW) == MaintenancePriority.HIGH
        assert determine_priority(RiskLevel.HIGH, SeverityLevel.MEDIUM) == MaintenancePriority.HIGH

    def test_high_severity_increasing_trend(self):
        """severity=HIGH + trend=INCREASING → HIGH."""
        assert determine_priority(
            RiskLevel.LOW, SeverityLevel.HIGH, TrendDirection.INCREASING,
        ) == MaintenancePriority.HIGH

    def test_critical_severity_without_high_risk(self):
        """severity=CRITICAL but risk=LOW → CRITICAL (severity override)."""
        assert determine_priority(RiskLevel.LOW, SeverityLevel.CRITICAL) == MaintenancePriority.CRITICAL

    def test_medium_risk_or_severity(self):
        assert determine_priority(RiskLevel.MEDIUM, SeverityLevel.LOW) == MaintenancePriority.MEDIUM
        assert determine_priority(RiskLevel.LOW, SeverityLevel.MEDIUM) == MaintenancePriority.MEDIUM

    def test_low_severity(self):
        assert determine_priority(RiskLevel.LOW, SeverityLevel.LOW) == MaintenancePriority.LOW

    def test_normal_severity_routine(self):
        assert determine_priority(RiskLevel.LOW, SeverityLevel.NORMAL) == MaintenancePriority.ROUTINE

    def test_trend_none_defaults(self):
        """No trend passed → should still work."""
        result = determine_priority(RiskLevel.LOW, SeverityLevel.HIGH)
        assert result in MaintenancePriority


# ── infer_sampling_hours ────────────────────────────────────────────────────

class TestInferSamplingHours:
    def test_hourly_data(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=50, freq="h"),
            "value": range(50),
        })
        hours = infer_sampling_hours(df)
        assert abs(hours - 1.0) < 0.1

    def test_daily_data(self, daily_df):
        hours = infer_sampling_hours(daily_df)
        assert abs(hours - 24.0) < 1.0

    def test_minute_data(self, minute_df):
        hours = infer_sampling_hours(minute_df)
        assert abs(hours - 1 / 60) < 0.01

    def test_no_timestamp(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert infer_sampling_hours(df) == 1.0

    def test_none_data(self):
        assert infer_sampling_hours(None) == 1.0

    def test_single_row(self):
        df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")], "v": [1]})
        assert infer_sampling_hours(df) == 1.0

    def test_weekly_data(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="W"),
            "value": range(20),
        })
        hours = infer_sampling_hours(df)
        assert abs(hours - 168.0) < 10.0


# ── _scale_priority_hours ──────────────────────────────────────────────────

class TestScalePriorityHours:
    def test_hourly_scale(self):
        scaled = _scale_priority_hours(1.0)
        assert scaled[MaintenancePriority.CRITICAL] == 2
        assert scaled[MaintenancePriority.HIGH] == 24

    def test_daily_scale(self):
        scaled = _scale_priority_hours(24.0)
        assert scaled[MaintenancePriority.CRITICAL] == 48
        assert scaled[MaintenancePriority.HIGH] == 576

    def test_minute_scale_minimums(self):
        """Very fast sampling → minimums kick in."""
        scaled = _scale_priority_hours(1 / 60)
        assert scaled[MaintenancePriority.CRITICAL] >= 1
        assert scaled[MaintenancePriority.HIGH] >= 2
        assert scaled[MaintenancePriority.MEDIUM] >= 4
        assert scaled[MaintenancePriority.LOW] >= 8
        assert scaled[MaintenancePriority.ROUTINE] >= 24

    def test_ordering_preserved(self):
        for scale in [0.1, 1.0, 24.0, 168.0]:
            scaled = _scale_priority_hours(scale)
            assert scaled[MaintenancePriority.CRITICAL] <= scaled[MaintenancePriority.HIGH]
            assert scaled[MaintenancePriority.HIGH] <= scaled[MaintenancePriority.MEDIUM]
            assert scaled[MaintenancePriority.MEDIUM] <= scaled[MaintenancePriority.LOW]
            assert scaled[MaintenancePriority.LOW] <= scaled[MaintenancePriority.ROUTINE]


# ── estimate_completion_hours ───────────────────────────────────────────────

class TestEstimateCompletionHours:
    def test_no_trend(self):
        hours = estimate_completion_hours(MaintenancePriority.CRITICAL)
        assert hours == PRIORITY_HOURS[MaintenancePriority.CRITICAL]

    def test_high_trend_shrinks(self):
        base = PRIORITY_HOURS[MaintenancePriority.MEDIUM]
        hours = estimate_completion_hours(MaintenancePriority.MEDIUM, trend_pct=15.0)
        assert hours < base  # 0.4x factor
        assert hours == max(1, int(base * 0.4))

    def test_medium_trend(self):
        base = PRIORITY_HOURS[MaintenancePriority.MEDIUM]
        hours = estimate_completion_hours(MaintenancePriority.MEDIUM, trend_pct=7.0)
        assert hours == max(1, int(base * 0.6))

    def test_slight_trend(self):
        base = PRIORITY_HOURS[MaintenancePriority.MEDIUM]
        hours = estimate_completion_hours(MaintenancePriority.MEDIUM, trend_pct=3.0)
        assert hours == max(1, int(base * 0.75))

    def test_negative_trend_same_shrink(self):
        """Absolute trend matters, not direction."""
        hours_pos = estimate_completion_hours(MaintenancePriority.HIGH, trend_pct=8.0)
        hours_neg = estimate_completion_hours(MaintenancePriority.HIGH, trend_pct=-8.0)
        assert hours_pos == hours_neg

    def test_minimum_one_hour(self):
        hours = estimate_completion_hours(MaintenancePriority.CRITICAL, trend_pct=50.0)
        assert hours >= 1

    def test_custom_priority_hours(self):
        custom = {MaintenancePriority.CRITICAL: 100}
        hours = estimate_completion_hours(MaintenancePriority.CRITICAL, priority_hours=custom)
        assert hours == 100


# ── generate_task ───────────────────────────────────────────────────────────

class TestGenerateTask:
    def _make_health(self, **overrides):
        defaults = dict(
            component="Motor", health_pct=60.0, risk_level=RiskLevel.MEDIUM,
            severity=SeverityLevel.HIGH, action=MaintenanceAction.ALERT,
            anomaly_score=0.5, vote_count=20, total_models=4,
            metadata={"columns": ["vibration", "rpm"]},
        )
        defaults.update(overrides)
        return HealthScore(**defaults)

    def _make_risk(self, **overrides):
        defaults = dict(
            feature="vibration", current_value=5.0, forecast_value=7.0,
            risk_level=RiskLevel.HIGH, trend=TrendDirection.INCREASING,
            trend_pct=40.0, confidence_lower=4.0, confidence_upper=8.0,
        )
        defaults.update(overrides)
        return RiskAssessment(**defaults)

    def test_basic_task_generation(self):
        health = self._make_health()
        task = generate_task("Motor", "vibration", health)
        assert isinstance(task, MaintenanceTask)
        assert task.component == "Motor"
        assert task.feature == "vibration"
        assert len(task.task_id) == 12

    def test_with_risk_assessment(self):
        health = self._make_health()
        risk = self._make_risk()
        task = generate_task("Motor", "vibration", health, risk)
        assert task.risk_assessment is not None
        assert task.health_score is not None

    def test_uses_consensus_gated_action(self):
        """Task should use health.action (already consensus-gated), not static map."""
        health = self._make_health(action=MaintenanceAction.LOG_AND_MONITOR)
        task = generate_task("Motor", "vibration", health)
        assert task.action == MaintenanceAction.LOG_AND_MONITOR

    def test_description_is_readable(self):
        health = self._make_health()
        risk = self._make_risk()
        task = generate_task("Motor", "vibration", health, risk)
        assert "Motor" in task.description
        assert "health:" in task.description.lower() or "%" in task.description

    def test_reason_includes_severity(self):
        health = self._make_health()
        task = generate_task("Motor", "overall", health)
        assert "Severity" in task.reason or "severity" in task.reason


# ── generate_schedule ───────────────────────────────────────────────────────

class TestGenerateSchedule:
    def _make_health(self, component, severity, action=MaintenanceAction.ALERT, risk=RiskLevel.MEDIUM, cols=None):
        return HealthScore(
            component=component, health_pct=50.0, risk_level=risk,
            severity=severity, action=action,
            anomaly_score=0.6, vote_count=20, total_models=4,
            metadata={"columns": cols or []},
        )

    def _make_risk(self, feature, trend=TrendDirection.INCREASING, risk=RiskLevel.MEDIUM, trend_pct=5.0):
        return RiskAssessment(
            feature=feature, current_value=5.0, forecast_value=6.0,
            risk_level=risk, trend=trend, trend_pct=trend_pct,
            confidence_lower=4.0, confidence_upper=7.0,
        )

    def test_empty_schedule_for_normal(self):
        health = [self._make_health("Motor", SeverityLevel.NORMAL, MaintenanceAction.NONE)]
        schedule = generate_schedule(health)
        assert isinstance(schedule, MaintenanceSchedule)
        assert schedule.total_tasks == 0

    def test_generates_tasks_for_non_normal(self):
        health = [self._make_health("Motor", SeverityLevel.HIGH)]
        schedule = generate_schedule(health)
        assert schedule.total_tasks >= 1

    def test_sorted_by_priority(self):
        healths = [
            self._make_health("A", SeverityLevel.LOW, MaintenanceAction.LOG, RiskLevel.LOW),
            self._make_health("B", SeverityLevel.CRITICAL, MaintenanceAction.ALERT_AND_REMEDIATE, RiskLevel.HIGH),
        ]
        schedule = generate_schedule(healths)
        if schedule.total_tasks >= 2:
            priorities = [t.priority for t in schedule.tasks]
            # CRITICAL should come before LOW
            priority_order = {
                MaintenancePriority.CRITICAL: 1,
                MaintenancePriority.HIGH: 2,
                MaintenancePriority.MEDIUM: 3,
                MaintenancePriority.LOW: 4,
                MaintenancePriority.ROUTINE: 5,
            }
            orders = [priority_order[p] for p in priorities]
            assert orders == sorted(orders)

    def test_failure_mode_splitting(self):
        """Different trend directions → separate tasks for same component."""
        health = [self._make_health(
            "Motor", SeverityLevel.HIGH,
            cols=["vibration", "rpm"],
        )]
        risks = [
            self._make_risk("vibration", TrendDirection.INCREASING, trend_pct=10.0),
            self._make_risk("rpm", TrendDirection.DECREASING, trend_pct=-8.0),
        ]
        schedule = generate_schedule(health, risks)
        # Should have 2 tasks: one for increasing, one for decreasing
        assert schedule.total_tasks == 2
        features = {t.feature for t in schedule.tasks}
        assert "vibration" in features
        assert "rpm" in features

    def test_same_direction_grouped(self):
        """Same trend direction → single task (worst risk within mode)."""
        health = [self._make_health(
            "Motor", SeverityLevel.HIGH,
            cols=["vibration", "rpm"],
        )]
        risks = [
            self._make_risk("vibration", TrendDirection.INCREASING, RiskLevel.HIGH, trend_pct=15.0),
            self._make_risk("rpm", TrendDirection.INCREASING, RiskLevel.MEDIUM, trend_pct=5.0),
        ]
        schedule = generate_schedule(health, risks)
        assert schedule.total_tasks == 1
        assert schedule.tasks[0].feature == "vibration"  # worst risk

    def test_priority_breakdown(self):
        health = [
            self._make_health("A", SeverityLevel.HIGH, MaintenanceAction.ALERT, RiskLevel.HIGH, ["a"]),
            self._make_health("B", SeverityLevel.MEDIUM, MaintenanceAction.LOG_AND_MONITOR, RiskLevel.MEDIUM, ["b"]),
        ]
        schedule = generate_schedule(health)
        assert isinstance(schedule.priority_breakdown, dict)
        total = sum(schedule.priority_breakdown.values())
        assert total == schedule.total_tasks

    def test_summary_generated(self):
        health = [self._make_health("Motor", SeverityLevel.HIGH)]
        schedule = generate_schedule(health)
        assert len(schedule.summary) > 0
        assert "action item" in schedule.summary.lower()

    def test_empty_summary_for_no_tasks(self):
        schedule = generate_schedule([self._make_health("X", SeverityLevel.NORMAL, MaintenanceAction.NONE)])
        assert "no action" in schedule.summary.lower() or "normal" in schedule.summary.lower()

    def test_data_driven_priority_hours(self):
        """Passing data → should use scaled priority hours."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=50, freq="D"),
            "value": range(50),
        })
        health = [self._make_health("X", SeverityLevel.HIGH, cols=["value"])]
        risks = [self._make_risk("value")]
        schedule = generate_schedule(health, risks, data=df)
        # Daily data → scaled hours should be 24x default
        # Tasks should exist
        assert schedule.total_tasks >= 1

    def test_generated_at_is_iso(self):
        health = [self._make_health("X", SeverityLevel.LOW)]
        schedule = generate_schedule(health)
        assert "T" in schedule.generated_at  # ISO format
