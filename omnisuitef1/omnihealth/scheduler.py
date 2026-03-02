"""Maintenance task generation + priority scheduling.

Ports DataSense autonomous_maintenance_scheduler.py (pure computation).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

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
    SEVERITY_TO_ACTION,
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Severity ordering for comparisons ────────────────────────────────────────

_SEVERITY_ORDER = {
    SeverityLevel.NORMAL: 0,
    SeverityLevel.LOW: 1,
    SeverityLevel.MEDIUM: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.CRITICAL: 4,
}

_PRIORITY_ORDER = {
    MaintenancePriority.ROUTINE: 5,
    MaintenancePriority.LOW: 4,
    MaintenancePriority.MEDIUM: 3,
    MaintenancePriority.HIGH: 2,
    MaintenancePriority.CRITICAL: 1,
}


# ── Priority determination ───────────────────────────────────────────────────

def determine_priority(
    risk_level: RiskLevel,
    severity: SeverityLevel,
    trend: Optional[TrendDirection] = None,
) -> MaintenancePriority:
    """Map risk + severity + trend → maintenance priority.

    Logic (from DataSense scheduler):
    - CRITICAL: risk=HIGH and severity >= HIGH
    - HIGH: risk=HIGH, or severity=HIGH and trend=INCREASING
    - MEDIUM: risk=MEDIUM, or severity=MEDIUM
    - LOW: severity=LOW
    - ROUTINE: everything else
    """
    sev_ord = _SEVERITY_ORDER.get(severity, 0)

    if risk_level == RiskLevel.HIGH and sev_ord >= _SEVERITY_ORDER[SeverityLevel.HIGH]:
        return MaintenancePriority.CRITICAL
    elif risk_level == RiskLevel.HIGH:
        return MaintenancePriority.HIGH
    elif severity == SeverityLevel.HIGH and trend == TrendDirection.INCREASING:
        return MaintenancePriority.HIGH
    elif severity == SeverityLevel.CRITICAL:
        return MaintenancePriority.CRITICAL
    elif risk_level == RiskLevel.MEDIUM or severity == SeverityLevel.MEDIUM:
        return MaintenancePriority.MEDIUM
    elif severity == SeverityLevel.LOW:
        return MaintenancePriority.LOW
    return MaintenancePriority.ROUTINE


# ── Data-driven time scaling ─────────────────────────────────────────────────

def infer_sampling_hours(data: Optional[pd.DataFrame] = None) -> float:
    """Infer the median sampling interval in hours from timestamp columns.

    Scans for datetime columns, computes median diff. Returns 1.0 (hourly)
    as default if no timestamps found or data is None.
    """
    if data is None or len(data) < 2:
        return 1.0

    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            diffs = data[col].dropna().diff().dropna()
            if len(diffs) > 0:
                median_td = diffs.median()
                hours = median_td.total_seconds() / 3600.0
                return max(hours, 1 / 60)  # floor at 1 minute
    return 1.0


def _scale_priority_hours(sampling_hours: float) -> Dict[MaintenancePriority, int]:
    """Scale PRIORITY_HOURS proportionally to the data's sampling frequency.

    The default PRIORITY_HOURS assume hourly data (sampling_hours=1).
    For slower data (e.g. daily=24h), all windows stretch by 24x.
    For faster data (e.g. per-minute=1/60), windows shrink.

    Rationale: CRITICAL means "within ~2 sampling intervals", not "within 2 wall-clock hours".
    """
    # Base ratios: CRITICAL=2, HIGH=24, MEDIUM=72, LOW=168, ROUTINE=720
    # These are in units of "sampling intervals" when sampling_hours=1
    scale = sampling_hours  # hours per sample
    return {
        MaintenancePriority.CRITICAL: max(1, int(2 * scale)),
        MaintenancePriority.HIGH: max(2, int(24 * scale)),
        MaintenancePriority.MEDIUM: max(4, int(72 * scale)),
        MaintenancePriority.LOW: max(8, int(168 * scale)),
        MaintenancePriority.ROUTINE: max(24, int(720 * scale)),
    }


# ── Completion time estimation ───────────────────────────────────────────────

def estimate_completion_hours(
    priority: MaintenancePriority,
    trend_pct: float = 0.0,
    priority_hours: Optional[Dict[MaintenancePriority, int]] = None,
) -> int:
    """Estimate completion hours based on priority + trend severity.

    Uses data-driven priority_hours if provided, else falls back to
    static PRIORITY_HOURS defaults.

    Trend adjustment:
    - >10% trend → 0.4x (aggressive shrink)
    - >5% trend → 0.6x
    - >2% trend → 0.75x
    """
    hours_map = priority_hours or PRIORITY_HOURS
    base = hours_map.get(priority, 72)
    abs_trend = abs(trend_pct)

    if abs_trend > 10:
        factor = 0.4
    elif abs_trend > 5:
        factor = 0.6
    elif abs_trend > 2:
        factor = 0.75
    else:
        factor = 1.0

    return max(1, int(base * factor))


# ── Task generation ──────────────────────────────────────────────────────────

def generate_task(
    component: str,
    feature: str,
    health: HealthScore,
    risk: Optional[RiskAssessment] = None,
    priority_hours: Optional[Dict[MaintenancePriority, int]] = None,
) -> MaintenanceTask:
    """Create a single maintenance task from health + optional risk assessment."""
    trend = risk.trend if risk else TrendDirection.STABLE
    trend_pct = risk.trend_pct if risk else 0.0
    risk_level = risk.risk_level if risk else health.risk_level

    priority = determine_priority(risk_level, health.severity, trend)
    action = health.action  # use the consensus-gated action from health scoring
    hours = estimate_completion_hours(priority, trend_pct, priority_hours)

    # Build description
    description = _build_description(component, feature, health, risk)
    reason = _build_reason(health, risk)

    return MaintenanceTask(
        task_id=uuid.uuid4().hex[:12],
        component=component,
        feature=feature,
        priority=priority,
        action=action,
        description=description,
        reason=reason,
        estimated_hours=hours,
        risk_assessment=risk,
        health_score=health,
    )


def _build_description(
    component: str, feature: str,
    health: HealthScore, risk: Optional[RiskAssessment],
) -> str:
    """Build human-readable task description from data — no domain assumptions."""
    parts = [f"{component}"]
    if health.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
        parts.append(f"requires immediate action")
    elif health.severity == SeverityLevel.MEDIUM:
        parts.append(f"needs scheduled review")
    else:
        parts.append(f"routine check recommended")

    parts.append(f"(health: {health.health_pct:.0f}%)")

    if risk and risk.trend != TrendDirection.STABLE:
        parts.append(f"— {feature} trending {risk.trend.value} ({risk.trend_pct:+.1f}%)")

    return " ".join(parts)


def _build_reason(health: HealthScore, risk: Optional[RiskAssessment]) -> str:
    """Build reason for the maintenance task."""
    reasons = []
    reasons.append(f"Severity: {health.severity.value}")
    reasons.append(f"Health: {health.health_pct:.0f}%")

    if risk:
        reasons.append(f"Risk: {risk.risk_level.value}")
        if risk.degradation_rate is not None:
            reasons.append(f"Degradation rate: {risk.degradation_rate:.4f}")
    return "; ".join(reasons)


# ── Schedule generation ──────────────────────────────────────────────────────

def generate_schedule(
    health_scores: List[HealthScore],
    risk_assessments: Optional[List[RiskAssessment]] = None,
    data: Optional[pd.DataFrame] = None,
) -> MaintenanceSchedule:
    """Generate prioritized maintenance schedule from health + risk data.

    Data-driven improvements:
    - Priority hours scale to the data's sampling frequency (hourly vs daily vs weekly)
    - Multiple tasks per component when distinct failure modes exist
      (features degrading in different directions = separate work orders)
    """
    # Infer time scale from data
    sampling_h = infer_sampling_hours(data)
    scaled_hours = _scale_priority_hours(sampling_h)

    risks_by_component: Dict[str, List[RiskAssessment]] = {}
    if risk_assessments:
        for ra in risk_assessments:
            risks_by_component.setdefault(ra.feature, []).append(ra)

    tasks: List[MaintenanceTask] = []

    for health in health_scores:
        if health.severity == SeverityLevel.NORMAL:
            continue

        # Find matching risk assessments for this component's columns only
        component_risks: List[RiskAssessment] = []
        if health.metadata and "columns" in health.metadata:
            for col in health.metadata["columns"]:
                component_risks.extend(risks_by_component.get(col, []))

        if component_risks:
            # Group risks by failure mode (trend direction)
            # Different directions = different root causes = separate tasks
            mode_groups: Dict[str, List[RiskAssessment]] = {}
            for r in component_risks:
                mode_groups.setdefault(r.trend.value, []).append(r)

            for _direction, risks_in_mode in mode_groups.items():
                # Pick worst risk within each failure mode
                worst_risk = max(
                    risks_in_mode,
                    key=lambda r: (
                        {"high": 3, "medium": 2, "low": 1}.get(r.risk_level.value, 0),
                        abs(r.trend_pct),
                    ),
                )
                task = generate_task(
                    health.component, worst_risk.feature, health, worst_risk,
                    priority_hours=scaled_hours,
                )
                tasks.append(task)
        else:
            # No risk data — create task from health alone
            task = generate_task(health.component, "overall", health,
                                 priority_hours=scaled_hours)
            tasks.append(task)

    # Sort by priority (CRITICAL first)
    tasks.sort(key=lambda t: _PRIORITY_ORDER.get(t.priority, 99))

    # Priority breakdown
    breakdown: Dict[str, int] = {}
    for p in MaintenancePriority:
        count = sum(1 for t in tasks if t.priority == p)
        if count > 0:
            breakdown[p.value] = count

    summary = _build_summary(tasks, health_scores, scaled_hours)

    return MaintenanceSchedule(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_tasks=len(tasks),
        priority_breakdown=breakdown,
        tasks=tasks,
        summary=summary,
    )


def _build_summary(
    tasks: List[MaintenanceTask],
    health_scores: List[HealthScore],
    scaled_hours: Optional[Dict[MaintenancePriority, int]] = None,
) -> str:
    """Build human-readable schedule summary."""
    if not tasks:
        return "All components within normal parameters. No action items generated."

    hours_map = scaled_hours or PRIORITY_HOURS
    n_critical = sum(1 for t in tasks if t.priority == MaintenancePriority.CRITICAL)
    n_high = sum(1 for t in tasks if t.priority == MaintenancePriority.HIGH)
    avg_health = sum(h.health_pct for h in health_scores) / len(health_scores) if health_scores else 0

    parts = [f"{len(tasks)} action item(s) generated."]

    if n_critical > 0:
        crit_h = hours_map.get(MaintenancePriority.CRITICAL, 2)
        parts.append(f"{n_critical} CRITICAL (immediate action within {crit_h}h).")
    if n_high > 0:
        high_h = hours_map.get(MaintenancePriority.HIGH, 24)
        parts.append(f"{n_high} HIGH priority (within {high_h}h).")

    parts.append(f"Average health: {avg_health:.0f}%.")

    # Total estimated hours
    total_hours = sum(t.estimated_hours for t in tasks)
    parts.append(f"Total estimated effort: {total_hours}h.")

    return " ".join(parts)
