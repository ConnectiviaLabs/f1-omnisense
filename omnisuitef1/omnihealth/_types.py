"""Shared type definitions for the omnihealth service."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from omnianalytics._types import SeverityLevel


# ── Enums ────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MaintenancePriority(str, Enum):
    CRITICAL = "critical"       # 0-2 hours
    HIGH = "high"               # 2-24 hours
    MEDIUM = "medium"           # within 3 days
    LOW = "low"                 # within 7 days
    ROUTINE = "routine"         # within 30 days


class MaintenanceAction(str, Enum):
    ALERT_AND_REMEDIATE = "alert_and_remediate"
    ALERT = "alert"
    LOG_AND_MONITOR = "log_and_monitor"
    LOG = "log"
    NONE = "none"


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


# ── Constants ────────────────────────────────────────────────────────────────

# Optional display labels and units for known feature names.
# NOT used by the pipeline — purely for user-facing formatting.
# Users can pass their own via assess(feature_units={...}).
SENSOR_UNITS: Dict[str, tuple] = {
    "vibration": ("Vibration", "Hz"),
    "temperature": ("Temperature", "\u00b0C"),
    "humidity": ("Humidity", "%"),
    "pressure": ("Pressure", "Pa"),
    "voltage": ("Voltage", "V"),
    "current": ("Current", "A"),
    "power": ("Power", "W"),
    "rpm": ("RPM", "rev/min"),
    "speed": ("Speed", "km/h"),
    "frequency": ("Frequency", "Hz"),
    "light": ("Light Intensity", "lux"),
    "sound": ("Sound Level", "dB"),
    "batterylevel": ("Battery Level", "%"),
}

SEVERITY_TO_ACTION: Dict[SeverityLevel, MaintenanceAction] = {
    SeverityLevel.CRITICAL: MaintenanceAction.ALERT_AND_REMEDIATE,
    SeverityLevel.HIGH: MaintenanceAction.ALERT,
    SeverityLevel.MEDIUM: MaintenanceAction.LOG_AND_MONITOR,
    SeverityLevel.LOW: MaintenanceAction.LOG,
    SeverityLevel.NORMAL: MaintenanceAction.NONE,
}

PRIORITY_HOURS: Dict[MaintenancePriority, int] = {
    MaintenancePriority.CRITICAL: 2,
    MaintenancePriority.HIGH: 24,
    MaintenancePriority.MEDIUM: 72,
    MaintenancePriority.LOW: 168,
    MaintenancePriority.ROUTINE: 720,
}


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class HealthScore:
    component: str
    health_pct: float
    risk_level: RiskLevel
    severity: SeverityLevel
    action: MaintenanceAction
    anomaly_score: float
    vote_count: int
    total_models: int
    top_features: List[Dict[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        d["severity"] = self.severity.value
        d["action"] = self.action.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HealthScore":
        d = dict(d)
        d["risk_level"] = RiskLevel(d["risk_level"])
        d["severity"] = SeverityLevel(d["severity"])
        d["action"] = MaintenanceAction(d["action"])
        return cls(**d)


@dataclass
class TimeSeriesAnalysis:
    trend: TrendDirection
    trend_strength: float
    slope: float
    drift_rate: float
    is_stationary: bool
    seasonality_detected: bool
    seasonal_period: Optional[int]
    seasonal_strength: float
    anomaly_count: int
    anomaly_pct: float
    forecastability_score: float
    forecastability_rating: str
    operational_zones: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trend"] = self.trend.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TimeSeriesAnalysis":
        d = dict(d)
        d["trend"] = TrendDirection(d["trend"])
        return cls(**d)


@dataclass
class RiskAssessment:
    feature: str
    current_value: float
    forecast_value: float
    risk_level: RiskLevel
    trend: TrendDirection
    trend_pct: float
    confidence_lower: float
    confidence_upper: float
    degradation_rate: Optional[float] = None
    time_series_analysis: Optional[TimeSeriesAnalysis] = None
    horizon_results: Optional[List[Dict[str, Any]]] = None  # per-horizon forecasts

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "feature": self.feature,
            "current_value": self.current_value,
            "forecast_value": self.forecast_value,
            "risk_level": self.risk_level.value,
            "trend": self.trend.value,
            "trend_pct": round(self.trend_pct, 2),
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "degradation_rate": self.degradation_rate,
        }
        if self.time_series_analysis is not None:
            d["time_series_analysis"] = self.time_series_analysis.to_dict()
        if self.horizon_results is not None:
            d["horizon_results"] = self.horizon_results
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskAssessment":
        d = dict(d)
        d["risk_level"] = RiskLevel(d["risk_level"])
        d["trend"] = TrendDirection(d["trend"])
        tsa = d.pop("time_series_analysis", None)
        if tsa is not None:
            tsa = TimeSeriesAnalysis.from_dict(tsa)
        hr = d.pop("horizon_results", None)
        return cls(**d, time_series_analysis=tsa, horizon_results=hr)


@dataclass
class MaintenanceTask:
    task_id: str
    component: str
    feature: str
    priority: MaintenancePriority
    action: MaintenanceAction
    description: str
    reason: str
    estimated_hours: int
    risk_assessment: Optional[RiskAssessment] = None
    health_score: Optional[HealthScore] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "task_id": self.task_id,
            "component": self.component,
            "feature": self.feature,
            "priority": self.priority.value,
            "action": self.action.value,
            "description": self.description,
            "reason": self.reason,
            "estimated_hours": self.estimated_hours,
            "created_at": self.created_at,
        }
        if self.risk_assessment is not None:
            d["risk_assessment"] = self.risk_assessment.to_dict()
        if self.health_score is not None:
            d["health_score"] = self.health_score.to_dict()
        return d


@dataclass
class MaintenanceSchedule:
    generated_at: str
    total_tasks: int
    priority_breakdown: Dict[str, int]
    tasks: List[MaintenanceTask]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "total_tasks": self.total_tasks,
            "priority_breakdown": self.priority_breakdown,
            "tasks": [t.to_dict() for t in self.tasks],
            "summary": self.summary,
        }


@dataclass
class HealthReport:
    components: List[HealthScore]
    risk_assessments: List[RiskAssessment]
    schedule: MaintenanceSchedule
    overall_health: float
    overall_risk: RiskLevel
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "risk_assessments": [r.to_dict() for r in self.risk_assessments],
            "schedule": self.schedule.to_dict(),
            "overall_health": round(self.overall_health, 1),
            "overall_risk": self.overall_risk.value,
            "generated_at": self.generated_at,
        }
