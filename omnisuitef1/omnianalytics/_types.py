"""Shared type definitions for the omnianalytics service."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SeverityLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnomalyScore:
    row_index: int
    score_mean: float
    score_std: float
    severity: SeverityLevel = SeverityLevel.NORMAL
    vote_count: int = 0
    total_models: int = 4
    model_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AnomalyScore":
        severity = d.get("severity", "normal")
        if isinstance(severity, str):
            severity = SeverityLevel(severity)
        return cls(
            row_index=d.get("row_index", 0),
            score_mean=d.get("score_mean", 0.0),
            score_std=d.get("score_std", 0.0),
            severity=severity,
            vote_count=d.get("vote_count", 0),
            total_models=d.get("total_models", 4),
            model_scores=d.get("model_scores", {}),
        )


@dataclass
class AnomalyResult:
    scores: List[AnomalyScore]
    contamination_estimate: float
    threshold: float
    anomaly_count: int
    total_rows: int
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_count": self.anomaly_count,
            "total_rows": self.total_rows,
            "contamination_estimate": round(self.contamination_estimate, 4),
            "threshold": round(self.threshold, 4),
            "severity_distribution": self.severity_distribution,
            "model_weights": self.model_weights,
            "scores": [s.to_dict() for s in self.scores],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnomalyResult":
        scores = [
            AnomalyScore.from_dict(s) if isinstance(s, dict) else s
            for s in d.get("scores", [])
        ]
        return cls(
            scores=scores,
            contamination_estimate=d.get("contamination_estimate", 0),
            threshold=d.get("threshold", 0),
            anomaly_count=d.get("anomaly_count", 0),
            total_rows=d.get("total_rows", 0),
            severity_distribution=d.get("severity_distribution", {}),
            model_weights=d.get("model_weights", {}),
        )


@dataclass
class ForecastResult:
    column: str
    method: str
    horizon: int
    timestamps: List[str]
    values: List[float]
    lower_bound: List[float] = field(default_factory=list)
    upper_bound: List[float] = field(default_factory=list)
    mae: Optional[float] = None
    rmse: Optional[float] = None
    # Heuristic fields
    trend_direction: Optional[str] = None   # "rising" | "falling" | "stable"
    trend_pct: Optional[float] = None       # % change first→last forecast value
    volatility: Optional[float] = None      # normalised confidence band width
    risk_flag: bool = False                 # True when forecast suggests degradation
    # Historical context
    history: List[float] = field(default_factory=list)
    history_timestamps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "ForecastResult":
        return cls(
            column=d.get("column", ""),
            method=d.get("method", ""),
            horizon=d.get("horizon", 0),
            timestamps=d.get("timestamps", []),
            values=d.get("values", []),
            lower_bound=d.get("lower_bound", []),
            upper_bound=d.get("upper_bound", []),
            mae=d.get("mae"),
            rmse=d.get("rmse"),
            trend_direction=d.get("trend_direction"),
            trend_pct=d.get("trend_pct"),
            volatility=d.get("volatility"),
            risk_flag=d.get("risk_flag", False),
            history=d.get("history", []),
            history_timestamps=d.get("history_timestamps", []),
        )


@dataclass
class JobState:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    phase: str = ""
    phase_detail: str = ""
    progress_pct: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "phase": self.phase,
            "phase_detail": self.phase_detail,
            "progress_pct": round(self.progress_pct, 1),
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
