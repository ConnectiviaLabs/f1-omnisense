"""omnidapt type definitions — enums, dataclasses, protocols.

Continuous Adaptation Engine types for dataset management,
training pipelines, model versioning, and health monitoring.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ────────────────────────────────────────────────────────────────

class ModelDomain(str, Enum):
    CV = "cv"
    AUDIO = "audio"


class ModelStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Dataset Types ────────────────────────────────────────────────────────

@dataclass
class DatasetInfo:
    name: str
    domain: ModelDomain
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    description: str = ""
    labels: List[str] = field(default_factory=list)
    sample_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description": self.description,
            "labels": self.labels,
            "sample_count": self.sample_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DatasetInfo:
        return cls(
            name=d["name"],
            domain=ModelDomain(d["domain"]),
            created_at=d.get("created_at", 0),
            updated_at=d.get("updated_at", 0),
            description=d.get("description", ""),
            labels=d.get("labels", []),
            sample_count=d.get("sample_count", 0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class DatasetStats:
    name: str
    domain: ModelDomain
    total_samples: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    avg_sample_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain.value,
            "total_samples": self.total_samples,
            "label_distribution": self.label_distribution,
            "total_size_bytes": self.total_size_bytes,
            "avg_sample_size_bytes": self.avg_sample_size_bytes,
        }


# ── Training Types ───────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    domain: ModelDomain
    dataset_name: str
    model_name: str = ""
    base_model: str = ""
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.01
    image_size: int = 640
    patience: int = 10
    device: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.model_name:
            self.model_name = self.dataset_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "base_model": self.base_model,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "image_size": self.image_size,
            "patience": self.patience,
            "device": self.device,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrainingConfig:
        return cls(
            domain=ModelDomain(d["domain"]),
            dataset_name=d["dataset_name"],
            model_name=d.get("model_name", ""),
            base_model=d.get("base_model", ""),
            epochs=d.get("epochs", 50),
            batch_size=d.get("batch_size", 16),
            learning_rate=d.get("learning_rate", 0.01),
            image_size=d.get("image_size", 640),
            patience=d.get("patience", 10),
            device=d.get("device", ""),
            extra=d.get("extra", {}),
        )


@dataclass
class TrainingMetrics:
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "mAP50": self.mAP50,
            "mAP50_95": self.mAP50_95,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrainingMetrics:
        return cls(
            epoch=d.get("epoch", 0),
            train_loss=d.get("train_loss", 0.0),
            val_loss=d.get("val_loss", 0.0),
            accuracy=d.get("accuracy", 0.0),
            f1_score=d.get("f1_score", 0.0),
            precision=d.get("precision", 0.0),
            recall=d.get("recall", 0.0),
            mAP50=d.get("mAP50", 0.0),
            mAP50_95=d.get("mAP50_95", 0.0),
            extra=d.get("extra", {}),
        )


# ── Job Types ────────────────────────────────────────────────────────────

@dataclass
class TrainingJob:
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    config: Optional[TrainingConfig] = None
    status: TrainingStatus = TrainingStatus.QUEUED
    progress: float = 0.0
    phase: str = ""
    detail: str = ""
    metrics: Optional[TrainingMetrics] = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    result_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "config": self.config.to_dict() if self.config else None,
            "status": self.status.value,
            "progress": self.progress,
            "phase": self.phase,
            "detail": self.detail,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result_path": self.result_path,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrainingJob:
        return cls(
            job_id=d.get("job_id", uuid.uuid4().hex[:12]),
            config=TrainingConfig.from_dict(d["config"]) if d.get("config") else None,
            status=TrainingStatus(d.get("status", "queued")),
            progress=d.get("progress", 0.0),
            phase=d.get("phase", ""),
            detail=d.get("detail", ""),
            metrics=TrainingMetrics.from_dict(d["metrics"]) if d.get("metrics") else None,
            error=d.get("error", ""),
            created_at=d.get("created_at", 0),
            started_at=d.get("started_at", 0),
            completed_at=d.get("completed_at", 0),
            result_path=d.get("result_path", ""),
        )


# ── Model Version Types ──────────────────────────────────────────────────

@dataclass
class ModelVersion:
    name: str
    domain: ModelDomain
    version: str = "1.0.0"
    status: ModelStatus = ModelStatus.DRAFT
    artifact_path: str = ""
    metrics: Optional[TrainingMetrics] = None
    base_model: str = ""
    dataset_name: str = ""
    created_at: float = field(default_factory=time.time)
    promoted_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain.value,
            "version": self.version,
            "status": self.status.value,
            "artifact_path": self.artifact_path,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "base_model": self.base_model,
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ModelVersion:
        return cls(
            name=d["name"],
            domain=ModelDomain(d["domain"]),
            version=d.get("version", "1.0.0"),
            status=ModelStatus(d.get("status", "draft")),
            artifact_path=d.get("artifact_path", ""),
            metrics=TrainingMetrics.from_dict(d["metrics"]) if d.get("metrics") else None,
            base_model=d.get("base_model", ""),
            dataset_name=d.get("dataset_name", ""),
            created_at=d.get("created_at", 0),
            promoted_at=d.get("promoted_at", 0),
            metadata=d.get("metadata", {}),
        )


# ── Health Report ────────────────────────────────────────────────────────

@dataclass
class HealthReport:
    model_name: str
    domain: ModelDomain
    status: ModelStatus = ModelStatus.DRAFT
    current_version: str = ""
    age_days: float = 0.0
    latest_metrics: Optional[TrainingMetrics] = None
    needs_retrain: bool = False
    retrain_reason: str = ""
    drift_detected: bool = False
    drift_score: float = 0.0
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "domain": self.domain.value,
            "status": self.status.value,
            "current_version": self.current_version,
            "age_days": round(self.age_days, 1),
            "latest_metrics": self.latest_metrics.to_dict() if self.latest_metrics else None,
            "needs_retrain": self.needs_retrain,
            "retrain_reason": self.retrain_reason,
            "drift_detected": self.drift_detected,
            "drift_score": round(self.drift_score, 4),
            "checked_at": self.checked_at,
        }
