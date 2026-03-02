"""omnidapt — Continuous Adaptation Engine.

Fine-tuning pipelines for CV (YOLO) and Audio (AST) models,
with dataset management, model versioning, and scheduled retraining.

Part of omnisuite. Collaborates with omnivis for inference integration.
"""

from omnidapt._types import (
    DatasetInfo,
    DatasetStats,
    HealthReport,
    ModelDomain,
    ModelStatus,
    ModelVersion,
    TrainingConfig,
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)
from omnidapt._config import get_config, OmnidaptConfig
from omnidapt._storage import get_storage
from omnidapt.dataset_manager import get_dataset_manager
from omnidapt.model_registry import get_model_registry
from omnidapt.jobs import get_job_manager
from omnidapt.scheduler import get_scheduler
from omnidapt.yolo_trainer import train_yolo
from omnidapt.ast_trainer import train_ast

__all__ = [
    # Types
    "DatasetInfo",
    "DatasetStats",
    "HealthReport",
    "ModelDomain",
    "ModelStatus",
    "ModelVersion",
    "TrainingConfig",
    "TrainingJob",
    "TrainingMetrics",
    "TrainingStatus",
    # Config
    "OmnidaptConfig",
    "get_config",
    # Singletons
    "get_storage",
    "get_dataset_manager",
    "get_model_registry",
    "get_job_manager",
    "get_scheduler",
    # Training
    "train_yolo",
    "train_ast",
]
