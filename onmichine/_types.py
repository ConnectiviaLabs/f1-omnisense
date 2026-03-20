"""Shared type definitions for the agentml tabular AutoML agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ────────────────────────────────────────────────────────────────────


class TaskType(str, Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class ModelStage(str, Enum):
    BASELINE = "stage_0_baseline"
    BOOSTING = "stage_1_boosting"
    HPO = "stage_2_hpo"
    ENSEMBLE = "stage_3_ensemble"


class MetricName(str, Enum):
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    ACCURACY = "accuracy"
    F1_MACRO = "f1_macro"
    F1_WEIGHTED = "f1_weighted"
    AUC_ROC = "auc_roc"
    LOG_LOSS = "log_loss"


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_METRICS: Dict[TaskType, tuple] = {
    TaskType.REGRESSION: (MetricName.RMSE, [MetricName.MAE, MetricName.R2]),
    TaskType.BINARY_CLASSIFICATION: (
        MetricName.AUC_ROC,
        [MetricName.F1_MACRO, MetricName.ACCURACY, MetricName.LOG_LOSS],
    ),
    TaskType.MULTICLASS_CLASSIFICATION: (
        MetricName.F1_WEIGHTED,
        [MetricName.ACCURACY, MetricName.LOG_LOSS],
    ),
}

CLASSIFICATION_CARDINALITY_THRESHOLD = 20
OHE_CARDINALITY_LIMIT = 15


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """User-facing configuration for a full pipeline run."""

    data_path: str
    target_column: str
    output_dir: str = "agentml_output"
    task_type: Optional[TaskType] = None
    test_ratio: float = 0.2
    random_state: int = 42
    time_budget_s: float = 300.0
    max_hpo_trials: int = 50
    enable_ensemble: bool = True
    sample_rows: Optional[int] = None
    model: Optional[str] = None  # Groq model override

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("task_type"):
            d["task_type"] = d["task_type"].value
        return d


@dataclass
class ColumnSchema:
    """Single column in the schema contract."""

    name: str
    dtype: str
    role: str
    is_target: bool = False
    is_feature: bool = False
    cardinality: Optional[int] = None
    null_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class TrialResult:
    """A single model training trial."""

    trial_id: str
    model_name: str
    stage: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    train_time_s: float
    primary_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineState:
    """Evolving state passed between tool calls, holding all artifacts."""

    config: RunConfig
    # Data
    dataset: Any = None  # TabularDataset
    schema: List[ColumnSchema] = field(default_factory=list)
    data_profile: Optional[Dict[str, Any]] = None
    cleaning_plan: Optional[Dict[str, Any]] = None
    issues: List[str] = field(default_factory=list)
    # Task
    task_type: Optional[TaskType] = None
    num_classes: Optional[int] = None
    class_labels: Optional[List[str]] = None
    class_distribution: Optional[Dict[str, int]] = None
    primary_metric: Optional[MetricName] = None
    secondary_metrics: List[MetricName] = field(default_factory=list)
    # Features
    preprocessor: Any = None  # fitted ColumnTransformer
    label_encoder: Any = None  # fitted LabelEncoder (classification)
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None
    feature_names_out: List[str] = field(default_factory=list)
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    # Model search
    trials: List[TrialResult] = field(default_factory=list)
    best_trial: Optional[TrialResult] = None
    best_model: Any = None
    leaderboard: List[Dict[str, Any]] = field(default_factory=list)
    # Eval
    holdout_metrics: Optional[Dict[str, float]] = None
    confusion_matrix: Any = None
    classification_report: Optional[Dict[str, Any]] = None
    residual_stats: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_card_md: str = ""
    # Package
    output_files: List[str] = field(default_factory=list)
    reproducibility_lock: Optional[Dict[str, Any]] = None
    # Agent trace
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
