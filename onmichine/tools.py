"""ML tool implementations for the agentic AutoML pipeline.

Each tool is a function that reads/writes to PipelineState and returns
a JSON-serialisable dict the Groq agent can reason about.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add omnisuitef1 to path so `from omnidata import ...` resolves correctly
_omnisuite_dir = str(Path(__file__).resolve().parent.parent / "omnisuitef1")
if _omnisuite_dir not in sys.path:
    sys.path.insert(0, _omnisuite_dir)

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from omnidata import load, preprocess
from omnidata._types import ColumnRole, DType

from onmichine._types import (
    CLASSIFICATION_CARDINALITY_THRESHOLD,
    DEFAULT_METRICS,
    OHE_CARDINALITY_LIMIT,
    ColumnSchema,
    MetricName,
    ModelStage,
    PipelineState,
    TaskType,
    TrialResult,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1: load_and_profile
# ═══════════════════════════════════════════════════════════════════════════════


def load_and_profile(state: PipelineState) -> Dict[str, Any]:
    """Load the dataset, profile columns, detect issues, build schema."""
    cfg = state.config
    dataset = load(cfg.data_path, sample=cfg.sample_rows)
    dataset = preprocess(dataset)
    logger.info(
        "Loaded %d rows x %d cols from %s",
        len(dataset.df), len(dataset.df.columns), cfg.data_path,
    )

    prof = dataset.profile
    issues: List[str] = []
    cleaning_plan: Dict[str, str] = {}
    schema: List[ColumnSchema] = []

    df = dataset.df
    target = cfg.target_column

    if target not in df.columns:
        issues.append(f"Target column '{target}' not found in dataset")

    for cp in prof.columns:
        is_target = cp.name == target
        is_feature = (
            cp.role in (ColumnRole.METRIC, ColumnRole.CATEGORICAL)
            and not is_target
        )
        schema.append(ColumnSchema(
            name=cp.name,
            dtype=cp.dtype.value if isinstance(cp.dtype, DType) else str(cp.dtype),
            role=cp.role.value if isinstance(cp.role, ColumnRole) else str(cp.role),
            is_target=is_target,
            is_feature=is_feature,
            cardinality=cp.unique_count,
            null_pct=cp.null_pct,
        ))
        if cp.null_pct > 0:
            if cp.role == ColumnRole.METRIC:
                cleaning_plan[cp.name] = "fill_median"
            elif cp.role == ColumnRole.CATEGORICAL:
                cleaning_plan[cp.name] = "fill_mode"
            elif cp.null_pct > 50.0:
                cleaning_plan[cp.name] = "drop_column"

    # Detect constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues.append(f"Column '{col}' is constant")

    # Detect high-cardinality categoricals
    for cs in schema:
        if cs.role == "categorical" and cs.cardinality and cs.cardinality > 100:
            issues.append(f"Column '{cs.name}' high cardinality ({cs.cardinality})")

    # Target null check
    if target in df.columns:
        null_pct = df[target].isna().mean() * 100
        if null_pct > 0:
            issues.append(f"Target has {null_pct:.1f}% missing values")

    state.dataset = dataset
    state.schema = schema
    state.data_profile = prof.to_dict()
    state.cleaning_plan = cleaning_plan
    state.issues = issues

    return {
        "status": "ok",
        "rows": prof.row_count,
        "columns": prof.column_count,
        "features": sum(1 for s in schema if s.is_feature),
        "target_found": target in df.columns,
        "issues": issues,
        "cleaning_plan": cleaning_plan,
        "column_summary": [
            {"name": s.name, "dtype": s.dtype, "role": s.role,
             "is_feature": s.is_feature, "cardinality": s.cardinality,
             "null_pct": round(s.null_pct, 2)}
            for s in schema
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2: infer_task
# ═══════════════════════════════════════════════════════════════════════════════


def infer_task(state: PipelineState) -> Dict[str, Any]:
    """Infer ML task type from target column properties."""
    cfg = state.config
    df = state.dataset.df
    target = cfg.target_column

    if target not in df.columns:
        return {"status": "error", "message": f"Target '{target}' not found"}

    series = df[target].dropna()

    # User override
    if cfg.task_type is not None:
        task_type = cfg.task_type
    else:
        # Find column profile
        cp = None
        for c in state.dataset.profile.columns:
            if c.name == target:
                cp = c
                break

        # Deterministic rules
        if cp and cp.dtype == DType.BOOL:
            task_type = TaskType.BINARY_CLASSIFICATION
        elif cp and cp.dtype == DType.STRING:
            task_type = (
                TaskType.BINARY_CLASSIFICATION
                if series.nunique() == 2
                else TaskType.MULTICLASS_CLASSIFICATION
            )
        elif series.nunique() == 2:
            task_type = TaskType.BINARY_CLASSIFICATION
        elif series.nunique() <= CLASSIFICATION_CARDINALITY_THRESHOLD:
            task_type = TaskType.MULTICLASS_CLASSIFICATION
        else:
            task_type = TaskType.REGRESSION

    num_classes = None
    class_labels = None
    class_dist = None

    if task_type != TaskType.REGRESSION:
        vc = series.value_counts()
        num_classes = len(vc)
        class_labels = [str(v) for v in vc.index.tolist()]
        class_dist = {str(k): int(v) for k, v in vc.items()}

    primary, secondaries = DEFAULT_METRICS[task_type]

    state.task_type = task_type
    state.num_classes = num_classes
    state.class_labels = class_labels
    state.class_distribution = class_dist
    state.primary_metric = primary
    state.secondary_metrics = secondaries

    return {
        "status": "ok",
        "task_type": task_type.value,
        "num_classes": num_classes,
        "class_labels": class_labels,
        "class_distribution": class_dist,
        "primary_metric": primary.value,
        "secondary_metrics": [m.value for m in secondaries],
        "rationale": (
            f"Target dtype={series.dtype}, {series.nunique()} unique values, "
            f"{len(series)} non-null → {task_type.value}"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 3: build_features
# ═══════════════════════════════════════════════════════════════════════════════


def build_features(state: PipelineState) -> Dict[str, Any]:
    """Build sklearn ColumnTransformer, encode target, train/test split."""
    cfg = state.config
    df = state.dataset.df.copy()
    target = cfg.target_column
    task_type = state.task_type

    if task_type is None:
        return {"status": "error", "message": "Run infer_task first"}

    numeric_features: List[str] = []
    categorical_features: List[str] = []
    dropped: List[str] = []

    for cs in state.schema:
        if cs.is_target:
            continue
        if not cs.is_feature:
            dropped.append(cs.name)
            continue
        if cs.role == "categorical":
            categorical_features.append(cs.name)
        elif cs.role == "metric":
            numeric_features.append(cs.name)
        else:
            dropped.append(cs.name)

    # Build ColumnTransformer
    transformers = []

    if numeric_features:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_features))

    low_card = [c for c in categorical_features if df[c].nunique() <= OHE_CARDINALITY_LIMIT]
    high_card = [c for c in categorical_features if df[c].nunique() > OHE_CARDINALITY_LIMIT]

    if low_card:
        cat_ohe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat_ohe", cat_ohe, low_card))

    if high_card:
        cat_ord = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1,
            )),
        ])
        transformers.append(("cat_ord", cat_ord, high_card))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Prepare target
    y = df[target].copy()
    le = None
    if task_type != TaskType.REGRESSION:
        if y.dtype == object or y.dtype.name == "category":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
        else:
            y = y.astype(int)

    # Drop missing target rows
    valid = y.notna()
    df = df.loc[valid]
    y = y.loc[valid]

    # Split
    stratify = y if task_type != TaskType.REGRESSION else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            df, y, test_size=cfg.test_ratio,
            random_state=cfg.random_state, stratify=stratify,
        )
    except ValueError:
        # Stratify can fail with very rare classes
        X_tr, X_te, y_tr, y_te = train_test_split(
            df, y, test_size=cfg.test_ratio, random_state=cfg.random_state,
        )

    X_train = preprocessor.fit_transform(X_tr)
    X_test = preprocessor.transform(X_te)

    try:
        feat_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feat_names = [f"f_{i}" for i in range(X_train.shape[1])]

    state.preprocessor = preprocessor
    state.label_encoder = le
    state.X_train = X_train
    state.X_test = X_test
    state.y_train = np.asarray(y_tr)
    state.y_test = np.asarray(y_te)
    state.feature_names_out = feat_names
    state.numeric_features = numeric_features
    state.categorical_features = categorical_features

    return {
        "status": "ok",
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X_train.shape[1],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_columns": dropped,
        "feature_names_sample": feat_names[:20],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 4: train_models
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    task_type: TaskType,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if task_type == TaskType.REGRESSION:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
    else:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
        if y_proba is not None:
            try:
                if task_type == TaskType.BINARY_CLASSIFICATION:
                    metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics["auc_roc"] = float(roc_auc_score(
                        y_true, y_proba, multi_class="ovr",
                    ))
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
            except Exception:
                pass
    return metrics


def _higher_is_better(metric: MetricName) -> bool:
    return metric not in (MetricName.RMSE, MetricName.MAE, MetricName.LOG_LOSS)


def _primary_score(metrics: Dict[str, float], primary: MetricName) -> float:
    val = metrics.get(primary.value, 0.0)
    return val if _higher_is_better(primary) else -val


def _fit_and_score(
    name: str,
    model: Any,
    stage: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: TaskType,
    primary: MetricName,
) -> TrialResult:
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass

    metrics = _compute_metrics(y_test, y_pred, y_proba, task_type)
    return TrialResult(
        trial_id=uuid.uuid4().hex[:8],
        model_name=name,
        stage=stage,
        params=_safe_params(model),
        metrics=metrics,
        train_time_s=round(elapsed, 3),
        primary_score=_primary_score(metrics, primary),
    )


def _safe_params(model: Any) -> Dict[str, Any]:
    try:
        params = model.get_params()
        return {k: _jsonable(v) for k, v in params.items()}
    except Exception:
        return {}


def _jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    return str(v)


def train_models(
    state: PipelineState,
    *,
    stages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Multi-stage model search: baseline → boosting → HPO → ensemble."""
    if state.X_train is None:
        return {"status": "error", "message": "Run build_features first"}

    cfg = state.config
    task_type = state.task_type
    primary = state.primary_metric
    X_tr, y_tr = state.X_train, state.y_train
    X_te, y_te = state.X_test, state.y_test
    rs = cfg.random_state
    deadline = time.time() + cfg.time_budget_s

    allowed = set(stages or ["baseline", "boosting", "hpo", "ensemble"])
    trials: List[TrialResult] = list(state.trials)  # keep existing
    fitted_models: Dict[str, Any] = {}

    # ── Stage 0: Baselines ───────────────────────────────────────────────
    if "baseline" in allowed:
        logger.info("Stage 0: baselines")
        if task_type == TaskType.REGRESSION:
            bl = Ridge(random_state=rs)
        else:
            bl = LogisticRegression(max_iter=1000, random_state=rs)
        tr = _fit_and_score(
            type(bl).__name__, bl, ModelStage.BASELINE.value,
            X_tr, y_tr, X_te, y_te, task_type, primary,
        )
        trials.append(tr)
        fitted_models[tr.model_name] = bl
        logger.info("  %s → %s", tr.model_name, tr.metrics)

    # ── Stage 1: Boosting ────────────────────────────────────────────────
    if "boosting" in allowed and time.time() < deadline:
        logger.info("Stage 1: boosting models")
        boosters = _get_boosters(task_type, rs)
        for name, model in boosters:
            if time.time() >= deadline:
                break
            try:
                tr = _fit_and_score(
                    name, model, ModelStage.BOOSTING.value,
                    X_tr, y_tr, X_te, y_te, task_type, primary,
                )
                trials.append(tr)
                fitted_models[name] = model
                logger.info("  %s → %s", name, tr.metrics)
            except Exception as e:
                logger.warning("  %s failed: %s", name, e)

    # ── Stage 2: HPO ─────────────────────────────────────────────────────
    if "hpo" in allowed and time.time() < deadline:
        hpo_trials = _stage_hpo(
            trials, task_type, primary, X_tr, y_tr, X_te, y_te,
            rs, cfg.max_hpo_trials, deadline, fitted_models,
        )
        trials.extend(hpo_trials)

    # ── Stage 3: Ensemble ────────────────────────────────────────────────
    if (
        "ensemble" in allowed
        and cfg.enable_ensemble
        and time.time() < deadline
        and len(fitted_models) >= 2
        and X_tr.shape[0] >= 500
    ):
        ens_trial = _stage_ensemble(
            fitted_models, task_type, primary,
            X_tr, y_tr, X_te, y_te, rs,
        )
        if ens_trial:
            trials.append(ens_trial)
            fitted_models[ens_trial.model_name] = fitted_models.get(
                "__ensemble__", None
            )

    # ── Select best ──────────────────────────────────────────────────────
    if not trials:
        return {"status": "error", "message": "No trials completed"}

    reverse = _higher_is_better(primary)
    sorted_trials = sorted(trials, key=lambda t: t.primary_score, reverse=True)
    best = sorted_trials[0]

    # Re-fit best on full training data
    best_model = fitted_models.get(best.model_name)
    if best_model is None:
        # Fallback: pick first available fitted model
        best_model = next(iter(fitted_models.values()), None)

    state.trials = trials
    state.best_trial = best
    state.best_model = best_model
    state.leaderboard = [
        {"rank": i + 1, **t.to_dict()}
        for i, t in enumerate(sorted_trials)
    ]

    return {
        "status": "ok",
        "total_trials": len(trials),
        "best_model": best.model_name,
        "best_stage": best.stage,
        "best_metrics": best.metrics,
        "leaderboard_top5": [
            {"rank": i + 1, "model": t.model_name, "stage": t.stage,
             "primary_score": round(t.primary_score, 4),
             "metrics": {k: round(v, 4) for k, v in t.metrics.items()}}
            for i, t in enumerate(sorted_trials[:5])
        ],
    }


def _get_boosters(task_type: TaskType, rs: int) -> List[tuple]:
    """Return available boosting models with graceful import fallback."""
    boosters = []
    is_reg = task_type == TaskType.REGRESSION

    # LightGBM
    try:
        import lightgbm as lgb
        cls = lgb.LGBMRegressor if is_reg else lgb.LGBMClassifier
        boosters.append(("LightGBM", cls(n_estimators=200, random_state=rs, verbosity=-1)))
    except ImportError:
        logger.info("lightgbm not installed, skipping")

    # XGBoost
    try:
        import xgboost as xgb
        cls = xgb.XGBRegressor if is_reg else xgb.XGBClassifier
        boosters.append(("XGBoost", cls(
            n_estimators=200, random_state=rs, verbosity=0,
            use_label_encoder=False, eval_metric="logloss" if not is_reg else "rmse",
        )))
    except ImportError:
        logger.info("xgboost not installed, skipping")

    # CatBoost
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
        cls = CatBoostRegressor if is_reg else CatBoostClassifier
        boosters.append(("CatBoost", cls(
            iterations=200, random_seed=rs, verbose=0,
        )))
    except ImportError:
        logger.info("catboost not installed, skipping")

    # sklearn HistGradientBoosting (always available)
    cls = HistGradientBoostingRegressor if is_reg else HistGradientBoostingClassifier
    boosters.append(("HistGradientBoosting", cls(
        max_iter=200, random_state=rs,
    )))

    return boosters


def _stage_hpo(
    prior_trials, task_type, primary,
    X_tr, y_tr, X_te, y_te, rs, max_trials, deadline,
    fitted_models,
) -> List[TrialResult]:
    """Optuna HPO on the top-2 model families."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.info("optuna not installed, skipping HPO")
        return []

    if not prior_trials:
        return []

    # Pick top-2 model names
    sorted_t = sorted(prior_trials, key=lambda t: t.primary_score, reverse=True)
    top_names = []
    seen = set()
    for t in sorted_t:
        if t.model_name not in seen:
            top_names.append(t.model_name)
            seen.add(t.model_name)
        if len(top_names) >= 2:
            break

    hpo_trials: List[TrialResult] = []
    is_reg = task_type == TaskType.REGRESSION

    for model_name in top_names:
        remaining = deadline - time.time()
        if remaining < 10:
            break
        timeout = min(remaining / 2, 60.0)

        try:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=rs),
            )

            def objective(trial, _name=model_name):
                model = _build_hpo_model(trial, _name, is_reg, rs)
                if model is None:
                    raise optuna.TrialPruned()
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                y_proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X_te)
                    except Exception:
                        pass
                metrics = _compute_metrics(y_te, y_pred, y_proba, task_type)
                return _primary_score(metrics, primary)

            study.optimize(objective, n_trials=max_trials, timeout=timeout)

            if study.best_trial is not None:
                best_t = study.best_trial
                # Re-fit best params
                model = _build_hpo_model_from_params(
                    model_name, best_t.params, is_reg, rs,
                )
                if model:
                    tr = _fit_and_score(
                        f"{model_name}_HPO", model, ModelStage.HPO.value,
                        X_tr, y_tr, X_te, y_te, task_type, primary,
                    )
                    hpo_trials.append(tr)
                    fitted_models[tr.model_name] = model
                    logger.info("  HPO %s → %s", model_name, tr.metrics)
        except Exception as e:
            logger.warning("HPO for %s failed: %s", model_name, e)

    return hpo_trials


def _build_hpo_model(trial, model_name: str, is_reg: bool, rs: int):
    """Build a model with Optuna-sampled hyperparameters."""
    if model_name == "HistGradientBoosting":
        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "random_state": rs,
        }
        cls = HistGradientBoostingRegressor if is_reg else HistGradientBoostingClassifier
        return cls(**params)

    if model_name == "LightGBM":
        try:
            import lightgbm as lgb
        except ImportError:
            return None
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_state": rs,
            "verbosity": -1,
        }
        cls = lgb.LGBMRegressor if is_reg else lgb.LGBMClassifier
        return cls(**params)

    if model_name == "XGBoost":
        try:
            import xgboost as xgb
        except ImportError:
            return None
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": rs,
            "verbosity": 0,
        }
        cls = xgb.XGBRegressor if is_reg else xgb.XGBClassifier
        return cls(**params)

    if model_name == "CatBoost":
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
        except ImportError:
            return None
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "random_seed": rs,
            "verbose": 0,
        }
        cls = CatBoostRegressor if is_reg else CatBoostClassifier
        return cls(**params)

    # Baseline models
    if model_name in ("Ridge", "LogisticRegression"):
        alpha_or_C = trial.suggest_float("alpha", 0.001, 100, log=True)
        if is_reg:
            return Ridge(alpha=alpha_or_C, random_state=rs)
        return LogisticRegression(C=1.0 / alpha_or_C, max_iter=1000, random_state=rs)

    return None


def _build_hpo_model_from_params(model_name: str, params: dict, is_reg: bool, rs: int):
    """Rebuild model from best Optuna params (no trial object)."""
    if model_name == "HistGradientBoosting":
        cls = HistGradientBoostingRegressor if is_reg else HistGradientBoostingClassifier
        return cls(random_state=rs, **params)
    if model_name == "LightGBM":
        try:
            import lightgbm as lgb
        except ImportError:
            return None
        cls = lgb.LGBMRegressor if is_reg else lgb.LGBMClassifier
        return cls(random_state=rs, verbosity=-1, **params)
    if model_name == "XGBoost":
        try:
            import xgboost as xgb
        except ImportError:
            return None
        cls = xgb.XGBRegressor if is_reg else xgb.XGBClassifier
        return cls(random_state=rs, verbosity=0, **params)
    if model_name == "CatBoost":
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
        except ImportError:
            return None
        cls = CatBoostRegressor if is_reg else CatBoostClassifier
        return cls(random_seed=rs, verbose=0, **params)
    return None


def _stage_ensemble(
    fitted_models, task_type, primary,
    X_tr, y_tr, X_te, y_te, rs,
) -> Optional[TrialResult]:
    """Build a simple voting ensemble from top fitted models."""
    # Pick up to 3 models (skip ensembles)
    candidates = [
        (name, m) for name, m in fitted_models.items()
        if m is not None and not name.startswith("Voting")
    ][:3]

    if len(candidates) < 2:
        return None

    try:
        estimators = [(n, m) for n, m in candidates]
        is_reg = task_type == TaskType.REGRESSION

        if is_reg:
            ens = VotingRegressor(estimators=estimators)
        else:
            ens = VotingClassifier(estimators=estimators, voting="soft")

        tr = _fit_and_score(
            "VotingEnsemble", ens, ModelStage.ENSEMBLE.value,
            X_tr, y_tr, X_te, y_te, task_type, primary,
        )
        fitted_models["__ensemble__"] = ens
        logger.info("  Ensemble → %s", tr.metrics)
        return tr
    except Exception as e:
        logger.warning("Ensemble failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 5: evaluate_model
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_model(state: PipelineState) -> Dict[str, Any]:
    """Deep evaluation of the best model: holdout, confusion, importance, model card."""
    model = state.best_model
    if model is None:
        return {"status": "error", "message": "No model trained yet"}

    X_te, y_te = state.X_test, state.y_test
    task_type = state.task_type

    y_pred = model.predict(X_te)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_te)
        except Exception:
            pass

    holdout = _compute_metrics(y_te, y_pred, y_proba, task_type)

    cm = None
    cr = None
    residual_stats = None

    if task_type != TaskType.REGRESSION:
        cm = confusion_matrix(y_te, y_pred).tolist()
        cr = classification_report(y_te, y_pred, output_dict=True)
        # Convert numpy types in cr
        cr = json.loads(json.dumps(cr, default=str))
    else:
        residuals = y_te - y_pred
        residual_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "p5": float(np.percentile(residuals, 5)),
            "p95": float(np.percentile(residuals, 95)),
        }

    fi = _extract_feature_importance(model, state.feature_names_out)

    # Model card
    model_card = _generate_model_card(state, holdout, fi)

    state.holdout_metrics = holdout
    state.confusion_matrix = cm
    state.classification_report = cr
    state.residual_stats = residual_stats
    state.feature_importance = fi
    state.model_card_md = model_card

    result: Dict[str, Any] = {
        "status": "ok",
        "holdout_metrics": {k: round(v, 4) for k, v in holdout.items()},
    }
    if cm is not None:
        result["confusion_matrix"] = cm
    if residual_stats:
        result["residual_stats"] = residual_stats
    if fi:
        result["top_features"] = dict(list(fi.items())[:10])
    return result


def _extract_feature_importance(model, feature_names: List[str]) -> Optional[Dict[str, float]]:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return dict(sorted(
            zip(feature_names, [float(v) for v in imp]),
            key=lambda x: -x[1],
        ))
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
        if len(coef) == len(feature_names):
            return dict(sorted(
                zip(feature_names, [float(v) for v in coef]),
                key=lambda x: -x[1],
            ))
    return None


def _generate_model_card(state: PipelineState, metrics: dict, fi: Optional[dict]) -> str:
    bt = state.best_trial
    lines = [
        "# Model Card",
        "",
        f"## Task: {state.task_type.value}",
        f"## Target: {state.config.target_column}",
        f"## Best Model: {bt.model_name} (Stage: {bt.stage})",
        "",
        "## Holdout Metrics",
        "",
    ]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v:.4f}")
    lines.append("")
    lines.append("## Training Details")
    lines.append(f"- Train rows: {state.X_train.shape[0]}")
    lines.append(f"- Test rows: {state.X_test.shape[0]}")
    lines.append(f"- Features: {len(state.feature_names_out)}")
    lines.append(f"- Random seed: {state.config.random_state}")
    lines.append("")
    if fi:
        lines.append("## Top Features")
        for fname, imp in list(fi.items())[:10]:
            lines.append(f"- {fname}: {imp:.4f}")
    lines.append("")
    lines.append("## Parameters")
    lines.append("```json")
    lines.append(json.dumps(bt.params, indent=2, default=str))
    lines.append("```")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 6: package_model
# ═══════════════════════════════════════════════════════════════════════════════


def package_model(state: PipelineState) -> Dict[str, Any]:
    """Bundle all artifacts to the output directory."""
    out_dir = Path(state.config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []

    # model.pkl
    if state.best_model is not None:
        p = out_dir / "model.pkl"
        with open(p, "wb") as f:
            pickle.dump(state.best_model, f)
        files.append(str(p))

    # preprocessor.pkl
    if state.preprocessor is not None:
        p = out_dir / "preprocessor.pkl"
        with open(p, "wb") as f:
            pickle.dump(state.preprocessor, f)
        files.append(str(p))

    # label_encoder.pkl
    if state.label_encoder is not None:
        p = out_dir / "label_encoder.pkl"
        with open(p, "wb") as f:
            pickle.dump(state.label_encoder, f)
        files.append(str(p))

    # schema.json
    if state.schema:
        p = out_dir / "schema.json"
        with open(p, "w") as f:
            json.dump([cs.to_dict() for cs in state.schema], f, indent=2)
        files.append(str(p))

    # task_inference.json
    if state.task_type:
        p = out_dir / "task_inference.json"
        with open(p, "w") as f:
            json.dump({
                "task_type": state.task_type.value,
                "num_classes": state.num_classes,
                "class_labels": state.class_labels,
                "primary_metric": state.primary_metric.value if state.primary_metric else None,
            }, f, indent=2)
        files.append(str(p))

    # eval_report.json
    if state.holdout_metrics:
        p = out_dir / "eval_report.json"
        with open(p, "w") as f:
            json.dump({
                "holdout_metrics": state.holdout_metrics,
                "confusion_matrix": state.confusion_matrix,
                "residual_stats": state.residual_stats,
            }, f, indent=2, default=str)
        files.append(str(p))

    # model_card.md
    if state.model_card_md:
        p = out_dir / "model_card.md"
        with open(p, "w") as f:
            f.write(state.model_card_md)
        files.append(str(p))

    # leaderboard.csv
    if state.leaderboard:
        p = out_dir / "leaderboard.csv"
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=state.leaderboard[0].keys())
            writer.writeheader()
            for row in state.leaderboard:
                writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v
                                 for k, v in row.items()})
        files.append(str(p))

    # trial_log.jsonl
    if state.trials:
        p = out_dir / "trial_log.jsonl"
        with open(p, "w") as f:
            for t in state.trials:
                f.write(json.dumps(t.to_dict(), default=str) + "\n")
        files.append(str(p))

    # reproducibility.lock
    lock = _build_lock(state)
    p = out_dir / "reproducibility.lock"
    with open(p, "w") as f:
        json.dump(lock, f, indent=2)
    files.append(str(p))

    state.output_files = files
    state.reproducibility_lock = lock

    return {
        "status": "ok",
        "output_dir": str(out_dir),
        "files": [os.path.basename(f) for f in files],
        "total_files": len(files),
    }


def _build_lock(state: PipelineState) -> Dict[str, Any]:
    lock: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_state": state.config.random_state,
        "packages": {},
    }
    for pkg in ["sklearn", "numpy", "pandas", "lightgbm", "xgboost", "catboost", "optuna"]:
        try:
            mod = __import__(pkg)
            lock["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            lock["packages"][pkg] = "not_installed"
    try:
        import sklearn
        lock["packages"]["scikit-learn"] = sklearn.__version__
    except ImportError:
        pass
    return lock


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 7: web_search — search the web for domain knowledge
# ═══════════════════════════════════════════════════════════════════════════════


def web_search(state: PipelineState, query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web for domain knowledge relevant to the ML task.

    Uses DuckDuckGo (no API key required) to find information about
    the dataset domain, feature engineering ideas, or modeling strategies.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return {
            "status": "error",
            "message": "duckduckgo-search not installed. Run: pip install duckduckgo-search",
        }

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        snippets = []
        for r in results:
            snippets.append({
                "title": r.get("title", ""),
                "body": r.get("body", "")[:300],
                "href": r.get("href", ""),
            })

        return {
            "status": "ok",
            "query": query,
            "results_count": len(snippets),
            "results": snippets,
        }
    except Exception as e:
        return {"status": "error", "message": f"Search failed: {e}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 8: data_fetch — fetch data from MongoDB collections
# ═══════════════════════════════════════════════════════════════════════════════


def data_fetch(
    state: PipelineState,
    collection: str,
    query: Optional[Dict[str, Any]] = None,
    projection: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Fetch data from a MongoDB collection for enrichment or analysis.

    The agent can pull additional data from the project's MongoDB to
    enrich features, validate results, or explore related datasets.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        return {"status": "error", "message": "pymongo not installed"}

    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/marip_f1")
    db_name = os.getenv("MONGODB_DB", "marip_f1")

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]

        if collection not in db.list_collection_names():
            available = sorted(db.list_collection_names())
            return {
                "status": "error",
                "message": f"Collection '{collection}' not found",
                "available_collections": available[:30],
            }

        cursor = db[collection].find(
            query or {},
            projection or {"_id": 0},
        ).limit(limit)

        docs = []
        for doc in cursor:
            # Convert ObjectId and other non-serializable types
            clean = {}
            for k, v in doc.items():
                if k == "_id":
                    clean[k] = str(v)
                elif hasattr(v, "isoformat"):
                    clean[k] = v.isoformat()
                else:
                    clean[k] = v
            docs.append(clean)

        total = db[collection].estimated_document_count()

        return {
            "status": "ok",
            "collection": collection,
            "documents_returned": len(docs),
            "total_in_collection": total,
            "documents": docs,
        }
    except Exception as e:
        return {"status": "error", "message": f"MongoDB fetch failed: {e}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 9: model_explain — SHAP/feature importance deep dive
# ═══════════════════════════════════════════════════════════════════════════════


def model_explain(
    state: PipelineState,
    top_n: int = 10,
    method: str = "shap",
) -> Dict[str, Any]:
    """Generate detailed model explanations using SHAP or feature importance.

    Goes beyond evaluate_model by providing:
    - SHAP summary (mean absolute SHAP values per feature)
    - Feature interaction detection
    - Per-class explanations (classification)
    - Actionable insights about which features matter most
    """
    if state.best_model is None:
        return {"status": "error", "message": "No trained model. Run train_models first."}
    if state.X_test is None:
        return {"status": "error", "message": "No test data. Run build_features first."}

    model = state.best_model
    X_test = state.X_test
    feature_names = state.feature_names_out or [f"f{i}" for i in range(X_test.shape[1])]

    result: Dict[str, Any] = {"status": "ok", "method": method}

    if method == "shap":
        try:
            import shap

            # Use TreeExplainer for tree models, KernelExplainer as fallback
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test[:min(200, len(X_test))])
            except Exception:
                # KernelExplainer fallback (slower, works for any model)
                sample = X_test[:min(50, len(X_test))]
                explainer = shap.KernelExplainer(model.predict, sample)
                shap_values = explainer.shap_values(sample)

            # Handle multi-output (classification) vs single output
            if isinstance(shap_values, list):
                # Multi-class: average absolute SHAP across classes
                sv = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                sv = np.abs(shap_values)

            mean_shap = np.mean(sv, axis=0)
            importance = dict(zip(feature_names, mean_shap.tolist()))
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            result["shap_importance"] = [
                {"feature": f, "mean_abs_shap": round(v, 4)}
                for f, v in sorted_imp[:top_n]
            ]
            result["total_features"] = len(feature_names)

            # Top feature interactions (features with correlated SHAP values)
            if sv.shape[0] > 10:
                top_idx = [feature_names.index(f) for f, _ in sorted_imp[:5]]
                interactions = []
                for i in range(len(top_idx)):
                    for j in range(i + 1, len(top_idx)):
                        corr = float(np.corrcoef(sv[:, top_idx[i]], sv[:, top_idx[j]])[0, 1])
                        if abs(corr) > 0.3:
                            interactions.append({
                                "feature_a": sorted_imp[i][0],
                                "feature_b": sorted_imp[j][0],
                                "shap_correlation": round(corr, 3),
                            })
                result["interactions"] = interactions

        except ImportError:
            result["status"] = "error"
            result["message"] = "shap not installed. Run: pip install shap"
            return result
    else:
        # Fallback: sklearn feature importance
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(feature_names, model.feature_importances_.tolist()))
        elif hasattr(model, "coef_"):
            coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            importance = dict(zip(feature_names, np.abs(coef).tolist()))
        else:
            return {"status": "error", "message": "Model has no feature_importances_ or coef_"}

        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        result["feature_importance"] = [
            {"feature": f, "importance": round(v, 4)}
            for f, v in sorted_imp[:top_n]
        ]
        result["total_features"] = len(feature_names)

    # Actionable insights
    insights = []
    top_feats = [f for f, _ in sorted_imp[:3]]
    bottom_feats = [f for f, _ in sorted_imp[-3:]] if len(sorted_imp) > 6 else []
    insights.append(f"Top 3 most impactful features: {', '.join(top_feats)}")
    if bottom_feats:
        insights.append(f"Least impactful features (candidates for removal): {', '.join(bottom_feats)}")
    result["insights"] = insights

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 10: knowledge_lookup — curated ML domain knowledge
# ═══════════════════════════════════════════════════════════════════════════════

# Curated ML knowledge entries — seeded into MongoDB on first use
_ML_KNOWLEDGE_ENTRIES = [
    # ── Feature Engineering ──
    {
        "topic": "feature_engineering",
        "tags": ["tabular", "features", "encoding", "transformation"],
        "title": "Tabular Feature Engineering Best Practices",
        "content": (
            "1. **Numeric features**: Apply log transform to right-skewed distributions "
            "(income, prices, counts). Use StandardScaler for linear models, leave raw for "
            "tree models. Create interaction features (A*B) for known domain relationships.\n"
            "2. **Categorical features**: Use target encoding for high-cardinality (>15 unique). "
            "One-hot encode low-cardinality (<15). Ordinal encode ordered categories. "
            "Never one-hot encode IDs or free text.\n"
            "3. **Temporal features**: Extract hour, day_of_week, month, is_weekend. "
            "Create lag features (value_t-1, value_t-7). Rolling statistics (mean_7d, std_30d). "
            "Time since last event.\n"
            "4. **Missing values**: For tree models, leave as-is (they handle NaN natively). "
            "For linear models, impute median (numeric) or mode (categorical). "
            "Add a binary 'is_missing' indicator column for features with >5% missing.\n"
            "5. **Domain-specific**: Ratios often outperform raw values (speed/distance, "
            "cost/unit). Differences from group means capture relative performance. "
            "Polynomial features rarely help with tree models."
        ),
    },
    {
        "topic": "feature_engineering",
        "tags": ["motorsport", "f1", "racing", "telemetry"],
        "title": "F1/Motorsport Feature Engineering",
        "content": (
            "1. **Lap time features**: Mean, std, CV (coefficient of variation) capture pace and "
            "consistency. Late-race pace drop (Q4 vs Q1 lap times) indicates tyre degradation.\n"
            "2. **Speed features**: Speed trap (SpeedST) shows straight-line performance. "
            "Intermediate speeds (SpeedI1, I2) reveal cornering capability. Finish line speed "
            "reflects exit speed from final corner.\n"
            "3. **Sector analysis**: Sector time std reveals consistency through track sections. "
            "High std in one sector suggests setup issues for that track segment.\n"
            "4. **Tyre management**: Tyre-speed correlation measures degradation sensitivity. "
            "Fresh vs worn tyre pace difference quantifies deg rate. Stint-normalized lap "
            "times remove fuel effect contamination.\n"
            "5. **Position dynamics**: Position range (max-min) during race shows race craft. "
            "Lap-to-lap instability (mean abs delta) captures overall consistency."
        ),
    },
    # ── Model Selection ──
    {
        "topic": "model_selection",
        "tags": ["algorithm", "model", "selection", "comparison"],
        "title": "Model Selection Guide for Tabular Data",
        "content": (
            "1. **Start with baselines**: Ridge/Logistic Regression for interpretability baseline. "
            "These set the floor — if they perform well, your features are strong.\n"
            "2. **Gradient boosting is king for tabular**: XGBoost, LightGBM, CatBoost consistently "
            "win Kaggle tabular competitions. LightGBM is fastest for large datasets. XGBoost "
            "for small-medium. CatBoost handles categoricals natively.\n"
            "3. **When to use what**: <1K rows → regularized linear models or small ensembles. "
            "1K-100K → XGBoost/LightGBM with HPO. >100K → LightGBM (speed advantage). "
            "High-cardinality categoricals → CatBoost.\n"
            "4. **Neural networks on tabular**: Generally worse than boosting for <100K rows. "
            "Consider TabNet or FT-Transformer for >100K with complex interactions.\n"
            "5. **Ensembles**: Voting ensemble of 2-3 diverse models (e.g., XGBoost + LightGBM + "
            "Ridge) usually beats any single model by 1-3%. Use weighted voting based on "
            "validation performance."
        ),
    },
    # ── Hyperparameter Tuning ──
    {
        "topic": "hyperparameter_tuning",
        "tags": ["hpo", "optuna", "tuning", "hyperparameters"],
        "title": "Hyperparameter Tuning Strategies",
        "content": (
            "1. **XGBoost key params**: learning_rate (0.01-0.3), max_depth (3-8), "
            "n_estimators (100-1000), subsample (0.6-1.0), colsample_bytree (0.6-1.0), "
            "min_child_weight (1-10), reg_alpha (0-1), reg_lambda (0-10).\n"
            "2. **Search strategy**: Use Optuna with TPE sampler (default). 50-100 trials "
            "is usually sufficient. Early stopping prevents overfitting during tuning.\n"
            "3. **Critical params first**: learning_rate and max_depth have the biggest impact. "
            "Tune these first, then regularization (alpha, lambda, min_child_weight).\n"
            "4. **Avoid overfitting during HPO**: Always use cross-validation (not train/test split) "
            "for HPO objective. Use stratified folds for classification.\n"
            "5. **LightGBM specifics**: num_leaves (more important than max_depth), "
            "min_data_in_leaf, feature_fraction, bagging_fraction."
        ),
    },
    # ── Data Quality ──
    {
        "topic": "data_quality",
        "tags": ["cleaning", "quality", "outliers", "leakage"],
        "title": "Data Quality and Preprocessing",
        "content": (
            "1. **Target leakage**: The #1 killer of ML models. Any feature that wouldn't be "
            "available at prediction time must be removed. Check correlations >0.95 with target.\n"
            "2. **Outlier handling**: For tree models, outliers rarely matter. For linear models, "
            "clip at 1st/99th percentile or use RobustScaler. Never remove outliers without "
            "domain justification.\n"
            "3. **Class imbalance**: For ratios up to 1:10, use scale_pos_weight or class_weight. "
            "For >1:10, combine SMOTE with undersampling. Always stratify train/test splits.\n"
            "4. **Duplicate detection**: Check for exact duplicates and near-duplicates. "
            "Duplicates across train/test cause inflated metrics.\n"
            "5. **Feature types**: Ensure numeric features aren't stored as strings. "
            "Check for sentinel values (-1, 999, -999) masquerading as real values."
        ),
    },
    # ── Evaluation ──
    {
        "topic": "evaluation",
        "tags": ["metrics", "validation", "cross-validation", "evaluation"],
        "title": "Model Evaluation Best Practices",
        "content": (
            "1. **Classification metrics**: AUC-ROC for ranking ability. F1 for imbalanced classes. "
            "Accuracy only when classes are balanced. Log loss for probability calibration.\n"
            "2. **Regression metrics**: RMSE penalizes large errors. MAE is robust to outliers. "
            "R² for explained variance. MAPE for percentage-based interpretation.\n"
            "3. **Cross-validation**: Use stratified k-fold (k=5) for classification. "
            "Time-series data needs temporal CV (no future leakage). Report mean ± std.\n"
            "4. **Holdout evaluation**: Always keep a final holdout set untouched during "
            "development. This is your true performance estimate.\n"
            "5. **Confusion matrix analysis**: Look at the specific failure modes. "
            "False negatives vs false positives have different costs in most domains. "
            "Use precision-recall curve for imbalanced problems."
        ),
    },
    # ── Explainability ──
    {
        "topic": "explainability",
        "tags": ["shap", "interpretability", "feature_importance", "explain"],
        "title": "Model Explainability with SHAP",
        "content": (
            "1. **SHAP values**: Additive explanations that sum to the prediction. "
            "Positive SHAP = pushes prediction higher. Uses game theory (Shapley values).\n"
            "2. **TreeExplainer**: Exact SHAP for tree models (XGBoost, LightGBM). Fast and "
            "precise. Always prefer over KernelExplainer for tree models.\n"
            "3. **Global importance**: Mean absolute SHAP values across all samples. "
            "More reliable than model.feature_importances_ (which can be misleading).\n"
            "4. **Feature interactions**: Correlated SHAP values between features indicate "
            "interaction effects. The model learned that these features work together.\n"
            "5. **Per-prediction explanation**: SHAP waterfall/force plots show why a specific "
            "prediction was made. Essential for debugging individual predictions and "
            "building trust with stakeholders."
        ),
    },
    # ── Production ML ──
    {
        "topic": "production",
        "tags": ["deployment", "monitoring", "drift", "production"],
        "title": "Production ML Considerations",
        "content": (
            "1. **Model serialization**: Use pickle/joblib for sklearn models. ONNX for "
            "cross-platform deployment. Save the preprocessor alongside the model.\n"
            "2. **Data drift detection**: Monitor input feature distributions over time. "
            "KS test or PSI (Population Stability Index) flag distribution shifts.\n"
            "3. **Reproducibility**: Lock random seeds, package versions, and data checksums. "
            "Store the exact training data reference (not copies).\n"
            "4. **Model card**: Document model type, training data, performance metrics, "
            "limitations, and intended use. Essential for team communication.\n"
            "5. **Retraining triggers**: Retrain when performance drops below threshold, "
            "when new data significantly changes distributions, or on a fixed schedule."
        ),
    },
]


def knowledge_lookup(
    state: PipelineState,
    topic: str,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Look up curated ML domain knowledge by topic or tags.

    Searches the ml_knowledge_base collection for relevant best practices,
    strategies, and domain-specific guidance. Topics include:
    feature_engineering, model_selection, hyperparameter_tuning,
    data_quality, evaluation, explainability, production.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        return {"status": "error", "message": "pymongo not installed"}

    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/marip_f1")
    db_name = os.getenv("MONGODB_DB", "marip_f1")

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        coll = db["ml_knowledge_base"]

        # Seed collection on first use if empty
        if coll.estimated_document_count() == 0:
            coll.insert_many(_ML_KNOWLEDGE_ENTRIES)
            coll.create_index("topic")
            coll.create_index("tags")
            logger.info("Seeded ml_knowledge_base with %d entries", len(_ML_KNOWLEDGE_ENTRIES))

        # Build query — match topic substring or tags
        query_filter: Dict[str, Any] = {}
        conditions = []

        if topic:
            conditions.append({"topic": {"$regex": topic, "$options": "i"}})
            conditions.append({"title": {"$regex": topic, "$options": "i"}})
            conditions.append({"content": {"$regex": topic, "$options": "i"}})
            # Also search tags
            conditions.append({"tags": {"$regex": topic, "$options": "i"}})

        if tags:
            conditions.append({"tags": {"$in": tags}})

        if conditions:
            query_filter["$or"] = conditions

        docs = list(coll.find(query_filter, {"_id": 0}).limit(5))

        if not docs:
            # Fallback: return all available topics
            all_topics = coll.distinct("topic")
            return {
                "status": "ok",
                "query": topic,
                "results_count": 0,
                "results": [],
                "available_topics": all_topics,
                "hint": "Try one of the available topics, or search by tag.",
            }

        return {
            "status": "ok",
            "query": topic,
            "tags_filter": tags,
            "results_count": len(docs),
            "results": docs,
        }
    except Exception as e:
        return {"status": "error", "message": f"Knowledge lookup failed: {e}"}


# ═══════════════════════════════════════════════════════════════════════════════
# RAG vector search
# ═══════════════════════════════════════════════════════════════════════════════

_rag_retriever = None


def _get_rag_retriever():
    """Lazy singleton for the RAG retriever (same pattern as omniagents.base)."""
    global _rag_retriever
    if _rag_retriever is None:
        try:
            from omnirag.retriever import RAGRetriever
            _rag_retriever = RAGRetriever()
            logger.info("RAG retriever initialized for OnMichine")
        except Exception as e:
            logger.warning("RAG retriever unavailable: %s", e)
            return None
    return _rag_retriever


def rag_search(
    state: PipelineState,
    query: str,
    k: int = 5,
) -> Dict[str, Any]:
    """Search the RAG vectorstore for relevant ML methodology and domain knowledge.

    Queries the BGE-1024 embedding vectorstore containing ML reference books
    (e.g. Timothy Masters' Statistically Sound Machine Learning), F1 regulations,
    and other ingested documents. Use for statistical validation methodology,
    feature engineering strategies, model evaluation best practices, etc.
    """
    retriever = _get_rag_retriever()
    if retriever is None:
        return {
            "status": "error",
            "message": "RAG retriever not available (check omnirag installation)",
        }

    try:
        context = retriever.get_relevant_context(query, k=min(k, 10))
        if not context or context == "No relevant context found.":
            return {
                "status": "ok",
                "query": query,
                "results_count": 0,
                "context": "",
                "hint": "No matching documents. Try different keywords or check that PDFs have been ingested.",
            }

        return {
            "status": "ok",
            "query": query,
            "results_count": context.count("\n---\n") + 1,
            "context": context,
        }
    except Exception as e:
        return {"status": "error", "message": f"RAG search failed: {e}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registry builder
# ═══════════════════════════════════════════════════════════════════════════════


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return OpenAI/Groq function-calling schemas for all tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "load_and_profile",
                "description": (
                    "Load the dataset from the configured path, profile all columns "
                    "(detect types, roles, nulls, cardinality), detect data quality issues, "
                    "and build the schema. This must be called first."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "infer_task",
                "description": (
                    "Infer the ML task type (regression, binary classification, or "
                    "multiclass classification) from the target column's dtype and cardinality. "
                    "Also sets the primary evaluation metric. Requires load_and_profile first."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "build_features",
                "description": (
                    "Build an sklearn preprocessing pipeline (imputation, scaling, encoding), "
                    "encode the target variable, and split into train/test sets. "
                    "Requires infer_task first."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "train_models",
                "description": (
                    "Run multi-stage model search: Stage 0 baselines (Ridge/Logistic), "
                    "Stage 1 boosting (LightGBM/XGBoost/CatBoost/HistGB), "
                    "Stage 2 Optuna HPO on top models, Stage 3 voting ensemble. "
                    "Time-budget aware. Requires build_features first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stages": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["baseline", "boosting", "hpo", "ensemble"]},
                            "description": (
                                "Which stages to run. Default: all. "
                                "Options: baseline, boosting, hpo, ensemble."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "evaluate_model",
                "description": (
                    "Deep evaluation of the best model: holdout metrics, confusion matrix "
                    "(classification) or residual analysis (regression), feature importance, "
                    "and model card generation. Requires train_models first."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "package_model",
                "description": (
                    "Serialize all artifacts to the output directory: model.pkl, "
                    "preprocessor.pkl, schema.json, leaderboard.csv, trial_log.jsonl, "
                    "model_card.md, eval_report.json, reproducibility.lock. "
                    "Requires evaluate_model first."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for domain knowledge about the dataset, "
                    "feature engineering techniques, or modeling strategies. "
                    "Use this to understand the problem domain better."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g. 'best features for predicting house prices')",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max results to return (default: 5)",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "data_fetch",
                "description": (
                    "Fetch data from a MongoDB collection for enrichment or analysis. "
                    "Use this to pull additional data that could improve the model, "
                    "validate results, or explore related datasets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "MongoDB collection name to query",
                        },
                        "query": {
                            "type": "object",
                            "description": "MongoDB query filter (default: {})",
                        },
                        "projection": {
                            "type": "object",
                            "description": "Fields to include/exclude (default: {_id: 0})",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max documents to return (default: 100)",
                        },
                    },
                    "required": ["collection"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "model_explain",
                "description": (
                    "Generate detailed model explanations using SHAP or feature importance. "
                    "Provides per-feature impact, feature interactions, and actionable insights "
                    "about which features matter most. Call after train_models."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": {
                            "type": "integer",
                            "description": "Number of top features to show (default: 10)",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["shap", "importance"],
                            "description": "Explanation method: 'shap' (detailed) or 'importance' (fast)",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "knowledge_lookup",
                "description": (
                    "Look up curated ML domain knowledge — best practices for feature engineering, "
                    "model selection, hyperparameter tuning, data quality, evaluation, explainability, "
                    "and production deployment. Also includes F1/motorsport-specific ML guidance. "
                    "Use this instead of web_search for ML methodology questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": (
                                "Topic to search: feature_engineering, model_selection, "
                                "hyperparameter_tuning, data_quality, evaluation, explainability, "
                                "production, f1, motorsport, or any keyword."
                            ),
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags to filter by (e.g. ['tabular', 'shap'])",
                        },
                    },
                    "required": ["topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": (
                    "Search the RAG vectorstore for ML methodology, statistical validation, "
                    "and domain knowledge from ingested reference books and documents. "
                    "Contains insights from 'Statistically Sound Machine Learning' (Timothy Masters), "
                    "F1 regulations, and other technical references. Use for: permutation testing, "
                    "cross-validation strategies, feature selection validation, model significance "
                    "testing, stationarity checks, and advanced ML best practices."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural language query describing the ML concept, technique, "
                                "or methodology to search for."
                            ),
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return (default 5, max 10)",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]
