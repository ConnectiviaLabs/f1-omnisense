"""Health scoring per component — wraps omnianalytics.AnomalyEnsemble.

Ports F1 health formula (100 - score*80) and temporal feature engineering.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from omnianalytics._types import AnomalyResult, SeverityLevel
from omnihealth._types import (
    HealthScore,
    MaintenanceAction,
    RiskLevel,
    SEVERITY_TO_ACTION,
)

logger = logging.getLogger(__name__)


# ── Health scoring ───────────────────────────────────────────────────────────

def score_health(
    anomaly_score: float,
    clamp_low: float = 10.0,
    clamp_high: float = 100.0,
) -> float:
    """Convert anomaly score (0-1) to health percentage (0-100).

    Formula from F1 FleetOverview: health = 100 - (score * 80).
    """
    raw = 100.0 - (anomaly_score * 80.0)
    return float(np.clip(raw, clamp_low, clamp_high))


def _worst_severity(scores: list) -> SeverityLevel:
    """Return the worst (highest) severity across a list of anomaly scores."""
    order = [SeverityLevel.NORMAL, SeverityLevel.LOW, SeverityLevel.MEDIUM,
             SeverityLevel.HIGH, SeverityLevel.CRITICAL]
    worst_idx = 0
    for s in scores:
        idx = order.index(s.severity) if s.severity in order else 0
        if idx > worst_idx:
            worst_idx = idx
    return order[worst_idx]


def _risk_from_severity(severity: SeverityLevel) -> RiskLevel:
    """Map severity → risk level."""
    if severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
        return RiskLevel.HIGH
    elif severity == SeverityLevel.MEDIUM:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# Action ordering for consensus downgrade
_ACTION_ORDER = [
    MaintenanceAction.NONE,
    MaintenanceAction.LOG,
    MaintenanceAction.LOG_AND_MONITOR,
    MaintenanceAction.ALERT,
    MaintenanceAction.ALERT_AND_REMEDIATE,
]


def _consensus_gated_action(
    severity: SeverityLevel,
    total_votes: int,
    total_models: int,
    n_rows: int,
) -> MaintenanceAction:
    """Data-driven action mapping: gate by ensemble vote consensus.

    The base action comes from the static severity map. Then the average
    vote consensus (what fraction of models agreed on anomaly per row)
    determines whether to keep, downgrade by 1, or downgrade by 2 levels.

    - >=75% consensus → full action (models strongly agree)
    - >=50% consensus → downgrade 1 level (moderate agreement)
    - <50% consensus  → downgrade 2 levels (weak — maybe noise)
    """
    base_action = SEVERITY_TO_ACTION.get(severity, MaintenanceAction.NONE)
    base_idx = _ACTION_ORDER.index(base_action)

    # Average consensus across rows
    max_possible_votes = total_models * max(n_rows, 1)
    consensus = total_votes / max_possible_votes if max_possible_votes > 0 else 0.0

    if consensus >= 0.75:
        downgrade = 0
    elif consensus >= 0.50:
        downgrade = 1
    else:
        downgrade = 2

    final_idx = max(0, base_idx - downgrade)
    return _ACTION_ORDER[final_idx]


# ── Component assessment ─────────────────────────────────────────────────────

def assess_component(
    data: pd.DataFrame,
    component: str,
    columns: Optional[List[str]] = None,
    *,
    weights: Optional[Dict[str, float]] = None,
) -> HealthScore:
    """Run omnianalytics.AnomalyEnsemble on component data, compute health score.

    Parameters
    ----------
    data : DataFrame with numeric columns.
    component : Component name (e.g. "Power Unit").
    columns : Columns to analyze (default: all numeric).
    weights : Model weights passed to AnomalyEnsemble.

    Returns
    -------
    HealthScore with health percentage, severity, and maintenance action.
    """
    from omnianalytics.anomaly import AnomalyEnsemble
    from omnidata._types import TabularDataset, DatasetProfile, ColumnProfile, ColumnRole, DType

    cols = columns or [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    if not cols:
        raise ValueError(f"No numeric columns found for component '{component}'")

    df_sub = data[cols].copy()

    # Build a minimal TabularDataset for AnomalyEnsemble
    col_profiles = [
        ColumnProfile(
            name=c,
            dtype=DType.FLOAT,
            role=ColumnRole.METRIC,
            null_count=int(df_sub[c].isna().sum()),
            unique_count=int(df_sub[c].nunique()),
        )
        for c in cols
    ]
    profile = DatasetProfile(
        row_count=len(df_sub),
        column_count=len(cols),
        columns=col_profiles,
        metric_cols=cols,
    )
    dataset = TabularDataset(df=df_sub, profile=profile)

    ensemble = AnomalyEnsemble()
    result: AnomalyResult = ensemble.run(
        dataset, columns=cols, weights=weights, explain_critical=True,
    )

    # Aggregate scores — SHAP-weighted
    # Rows with SHAP explanations (HIGH/CRITICAL) contribute more to the
    # health score, proportional to their total SHAP importance.
    # Rows without SHAP get a base weight of 1.0.
    row_scores = []
    row_weights = []
    for s in result.scores:
        row_scores.append(s.score_mean)
        if "shap_top_features" in s.model_scores:
            total_imp = sum(f.get("importance", 0.0) for f in s.model_scores["shap_top_features"])
            row_weights.append(1.0 + total_imp)
        else:
            row_weights.append(1.0)

    row_scores = np.array(row_scores)
    row_weights = np.array(row_weights)
    mean_score = float(np.average(row_scores, weights=row_weights))
    health_pct = score_health(mean_score)

    # Use the worst severity across all rows
    severity = _worst_severity(result.scores)
    risk = _risk_from_severity(severity)

    # Consensus-gated action: use vote agreement to gate escalation
    # High consensus (>=75%) → full action for severity
    # Medium consensus (>=50%) → downgrade one level
    # Low consensus (<50%) → downgrade two levels
    action = _consensus_gated_action(severity, total_votes=sum(s.vote_count for s in result.scores),
                                      total_models=result.scores[0].total_models if result.scores else 4,
                                      n_rows=len(result.scores))

    # Aggregate vote counts
    total_votes = sum(s.vote_count for s in result.scores)
    total_models = result.scores[0].total_models if result.scores else 4

    # Collect top features from SHAP explanations on critical rows
    top_features: List[Dict[str, float]] = []
    for s in result.scores:
        if "shap_top_features" in s.model_scores:
            for feat in s.model_scores["shap_top_features"]:
                top_features.append(feat)

    # Deduplicate features by name, keep highest importance
    seen: Dict[str, float] = {}
    for f in top_features:
        name = f.get("feature", "")
        imp = f.get("importance", 0.0)
        if name not in seen or imp > seen[name]:
            seen[name] = imp
    deduped = [{"feature": k, "importance": v} for k, v in
               sorted(seen.items(), key=lambda x: -x[1])[:5]]

    return HealthScore(
        component=component,
        health_pct=round(health_pct, 1),
        risk_level=risk,
        severity=severity,
        action=action,
        anomaly_score=round(mean_score, 4),
        vote_count=total_votes,
        total_models=total_models,
        top_features=deduped,
        metadata={
            "columns": cols,
            "anomaly_count": result.anomaly_count,
            "total_rows": result.total_rows,
            "severity_distribution": result.severity_distribution,
        },
    )


def assess_components(
    data: pd.DataFrame,
    component_map: Dict[str, List[str]],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> List[HealthScore]:
    """Assess multiple components.

    Parameters
    ----------
    data : Full DataFrame with all columns.
    component_map : {"Power Unit": ["RPM", "nGear"], "Brakes": ["Brake", "Speed"], ...}
    weights : Model weights for AnomalyEnsemble.

    Returns
    -------
    List of HealthScore, one per component.
    """
    results = []
    for component, cols in component_map.items():
        available = [c for c in cols if c in data.columns]
        if not available:
            logger.warning("No columns found for component '%s', skipping", component)
            continue
        try:
            hs = assess_component(data, component, available, weights=weights)
            results.append(hs)
        except (ValueError, Exception) as e:
            logger.warning("Failed to assess component '%s': %s", component, e)
    return results


# ── Temporal feature engineering ─────────────────────────────────────────────

def add_temporal_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    window: int = 3,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add degradation signals (from F1 F1FeatureEngineer).

    Creates for each column:
    - {col}_delta: row-over-row change (degradation signal)
    - {col}_roll{window}_mean: rolling mean
    - {col}_roll{window}_std: rolling std deviation

    Parameters
    ----------
    df : Input DataFrame.
    feature_cols : Columns to create temporal features for.
    window : Rolling window size (default 3).

    Returns
    -------
    (enriched_df, new_column_names)
    """
    out = df.copy()
    new_cols = []

    for col in feature_cols:
        if col not in out.columns:
            continue

        delta_col = f"{col}_delta"
        roll_mean = f"{col}_roll{window}_mean"
        roll_std = f"{col}_roll{window}_std"

        out[delta_col] = out[col].diff().fillna(0)
        out[roll_mean] = out[col].rolling(window, min_periods=1).mean()
        out[roll_std] = out[col].rolling(window, min_periods=1).std().fillna(0)

        new_cols.extend([delta_col, roll_mean, roll_std])

    return out, new_cols


def add_lifecycle_context(
    df: pd.DataFrame,
    total_periods: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add lifecycle wear features (from F1 season context).

    Creates:
    - period_index: row index (0 to n-1)
    - lifecycle_pct: normalized position (0.0 to 1.0)
    - is_second_half: binary flag (1 if lifecycle_pct >= 0.5)

    Parameters
    ----------
    df : Input DataFrame.
    total_periods : Total expected periods (default: len(df)).

    Returns
    -------
    (enriched_df, new_column_names)
    """
    out = df.copy()
    n = total_periods or len(out)

    out["period_index"] = range(len(out))
    out["lifecycle_pct"] = out["period_index"] / max(n - 1, 1)
    out["is_second_half"] = (out["lifecycle_pct"] >= 0.5).astype(int)

    return out, ["period_index", "lifecycle_pct", "is_second_half"]
