"""Temporal feature engineering, rolling statistics, and ML preparation."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_split
from sklearn.preprocessing import StandardScaler

from omnidata._types import TabularDataset

logger = logging.getLogger(__name__)


# ── Label Leakage Prevention (from F1 classifier.py) ─────────────────

_LABEL_LEAK_PATTERN = re.compile(
    r"(anomaly|error|ensemble|score|level|cluster|voted|weighted|"
    r"enhanced|dynamic|severity|distance|reliability|voting|prediction|"
    r"predicted|target|label|class_)",
    re.IGNORECASE,
)
_IDENTIFIER_PATTERN = re.compile(
    r"^(id|_id|index|idx|name|label|target|class|row_number)$",
    re.IGNORECASE,
)


def select_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove label-leaking and identifier columns for ML training."""
    numeric = df.select_dtypes(include=[np.number])
    safe = [
        c for c in numeric.columns
        if not _LABEL_LEAK_PATTERN.search(c) and not _IDENTIFIER_PATTERN.match(c)
    ]
    return numeric[safe]


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace chars that LightGBM/XGBoost can't handle."""
    sanitized = [re.sub(r"[^a-zA-Z0-9_]", "_", str(c)) for c in df.columns]
    seen: Dict[str, int] = {}
    unique = []
    for c in sanitized:
        if c in seen:
            seen[c] += 1
            unique.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            unique.append(c)
    out = df.copy()
    out.columns = unique
    return out


# ── Temporal Features ────────────────────────────────────────────────

def add_temporal_features(
    df: pd.DataFrame,
    columns: List[str],
    *,
    delta: bool = True,
    rolling_window: int = 3,
    rolling_mean: bool = True,
    rolling_std: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add row-over-row deltas and rolling statistics.

    For each column generates:
      - {col}_delta: row-over-row difference
      - {col}_roll{N}_mean: rolling mean
      - {col}_roll{N}_std: rolling standard deviation
    """
    new_cols = []
    for col in columns:
        if col not in df.columns:
            continue

        if delta:
            delta_col = f"{col}_delta"
            df[delta_col] = df[col].diff().fillna(0)
            new_cols.append(delta_col)

        if rolling_mean:
            rm_col = f"{col}_roll{rolling_window}_mean"
            df[rm_col] = df[col].rolling(rolling_window, min_periods=1).mean()
            new_cols.append(rm_col)

        if rolling_std:
            rs_col = f"{col}_roll{rolling_window}_std"
            df[rs_col] = df[col].rolling(rolling_window, min_periods=1).std().fillna(0)
            new_cols.append(rs_col)

    return df, new_cols


def add_sequence_context(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add position-in-sequence features for wear/degradation modeling."""
    n = len(df)
    df["row_index"] = np.arange(n)
    df["sequence_pct"] = df["row_index"] / max(n - 1, 1)
    df["is_second_half"] = (df["sequence_pct"] >= 0.5).astype(int)
    return df, ["row_index", "sequence_pct", "is_second_half"]


# ── ML Preparation ───────────────────────────────────────────────────

def train_test_split(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    *,
    target_col: Optional[str] = None,
    test_ratio: float = 0.2,
    scale: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Prepare train/test split with optional StandardScaler.

    Returns dict: X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    if columns:
        feature_df = df[columns].copy()
    else:
        feature_df = select_safe_features(df)

    feature_df = sanitize_feature_names(feature_df)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].values

    if y is not None:
        X_train, X_test, y_train, y_test = sklearn_split(
            feature_df, y, test_size=test_ratio, random_state=random_state,
        )
    else:
        X_train, X_test = sklearn_split(
            feature_df, test_size=test_ratio, random_state=random_state,
        )
        y_train, y_test = None, None

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index,
        )
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(feature_df.columns),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def engineer(
    dataset: TabularDataset,
    *,
    temporal: bool = True,
    rolling_window: int = 3,
    sequence_context: bool = True,
) -> Tuple[TabularDataset, List[str]]:
    """Full feature engineering pipeline on metric columns."""
    from omnidata.profiler import profile as run_profile

    if not dataset.profile.columns:
        run_profile(dataset)

    metric_cols = dataset.profile.metric_cols
    all_new = []

    if temporal and metric_cols:
        dataset.df, new = add_temporal_features(
            dataset.df, metric_cols, rolling_window=rolling_window,
        )
        all_new.extend(new)
        dataset.preprocessing_applied.append("temporal_features")

    if sequence_context:
        dataset.df, new = add_sequence_context(dataset.df)
        all_new.extend(new)
        dataset.preprocessing_applied.append("sequence_context")

    return dataset, all_new
