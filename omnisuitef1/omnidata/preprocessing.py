"""Safe type coercion, timestamp normalization, and data cleaning."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from omnidata._types import ColumnRole, DType, TabularDataset

logger = logging.getLogger(__name__)


# ── Safe Type Coercion (from cadAI) ──────────────────────────────────

def safe_float(val) -> Optional[float]:
    """Coerce to float, handling European comma decimals."""
    if val is None:
        return None
    try:
        if isinstance(val, str):
            val = val.replace(",", ".").strip()
            # Strip trailing non-numeric chars (unit suffixes like °C, V, dBm, %)
            val = re.sub(r"[^\d.\-eE]+$", "", val).strip()
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val) -> Optional[int]:
    """Coerce to int via float (handles '3.0' -> 3)."""
    if val is None:
        return None
    try:
        return int(float(str(val).replace(",", ".").strip()))
    except (ValueError, TypeError):
        return None


def safe_str(val) -> str:
    """Coerce to stripped string. None -> ''."""
    if val is None:
        return ""
    return str(val).strip()


# ── Timestamp Normalization ──────────────────────────────────────────

def normalize_timestamps(
    df: pd.DataFrame,
    col: str,
    *,
    target_tz: str = "UTC",
) -> pd.Series:
    """Normalize a timestamp column to pandas Timestamp.

    Handles ISO 8601, Unix seconds, Unix millis, and MongoDB $date.
    """
    series = df[col].copy()

    # Expand MongoDB $date objects
    if series.apply(lambda v: isinstance(v, dict) and "$date" in v).any():
        series = series.apply(_parse_mongo_date)

    # Try direct pandas parsing
    result = pd.to_datetime(series, errors="coerce", utc=True)

    # For numeric values, check if they're epoch timestamps
    mask_null = result.isna() & series.notna()
    if mask_null.any():
        numeric = pd.to_numeric(series[mask_null], errors="coerce")
        # Unix seconds (< 1e12) vs millis (>= 1e12)
        is_millis = numeric >= 1e12
        seconds = numeric.copy()
        seconds[is_millis] = numeric[is_millis] / 1000
        epoch_dt = pd.to_datetime(seconds, unit="s", errors="coerce", utc=True)
        result[mask_null] = epoch_dt

    return result


def _parse_mongo_date(val):
    """Parse MongoDB $date format."""
    if not isinstance(val, dict):
        return val
    date_val = val.get("$date")
    if isinstance(date_val, dict):
        # {$date: {$numberLong: "..."}}
        num_long = date_val.get("$numberLong", "0")
        return int(num_long)
    return date_val


# ── Data Cleaning ────────────────────────────────────────────────────

def coerce_column_types(dataset: TabularDataset) -> TabularDataset:
    """Apply safe type coercion based on detected column profiles."""
    df = dataset.df

    for cp in dataset.profile.columns:
        if cp.role == ColumnRole.METRIC:
            if cp.dtype in (DType.FLOAT, DType.MIXED):
                df[cp.name] = df[cp.name].apply(safe_float)
            elif cp.dtype == DType.INT:
                df[cp.name] = df[cp.name].apply(safe_int)

    dataset.preprocessing_applied.append("coerce_types")
    return dataset


def fill_missing(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "median",
) -> pd.DataFrame:
    """Fill missing values in specified columns."""
    for col in columns:
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        if strategy == "median":
            df[col] = numeric.fillna(numeric.median())
        elif strategy == "mean":
            df[col] = numeric.fillna(numeric.mean())
        elif strategy == "zero":
            df[col] = numeric.fillna(0)
        elif strategy == "ffill":
            df[col] = numeric.ffill().bfill()
        elif strategy == "drop":
            pass  # handled externally
        else:
            df[col] = numeric.fillna(numeric.median())

    return df


def preprocess(
    dataset: TabularDataset,
    *,
    coerce_types: bool = True,
    normalize_time: bool = True,
    fill_strategy: str = "median",
) -> TabularDataset:
    """Full preprocessing pipeline."""
    from omnidata.profiler import profile as run_profile

    # Ensure we have a profile
    if not dataset.profile.columns:
        run_profile(dataset)

    if coerce_types:
        coerce_column_types(dataset)

    if normalize_time and dataset.profile.timestamp_col:
        col = dataset.profile.timestamp_col
        dataset.df[col] = normalize_timestamps(dataset.df, col)
        dataset.preprocessing_applied.append("normalize_timestamps")

    # Fill missing in metric columns
    metric_cols = dataset.profile.metric_cols
    if metric_cols:
        fill_missing(dataset.df, metric_cols, strategy=fill_strategy)
        dataset.preprocessing_applied.append(f"fill_missing_{fill_strategy}")

    return dataset
