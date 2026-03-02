"""Column profiling, role detection, and sensor column filtering.

6-layer heuristic adapted from DataSense sensorColumns.js:
  1. Internal field patterns (_id, __v, etc.)
  2. Excluded exact names (timestamp, id, sensor_id, etc.)
  3. Excluded substrings (anomaly, _id, latitude, etc.)
  4. Numeric value check
  5. Epoch timestamp detection
  6. Geographic coordinate detection
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from omnidata._types import (
    ColumnProfile, ColumnRole, DatasetProfile, DType, TabularDataset,
)

logger = logging.getLogger(__name__)

# ── Layer 1: Internal field patterns ─────────────────────────────────

INTERNAL_RE = re.compile(r"^(_|__)", re.IGNORECASE)

# ── Layer 2: Excluded exact names ────────────────────────────────────

EXCLUDED_NAMES = {
    "id", "index", "idx", "pk", "uuid", "objectid", "row", "row_number",
    "timestamp", "date", "time", "datetime", "created_at", "updated_at",
    "created", "modified", "inserted_at",
    "latitude", "longitude", "lat", "lng", "lon",
    "sensor_id", "device_id", "site_id", "asset_id", "unit_id",
    "station_id", "node_id", "gateway_id",
}

# ── Layer 3: Excluded substrings ─────────────────────────────────────

EXCLUDED_SUBSTRINGS = [
    "anomaly", "_id", "timestamp", "latitude", "longitude",
    "created_at", "updated_at", "version", "object_id",
]

# ── Layer 5: Epoch detection constants ───────────────────────────────

EPOCH_SECONDS_MIN = 1e9    # ~2001
EPOCH_MILLIS_MIN = 1e12    # ~2001 in ms

# ── Layer 6: Geo detection ───────────────────────────────────────────

GEO_NAME_RE = re.compile(r"(lat|lng|lon|longitude|latitude|coord)", re.IGNORECASE)


def detect_column_dtype(series: pd.Series) -> DType:
    """Infer the coerced data type for a column."""
    if pd.api.types.is_bool_dtype(series):
        return DType.BOOL
    if pd.api.types.is_datetime64_any_dtype(series):
        return DType.DATETIME
    if pd.api.types.is_integer_dtype(series):
        return DType.INT
    if pd.api.types.is_float_dtype(series):
        return DType.FLOAT

    # Object columns — try to infer
    non_null = series.dropna()
    if len(non_null) == 0:
        return DType.STRING

    # Check if parseable as numeric
    numeric = pd.to_numeric(non_null, errors="coerce")
    valid_ratio = numeric.notna().sum() / len(non_null)
    if valid_ratio > 0.8:
        if (numeric.dropna() % 1 == 0).all():
            return DType.INT
        return DType.FLOAT

    # Check if parseable as datetime
    try:
        dt = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
        if dt.notna().sum() / len(non_null) > 0.8:
            return DType.DATETIME
    except Exception:
        pass

    return DType.STRING


def detect_column_role(name: str, series: pd.Series, dtype: DType) -> ColumnRole:
    """6-layer heuristic to classify column role."""
    lower = name.lower().strip()

    # Layer 1: Internal
    if INTERNAL_RE.match(name):
        return ColumnRole.INTERNAL

    # Layer 2: Excluded exact names
    if lower in EXCLUDED_NAMES:
        if lower in ("timestamp", "date", "time", "datetime", "created_at",
                      "updated_at", "created", "modified", "inserted_at"):
            return ColumnRole.TIMESTAMP
        if lower in ("latitude", "longitude", "lat", "lng", "lon"):
            return ColumnRole.GEO
        return ColumnRole.IDENTIFIER

    # Layer 3: Excluded substrings
    for sub in EXCLUDED_SUBSTRINGS:
        if sub in lower:
            if "timestamp" in lower or "date" in lower:
                return ColumnRole.TIMESTAMP
            return ColumnRole.INTERNAL

    # Layer 4: Numeric check
    if dtype in (DType.FLOAT, DType.INT):
        numeric_vals = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(numeric_vals) == 0:
            return ColumnRole.CATEGORICAL

        # Layer 5: Epoch detection
        if _is_epoch_column(lower, numeric_vals):
            return ColumnRole.TIMESTAMP

        # Layer 6: Geo detection
        if _is_geo_column(lower, numeric_vals):
            return ColumnRole.GEO

        return ColumnRole.METRIC

    if dtype == DType.DATETIME:
        return ColumnRole.TIMESTAMP

    if dtype == DType.BOOL:
        return ColumnRole.CATEGORICAL

    # String columns — check cardinality for identifier vs categorical
    non_null = series.dropna()
    if len(non_null) > 0:
        unique_ratio = non_null.nunique() / len(non_null)
        if unique_ratio > 0.9:
            return ColumnRole.IDENTIFIER
    return ColumnRole.CATEGORICAL


def _is_epoch_column(name: str, values: pd.Series) -> bool:
    """Detect if numeric column contains Unix timestamps."""
    if len(values) == 0:
        return False
    sample = values.head(100)
    epoch_count = ((sample >= EPOCH_SECONDS_MIN) & (sample < 1e15)).sum()
    return epoch_count / len(sample) > 0.5


def _is_geo_column(name: str, values: pd.Series) -> bool:
    """Detect lat/lon by name pattern and value range [-180, 180]."""
    if GEO_NAME_RE.search(name):
        return True
    # Value-range heuristic: all values in [-180, 180]
    sample = values.head(100)
    if len(sample) > 0 and sample.between(-180, 180).all():
        # Only flag if name hints at geo
        if any(kw in name.lower() for kw in ("lat", "lon", "lng", "coord", "geo")):
            return True
    return False


def profile_column(name: str, series: pd.Series) -> ColumnProfile:
    """Compute full statistical profile for a single column."""
    dtype = detect_column_dtype(series)
    role = detect_column_role(name, series, dtype)

    null_count = int(series.isna().sum())
    total = len(series)

    cp = ColumnProfile(
        name=name,
        dtype=dtype,
        role=role,
        null_count=null_count,
        null_pct=round(null_count / total, 4) if total > 0 else 0.0,
        unique_count=int(series.nunique()),
    )

    if dtype in (DType.FLOAT, DType.INT) and role == ColumnRole.METRIC:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) > 0:
            cp.min = round(float(numeric.min()), 4)
            cp.max = round(float(numeric.max()), 4)
            cp.mean = round(float(numeric.mean()), 4)
            cp.std = round(float(numeric.std()), 4)
            cp.median = round(float(numeric.median()), 4)
            cp.q25 = round(float(numeric.quantile(0.25)), 4)
            cp.q75 = round(float(numeric.quantile(0.75)), 4)

    elif dtype == DType.DATETIME or role == ColumnRole.TIMESTAMP:
        try:
            dt = pd.to_datetime(series, errors="coerce")
            valid = dt.dropna()
            if len(valid) > 0:
                cp.min_time = str(valid.min())
                cp.max_time = str(valid.max())
        except Exception:
            pass

    elif role == ColumnRole.CATEGORICAL:
        vc = series.value_counts().head(10)
        cp.top_values = [
            {"value": str(v), "count": int(c)} for v, c in vc.items()
        ]

    return cp


def profile(dataset: TabularDataset) -> DatasetProfile:
    """Compute full dataset profile with column roles and stats."""
    df = dataset.df
    columns = []
    timestamp_col = None
    identifier_cols = []
    metric_cols = []

    for col in df.columns:
        cp = profile_column(col, df[col])
        columns.append(cp)

        if cp.role == ColumnRole.TIMESTAMP and timestamp_col is None:
            timestamp_col = col
        elif cp.role == ColumnRole.IDENTIFIER:
            identifier_cols.append(col)
        elif cp.role == ColumnRole.METRIC:
            metric_cols.append(col)

    dataset.profile = DatasetProfile(
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        timestamp_col=timestamp_col,
        identifier_cols=identifier_cols,
        metric_cols=metric_cols,
        format_detected=dataset.profile.format_detected,
        size_bytes=dataset.profile.size_bytes,
        sampled=dataset.profile.sampled,
        sample_rows=dataset.profile.sample_rows,
    )
    return dataset.profile


def filter_sensor_columns(
    dataset: TabularDataset,
    *,
    extra_exclude: Optional[List[str]] = None,
) -> List[str]:
    """Return only metric column names suitable for analysis."""
    if not dataset.profile.columns:
        profile(dataset)

    exclude = set(extra_exclude or [])
    return [
        cp.name for cp in dataset.profile.columns
        if cp.role == ColumnRole.METRIC and cp.name not in exclude
    ]
