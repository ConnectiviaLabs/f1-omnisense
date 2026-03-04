"""OmniData — unified tabular data loading, profiling, and preprocessing.

Quick start:
    from omnidata import load, profile, preprocess, engineer

    dataset = load("data.csv")
    profile(dataset)
    dataset = preprocess(dataset)
    dataset, new_cols = engineer(dataset)
"""

from omnidata.loader import load, detect_format
from omnidata.profiler import profile, filter_sensor_columns
from omnidata.preprocessing import (
    preprocess, safe_float, safe_int, safe_str, normalize_timestamps,
)
from omnidata.features import (
    engineer, add_temporal_features, add_sequence_context,
    select_safe_features, sanitize_feature_names, train_test_split,
)

from omnidata._types import (
    TabularDataset, DatasetProfile, ColumnProfile, ColumnRole, DType,
)

__all__ = [
    "load", "detect_format",
    "profile", "filter_sensor_columns",
    "preprocess", "safe_float", "safe_int", "safe_str", "normalize_timestamps",
    "engineer", "add_temporal_features", "add_sequence_context",
    "select_safe_features", "sanitize_feature_names", "train_test_split",
    "TabularDataset", "DatasetProfile", "ColumnProfile", "ColumnRole", "DType",
]
