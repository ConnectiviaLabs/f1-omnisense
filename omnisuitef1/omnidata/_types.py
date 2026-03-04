"""Shared type definitions for the omnidata tabular data service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ColumnRole(str, Enum):
    TIMESTAMP = "timestamp"
    IDENTIFIER = "identifier"
    METRIC = "metric"
    CATEGORICAL = "categorical"
    GEO = "geo"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class DType(str, Enum):
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    DATETIME = "datetime"
    MIXED = "mixed"


@dataclass
class ColumnProfile:
    name: str
    dtype: DType
    role: ColumnRole
    null_count: int = 0
    null_pct: float = 0.0
    unique_count: int = 0
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    top_values: Optional[List[Dict[str, Any]]] = None
    min_time: Optional[str] = None
    max_time: Optional[str] = None
    time_format: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    timestamp_col: Optional[str] = None
    identifier_cols: List[str] = field(default_factory=list)
    metric_cols: List[str] = field(default_factory=list)
    format_detected: str = ""
    size_bytes: int = 0
    sampled: bool = False
    sample_rows: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [c.to_dict() for c in self.columns],
            "timestamp_col": self.timestamp_col,
            "identifier_cols": self.identifier_cols,
            "metric_cols": self.metric_cols,
            "format_detected": self.format_detected,
            "size_bytes": self.size_bytes,
            "sampled": self.sampled,
            "sample_rows": self.sample_rows,
        }


@dataclass
class TabularDataset:
    df: Any  # pd.DataFrame
    profile: DatasetProfile
    source: str = ""
    load_time_s: float = 0.0
    preprocessing_applied: List[str] = field(default_factory=list)

    def to_dict(self, include_data: bool = False, max_rows: int = 100) -> Dict[str, Any]:
        d = {
            "source": self.source,
            "profile": self.profile.to_dict(),
            "load_time_s": round(self.load_time_s, 3),
            "preprocessing_applied": self.preprocessing_applied,
        }
        if include_data:
            d["data"] = self.df.head(max_rows).to_dict(orient="records")
        return d
