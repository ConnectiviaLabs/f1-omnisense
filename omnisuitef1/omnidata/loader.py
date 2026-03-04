"""Multi-format tabular data loader with auto-detection and adaptive sampling."""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from omnidata._types import DatasetProfile, TabularDataset

logger = logging.getLogger(__name__)

SMALL_DATASET_MB = 50
SAMPLE_ROWS = 10_000

FORMAT_MAP = {
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def detect_format(source: Union[str, Path, bytes], filename: str = "") -> str:
    """Auto-detect tabular format from extension or content sniffing."""
    name = filename
    if not name and isinstance(source, (str, Path)):
        name = str(source)

    ext = Path(name).suffix.lower() if name else ""
    if ext in FORMAT_MAP:
        return FORMAT_MAP[ext]

    # Content sniffing for bytes/stream
    if isinstance(source, bytes):
        head = source[:32]
        if head[:4] == b"PAR1":
            return "parquet"
        if head[:1] in (b"[", b"{"):
            return "json"
        return "csv"

    return "csv"


def load(
    source: Union[str, Path, bytes, io.BytesIO],
    *,
    filename: str = "",
    format: Optional[str] = None,
    sheet_name: Optional[Union[str, int]] = 0,
    sample: Optional[int] = None,
    encoding: str = "utf-8",
) -> TabularDataset:
    """Load tabular data from file path, bytes, or stream.

    Auto-detects format (CSV, JSON, Excel, Parquet).
    Adaptive strategy: full load for small files, sampled for large.
    """
    t0 = time.time()
    fmt = format or detect_format(source, filename)

    # Determine file size for adaptive loading
    size_bytes = 0
    if isinstance(source, (str, Path)) and os.path.isfile(source):
        size_bytes = os.path.getsize(source)

    # Adaptive sampling
    use_sample = sample
    sampled = False
    if use_sample is None and size_bytes > SMALL_DATASET_MB * 1024 * 1024:
        use_sample = SAMPLE_ROWS
        sampled = True

    if fmt == "csv":
        df = _load_csv(source, encoding, use_sample)
    elif fmt == "json":
        df = _load_json(source, use_sample)
    elif fmt == "excel":
        df = _load_excel(source, sheet_name, use_sample)
    elif fmt == "parquet":
        df = _load_parquet(source, use_sample)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # Build shallow profile
    profile = DatasetProfile(
        row_count=len(df),
        column_count=len(df.columns),
        columns=[],
        format_detected=fmt,
        size_bytes=size_bytes,
        sampled=sampled,
        sample_rows=use_sample if sampled else None,
    )

    src_name = filename or (str(source) if isinstance(source, (str, Path)) else "<stream>")

    return TabularDataset(
        df=df,
        profile=profile,
        source=src_name,
        load_time_s=round(time.time() - t0, 3),
    )


def _load_csv(source, encoding: str, sample: Optional[int]) -> pd.DataFrame:
    kwargs = {"encoding": encoding, "low_memory": False}
    if sample:
        kwargs["nrows"] = sample

    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source), **kwargs)
    if isinstance(source, io.BytesIO):
        return pd.read_csv(source, **kwargs)
    return pd.read_csv(source, **kwargs)


def _load_json(source, sample: Optional[int]) -> pd.DataFrame:
    if isinstance(source, bytes):
        buf = io.BytesIO(source)
    elif isinstance(source, io.BytesIO):
        buf = source
    else:
        buf = source

    # Try JSON array first, then newline-delimited
    try:
        df = pd.read_json(buf, orient="records")
    except ValueError:
        if isinstance(buf, io.BytesIO):
            buf.seek(0)
        df = pd.read_json(buf, lines=True)

    if sample and len(df) > sample:
        df = df.head(sample)
    return df


def _load_excel(source, sheet_name, sample: Optional[int]) -> pd.DataFrame:
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    kwargs = {}
    if sheet_name is not None:
        kwargs["sheet_name"] = sheet_name
    if sample:
        kwargs["nrows"] = sample

    return pd.read_excel(source, **kwargs)


def _load_parquet(source, sample: Optional[int]) -> pd.DataFrame:
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    df = pd.read_parquet(source)
    if sample and len(df) > sample:
        df = df.head(sample)
    return df
