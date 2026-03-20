"""OmniAnalytics APIRouter — anomaly detection and forecasting for all F1 drivers.

Endpoints:
    POST /api/omni/analytics/anomaly/{driver_code}   — run anomaly ensemble
    POST /api/omni/analytics/forecast/{driver_code}   — forecast a telemetry column
    GET  /api/omni/analytics/grid                     — anomaly summary for all grid drivers
    GET  /api/omni/analytics/team/{team}              — anomaly summary for a team
    GET  /api/omni/analytics/rivals/{driver_code}     — compare a driver vs the field
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from pipeline.anomaly.mongo_loader import (
    load_driver_race_telemetry,
    get_grid_drivers,
)
from omnidata._types import TabularDataset, DatasetProfile, ColumnProfile, ColumnRole, DType
from omnianalytics import AnomalyEnsemble, forecast

import math

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/omni/analytics", tags=["OmniAnalytics"])

# ── Dataset cache (TTL 5 min) ────────────────────────────────────────────
_DS_CACHE: Dict[str, Tuple[TabularDataset, float]] = {}
_DS_CACHE_TTL = 300  # seconds


def _sanitize(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _build_dataset(driver_code: str) -> TabularDataset:
    """Load race telemetry from MongoDB and wrap in a TabularDataset.

    Results are cached for 5 minutes to avoid redundant MongoDB loads
    when forecasting multiple columns for the same driver.
    """
    now = time.time()
    cached = _DS_CACHE.get(driver_code)
    if cached and (now - cached[1]) < _DS_CACHE_TTL:
        return cached[0]

    df = load_driver_race_telemetry(driver_code)
    if df.empty:
        raise HTTPException(404, f"No telemetry data for driver {driver_code}")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    col_profiles = [
        ColumnProfile(
            name=c,
            dtype=DType.FLOAT,
            role=ColumnRole.METRIC,
            null_count=int(df[c].isna().sum()),
            unique_count=int(df[c].nunique()),
        )
        for c in numeric_cols
    ]
    profile = DatasetProfile(
        row_count=len(df),
        column_count=len(numeric_cols),
        columns=col_profiles,
        metric_cols=numeric_cols,
        timestamp_col=None,
    )
    ds = TabularDataset(df=df, profile=profile)
    _DS_CACHE[driver_code] = (ds, time.time())
    return ds


def _quick_anomaly_summary(driver_code: str) -> dict | None:
    """Run a fast anomaly check and return summary dict, or None if no data."""
    try:
        ds = _build_dataset(driver_code)
        ensemble = AnomalyEnsemble()
        result = ensemble.run(ds)
    except (HTTPException, ValueError, Exception) as e:
        logger.debug("Skipping %s: %s", driver_code, e)
        return None

    return {
        "code": driver_code,
        "total_rows": result.total_rows,
        "anomaly_count": result.anomaly_count,
        "anomaly_pct": round(result.anomaly_count / max(result.total_rows, 1) * 100, 1),
        "severity_distribution": result.severity_distribution,
        "threshold": result.threshold,
    }


class AnomalyRequest(BaseModel):
    columns: Optional[List[str]] = None


@router.post("/anomaly/{driver_code}")
def detect_anomalies(driver_code: str, body: AnomalyRequest = AnomalyRequest()):
    """Run the OmniAnalytics anomaly ensemble on any driver's race telemetry."""
    driver_code = driver_code.upper()
    ds = _build_dataset(driver_code)
    ensemble = AnomalyEnsemble()
    result = ensemble.run(ds, columns=body.columns)
    return _sanitize({
        "driver": driver_code,
        "code": driver_code,
        **result.to_dict(),
    })


@router.post("/forecast/{driver_code}")
def forecast_column(
    driver_code: str,
    column: str = Query(..., description="Telemetry column to forecast"),
    horizon: int = Query(5, ge=1, le=50),
    method: str = Query("auto", description="auto, arima, linear, or lightgbm"),
):
    """Forecast a specific telemetry column for any driver."""
    driver_code = driver_code.upper()
    ds = _build_dataset(driver_code)
    if column not in ds.df.columns:
        raise HTTPException(400, f"Column '{column}' not found. Available: {list(ds.df.columns)}")

    result = forecast(ds, column=column, horizon=horizon, method=method)
    return _sanitize({
        "driver": driver_code,
        "code": driver_code,
        **result.to_dict(),
    })


@router.get("/grid")
def grid_anomaly_summary(year: int = Query(2024)):
    """Anomaly summary for all drivers on the grid."""
    grid = get_grid_drivers(year)
    results = []
    for driver_info in grid:
        summary = _quick_anomaly_summary(driver_info["code"])
        if summary:
            summary["name"] = driver_info["name"]
            summary["team"] = driver_info["team"]
            summary["number"] = driver_info["number"]
            results.append(summary)

    results.sort(key=lambda x: x["anomaly_pct"], reverse=True)
    return _sanitize({"year": year, "drivers": results, "count": len(results)})


@router.get("/team/{team}")
def team_anomaly_summary(team: str, year: int = Query(2024)):
    """Anomaly summary for all drivers in a specific team."""
    grid = get_grid_drivers(year)
    team_drivers = [d for d in grid if team.lower() in d["team"].lower()]

    if not team_drivers:
        raise HTTPException(404, f"No drivers found for team '{team}' in {year}")

    results = []
    for driver_info in team_drivers:
        summary = _quick_anomaly_summary(driver_info["code"])
        if summary:
            summary["name"] = driver_info["name"]
            summary["team"] = driver_info["team"]
            summary["number"] = driver_info["number"]
            results.append(summary)

    return _sanitize({"team": team, "year": year, "drivers": results, "count": len(results)})


@router.post("/forecast/team/{team}")
def forecast_team(
    team: str,
    column: str = Query(..., description="Telemetry column to forecast"),
    horizon: int = Query(5, ge=1, le=50),
    method: str = Query("auto"),
    year: int = Query(2024),
):
    """Forecast a telemetry column averaged across all drivers in a team."""
    grid = get_grid_drivers(year)
    team_drivers = [d for d in grid if team.lower() in d["team"].lower()]
    if not team_drivers:
        raise HTTPException(404, f"No drivers found for team '{team}' in {year}")

    driver_dfs = []
    driver_codes_used = []
    for d in team_drivers:
        try:
            df = load_driver_race_telemetry(d["code"])
            if not df.empty and column in df.columns:
                driver_dfs.append(df)
                driver_codes_used.append(d["code"])
        except Exception:
            continue

    if not driver_dfs:
        raise HTTPException(404, f"No data for column '{column}' in team '{team}'")

    # Average across drivers race-by-race (align by row index)
    combined = pd.DataFrame()
    for df in driver_dfs:
        # Reindex to align race rows
        vals = df[column].reset_index(drop=True)
        combined = pd.concat([combined, vals.rename(len(combined.columns))], axis=1)

    # Team average per race
    team_series = combined.mean(axis=1).dropna()
    if len(team_series) < 3:
        raise HTTPException(400, f"Not enough data points for team forecast (got {len(team_series)})")

    # Build a minimal dataset for the forecaster
    team_df = pd.DataFrame({column: team_series.values})
    numeric_cols = [column]
    col_profiles = [
        ColumnProfile(
            name=column,
            dtype=DType.FLOAT,
            role=ColumnRole.METRIC,
            null_count=0,
            unique_count=int(team_series.nunique()),
        )
    ]
    profile = DatasetProfile(
        row_count=len(team_df),
        column_count=1,
        columns=col_profiles,
        metric_cols=numeric_cols,
        timestamp_col=None,
    )
    ds = TabularDataset(df=team_df, profile=profile)
    result = forecast(ds, column=column, horizon=horizon, method=method)

    return _sanitize({
        "team": team,
        "drivers": driver_codes_used,
        "column": result.column,
        "method": result.method,
        "values": list(result.values),
        "lower_bound": list(result.lower_bound) if result.lower_bound is not None else None,
        "upper_bound": list(result.upper_bound) if result.upper_bound is not None else None,
        "mae": result.mae,
        "rmse": result.rmse,
        "trend_direction": result.trend_direction,
        "trend_pct": result.trend_pct,
        "volatility": getattr(result, "volatility", None),
        "risk_flag": getattr(result, "risk_flag", None),
        "history": list(team_series.values),
        "timestamps": list(result.timestamps) if hasattr(result, "timestamps") and result.timestamps is not None else None,
    })


@router.get("/rivals/{driver_code}")
def rivals_comparison(driver_code: str, year: int = Query(2024)):
    """Compare a driver's anomaly profile against the rest of the field."""
    driver_code = driver_code.upper()

    grid = get_grid_drivers(year)
    target_info = next((d for d in grid if d["code"] == driver_code), None)

    target_summary = _quick_anomaly_summary(driver_code)
    if not target_summary:
        raise HTTPException(404, f"No data for driver {driver_code}")

    if target_info:
        target_summary["name"] = target_info["name"]
        target_summary["team"] = target_info["team"]

    # Run for all other drivers
    rivals = []
    for driver_info in grid:
        if driver_info["code"] == driver_code:
            continue
        summary = _quick_anomaly_summary(driver_info["code"])
        if summary:
            summary["name"] = driver_info["name"]
            summary["team"] = driver_info["team"]
            rivals.append(summary)

    rivals.sort(key=lambda x: x["anomaly_pct"], reverse=True)

    return _sanitize({
        "target": target_summary,
        "rivals": rivals,
        "year": year,
    })
