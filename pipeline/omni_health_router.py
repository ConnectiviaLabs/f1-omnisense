"""OmniHealth APIRouter — predictive maintenance for F1 Fleet Overview.

Endpoints:
    GET /api/omni/health/assess/{driver_code}  — full HealthReport for one driver
    GET /api/omni/health/fleet                 — fleet-wide health for all drivers
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from pipeline.anomaly.mongo_loader import (
    load_driver_race_telemetry,
    get_grid_drivers,
)
from omnihealth import assess

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/omni/health", tags=["OmniHealth"])

# Component map using fastf1_laps aggregated column names
COMPONENT_MAP = {
    "Speed": ["SpeedI1_mean", "SpeedI2_mean", "SpeedFL_mean", "SpeedST_mean",
              "SpeedI1_max", "SpeedST_max"],
    "Lap Pace": ["LapTime_mean", "LapTime_std",
                 "Sector1Time_mean", "Sector2Time_mean", "Sector3Time_mean"],
    "Tyre Management": ["TyreLife_mean", "TyreLife_max", "TyreLife_std",
                        "stint_count", "compound_variety"],
}


def _load_telemetry(driver_code: str):
    """Load race telemetry for a driver from MongoDB."""
    df = load_driver_race_telemetry(driver_code)
    if df.empty:
        raise HTTPException(404, f"No telemetry data for driver {driver_code}")
    return df


def _filter_component_map(df):
    """Only include columns that actually exist in the DataFrame."""
    return {
        system: [c for c in cols if c in df.columns]
        for system, cols in COMPONENT_MAP.items()
        if any(c in df.columns for c in cols)
    }


@router.get("/assess/{driver_code}")
def assess_driver(
    driver_code: str,
    horizon: int = Query(10, ge=1, le=50),
    forecast_method: str = Query("auto"),
):
    """Run omnihealth.assess() on a driver's aggregated race telemetry."""
    driver_code = driver_code.upper()
    merged = _load_telemetry(driver_code)
    cmap = _filter_component_map(merged)

    report = assess(
        merged, cmap,
        horizon=horizon,
        forecast_method=forecast_method,
    )
    return {
        "driver": driver_code,
        "code": driver_code,
        "races": len(merged),
        **report.to_dict(),
    }


@router.get("/fleet")
def fleet_health(
    year: int = Query(2024),
    horizon: int = Query(10, ge=1, le=50),
):
    """Run health assessment for all grid drivers."""
    grid = get_grid_drivers(year)
    results = []
    for driver_info in grid:
        code = driver_info["code"]
        try:
            merged = _load_telemetry(code)
            cmap = _filter_component_map(merged)
            report = assess(merged, cmap, horizon=horizon)
            results.append({
                "driver": driver_info["name"],
                "code": code,
                "team": driver_info["team"],
                "number": driver_info["number"],
                "races": len(merged),
                **report.to_dict(),
            })
        except Exception as e:
            logger.warning(f"OmniHealth assess failed for {code}: {e}")
            results.append({
                "driver": driver_info["name"],
                "code": code,
                "team": driver_info["team"],
                "error": str(e),
            })
    return {"drivers": results, "year": year}
