"""MongoDB-based telemetry loader for all drivers.

Replaces the CSV-based loader that only worked for McLaren (NOR/PIA).
Pulls from telemetry_lap_summary and fastf1_laps collections which
contain data for all 70+ drivers across 2018-2025.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from updater._db import get_db

logger = logging.getLogger(__name__)


def load_driver_race_telemetry(driver_code: str, year: Optional[int] = None) -> pd.DataFrame:
    """Load per-race aggregated telemetry for any driver from MongoDB.

    Queries fastf1_laps (accurate race laps only), aggregates per (Year, Race)
    to produce one row per race with mean/max/std of speed traps, sector times,
    lap times, tyre life, and stint data.

    Returns a DataFrame compatible with the anomaly ensemble pipeline.
    """
    db = get_db()
    query: dict = {"Driver": driver_code.upper(), "IsAccurate": True, "SessionType": "R"}
    if year:
        query["Year"] = year

    projection = {
        "_id": 0, "Year": 1, "Race": 1,
        "LapTime": 1, "Sector1Time": 1, "Sector2Time": 1, "Sector3Time": 1,
        "SpeedI1": 1, "SpeedI2": 1, "SpeedFL": 1, "SpeedST": 1,
        "TyreLife": 1, "Stint": 1, "Compound": 1,
    }

    cursor = db["fastf1_laps"].find(query, projection)
    df = pd.DataFrame(list(cursor))

    if df.empty:
        logger.warning("No fastf1_laps race data for %s", driver_code)
        return pd.DataFrame()

    # Ensure numeric
    metric_cols = [
        "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "TyreLife",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Stint" in df.columns:
        df["Stint"] = pd.to_numeric(df["Stint"], errors="coerce")

    available = [c for c in metric_cols if c in df.columns and df[c].notna().any()]

    # Aggregate per race
    agg_funcs = {}
    for col in available:
        agg_funcs[f"{col}_mean"] = (col, "mean")
        agg_funcs[f"{col}_max"] = (col, "max")
        agg_funcs[f"{col}_std"] = (col, "std")

    if "Stint" in df.columns:
        agg_funcs["stint_count"] = ("Stint", "nunique")

    if "Compound" in df.columns:
        agg_funcs["compound_variety"] = ("Compound", "nunique")

    agg_funcs["samples"] = (available[0], "count")

    result = df.groupby(["Year", "Race"]).agg(**agg_funcs).reset_index()
    result["driver"] = driver_code.upper()
    result["race"] = result["Year"].astype(str) + " " + result["Race"]

    # Fill NaN std (single-lap races) with 0
    std_cols = [c for c in result.columns if c.endswith("_std")]
    result[std_cols] = result[std_cols].fillna(0)

    return result


def load_driver_lap_data(
    driver_code: str,
    year: Optional[int] = None,
    session_type: str = "Race",
) -> pd.DataFrame:
    """Load granular lap-level data for a driver from fastf1_laps.

    Returns sector times, speed traps, compound, tyre life per lap.
    Used for finer-grained anomaly detection.
    """
    db = get_db()
    query: dict = {"Driver": driver_code.upper(), "IsAccurate": True}
    if year:
        query["Year"] = year
    if session_type:
        query["SessionType"] = session_type

    projection = {
        "_id": 0, "Driver": 1, "Year": 1, "Race": 1, "LapNumber": 1,
        "LapTime": 1, "Sector1Time": 1, "Sector2Time": 1, "Sector3Time": 1,
        "SpeedI1": 1, "SpeedI2": 1, "SpeedFL": 1, "SpeedST": 1,
        "Compound": 1, "TyreLife": 1, "Stint": 1, "Team": 1, "Position": 1,
    }

    cursor = db["fastf1_laps"].find(query, projection)
    df = pd.DataFrame(list(cursor))

    if df.empty:
        logger.warning("No fastf1_laps data for %s (year=%s)", driver_code, year)
        return pd.DataFrame()

    # Convert time fields to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure numeric speed traps
    for col in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "TyreLife", "LapNumber"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_grid_drivers(year: int = 2024) -> list[dict]:
    """Get all drivers from the most recent season in openf1_drivers.

    Returns list of {code, name, team, number} dicts.
    """
    db = get_db()

    pipeline = [
        {"$match": {"Year": year} if year else {}},
        {"$sort": {"session_key": -1}},
        {"$group": {
            "_id": "$name_acronym",
            "full_name": {"$first": "$full_name"},
            "team_name": {"$first": "$team_name"},
            "driver_number": {"$first": "$driver_number"},
        }},
        {"$sort": {"_id": 1}},
    ]

    # Try openf1_drivers first
    results = list(db["openf1_drivers"].aggregate(pipeline))
    if results:
        return [
            {
                "code": r["_id"],
                "name": r.get("full_name", r["_id"]),
                "team": r.get("team_name", "Unknown"),
                "number": r.get("driver_number", 0),
            }
            for r in results if r["_id"]
        ]

    # Fallback: get distinct drivers from fastf1_laps
    drivers = db["fastf1_laps"].distinct("Driver", {"Year": year, "IsAccurate": True})
    teams = {}
    for d in drivers:
        doc = db["fastf1_laps"].find_one({"Driver": d, "Year": year}, {"Team": 1})
        teams[d] = doc.get("Team", "Unknown") if doc else "Unknown"

    return [
        {"code": d, "name": d, "team": teams.get(d, "Unknown"), "number": 0}
        for d in sorted(drivers)
    ]
