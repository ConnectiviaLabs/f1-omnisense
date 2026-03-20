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


def load_race_summary_features(driver_code: str, year: int | None = None) -> pd.DataFrame:
    """Load per-race car telemetry features from telemetry_race_summary.

    Returns RPM, throttle, brake, DRS aggregates per (Year, Race) — data
    that fastf1_laps doesn't carry.  Merged into the anomaly pipeline to
    power the expanded 7-system grouping.
    """
    db = get_db()
    query: dict = {"Driver": driver_code.upper()}
    if year:
        query["Year"] = year

    projection = {
        "_id": 0, "Year": 1, "Race": 1,
        "avg_rpm": 1, "max_rpm": 1,
        "avg_throttle": 1, "brake_pct": 1,
        "drs_pct": 1, "avg_speed": 1, "top_speed": 1,
    }

    docs = list(db["telemetry_race_summary"].find(query, projection))
    if not docs:
        logger.info("No telemetry_race_summary data for %s", driver_code)
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    for col in ["avg_rpm", "max_rpm", "avg_throttle", "brake_pct", "drs_pct", "avg_speed", "top_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


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


MIN_RACES_THRESHOLD = 1


def get_grid_drivers(year: Optional[int] = None) -> list[dict]:
    """Get all drivers with sufficient historical data.

    Default (year=None): finds every unique driver in fastf1_laps with
    at least MIN_RACES_THRESHOLD races.  Enriches name/team from the
    latest jolpica standings or openf1_drivers when available.

    If year is set, returns that season's roster from openf1_drivers
    (legacy behaviour).

    Returns list of {code, name, team, number} dicts.
    """
    db = get_db()

    # ── All drivers from fastf1_laps meeting race threshold ────────
    if year is None:
        all_codes = db["fastf1_laps"].distinct(
            "Driver", {"IsAccurate": True, "SessionType": "R"}
        )

        # Build lookup for name/team from jolpica standings (latest season)
        standings_lookup: dict[str, dict] = {}
        for r in db["jolpica_driver_standings"].find(
            {}, {"driver_code": 1, "driver_name": 1, "constructor_name": 1, "season": 1, "_id": 0}
        ).sort("season", -1):
            code = r.get("driver_code")
            if code and code not in standings_lookup:
                standings_lookup[code] = {
                    "name": r.get("driver_name", code),
                    "team": r.get("constructor_name", "Unknown"),
                }

        # Build driver number lookup from openf1_drivers
        number_lookup: dict[str, int] = {}
        for r in db["openf1_drivers"].aggregate([
            {"$sort": {"session_key": -1}},
            {"$group": {"_id": "$name_acronym", "num": {"$first": "$driver_number"}}},
        ]):
            if r["_id"] and r.get("num"):
                number_lookup[r["_id"]] = r["num"]

        qualified = []
        for code in sorted(all_codes):
            if not code:
                continue
            race_count = len(db["fastf1_laps"].distinct(
                "Race",
                {"Driver": code, "IsAccurate": True, "SessionType": "R"},
            ))
            if race_count >= MIN_RACES_THRESHOLD:
                info = standings_lookup.get(code, {})
                # Fallback: get team from latest fastf1_laps entry
                if not info.get("team") or info["team"] == "Unknown":
                    doc = db["fastf1_laps"].find_one(
                        {"Driver": code, "IsAccurate": True},
                        {"Team": 1}, sort=[("Year", -1)]
                    )
                    info["team"] = doc.get("Team", "Unknown") if doc else "Unknown"
                qualified.append({
                    "code": code,
                    "name": info.get("name", code),
                    "team": info.get("team", "Unknown"),
                    "number": number_lookup.get(code, 0),
                })
            else:
                logger.info(
                    "Skipping %s (%d races < %d minimum)",
                    code, race_count, MIN_RACES_THRESHOLD,
                )

        return qualified

    # ── Legacy: single-season roster from openf1_drivers ──────────
    pipeline = [
        {"$match": {"Year": year}},
        {"$sort": {"session_key": -1}},
        {"$group": {
            "_id": "$name_acronym",
            "full_name": {"$first": "$full_name"},
            "team_name": {"$first": "$team_name"},
            "driver_number": {"$first": "$driver_number"},
        }},
        {"$sort": {"_id": 1}},
    ]

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

    # Lookup driver numbers from openf1_drivers
    num_lookup: dict[str, int] = {}
    for r in db["openf1_drivers"].aggregate([
        {"$sort": {"session_key": -1}},
        {"$group": {"_id": "$name_acronym", "num": {"$first": "$driver_number"}}},
    ]):
        if r["_id"] and r.get("num"):
            num_lookup[r["_id"]] = r["num"]

    return [
        {"code": d, "name": d, "team": teams.get(d, "Unknown"), "number": num_lookup.get(d, 0)}
        for d in sorted(drivers)
    ]
