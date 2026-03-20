"""AiM RaceStudio 3 data ingestion — RS3 CSV export, legacy CSVs, and binary.

Ingest paths:
    ingest_from_rs3_csv(csv_path) — reads a single RS3-exported CSV (preferred)
    ingest_from_csvs(csv_dir)     — reads 5 pre-exported CSVs (legacy)
    ingest_xrk_session(xrk_path)  — converts binary via xrk_to_csv, then ingests

Creates 4 MongoDB collections:
    aim_sessions     — one doc per session (metadata + lap list + summary)
    aim_telemetry    — one doc per telemetry sample (merged with track data)
    aim_gps_raw      — raw GPS data
    aim_lap_summary  — per-lap aggregated metrics (anomaly pipeline input)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────


def _get_db():
    from pipeline.updater._db import get_db
    return get_db()


def _build_session_id(meta: dict) -> str:
    """Deterministic session ID: aim_{date}_{time}_{driver_slug}."""
    driver_slug = re.sub(r"[^a-z0-9]+", "_", meta.get("driver", "unknown").lower()).strip("_")
    date_str = meta.get("date", "unknown").replace("-", "")
    time_str = meta.get("time", "000000").replace(":", "")[:6]
    return f"aim_{date_str}_{time_str}_{driver_slug}"


def _find_csvs(csv_dir: Path) -> dict[str, Path]:
    """Auto-detect the 5 CSV files from a directory by suffix pattern."""
    mapping = {}
    for f in csv_dir.glob("*.csv"):
        name = f.stem.lower()
        if name.endswith("_telemetry"):
            mapping["telemetry"] = f
        elif name.endswith("_track"):
            mapping["track"] = f
        elif name.endswith("_gps_raw"):
            mapping["gps_raw"] = f
        elif name.endswith("_laps"):
            mapping["laps"] = f
        elif name.endswith("_metadata"):
            mapping["metadata"] = f
    return mapping


def _merge_track_into_telemetry(telemetry_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
    """Merge track position data into telemetry on time_s (nearest, 0.1s tolerance)."""
    if "time_s" not in telemetry_df.columns or "time_s" not in track_df.columns:
        logger.warning("Missing time_s column — skipping track merge")
        return telemetry_df

    tel = telemetry_df.sort_values("time_s").reset_index(drop=True)
    trk = track_df.sort_values("time_s").reset_index(drop=True)

    track_cols = ["X_m", "Y_m", "Z_m", "Slope_deg", "LateralG_computed",
                  "LongG_computed", "Distance_m", "Speed_kmh", "Heading_deg",
                  "Vx_ms", "Vy_ms", "Vz_ms"]
    # Only keep columns that exist in track data
    available = [c for c in track_cols if c in trk.columns]
    trk_subset = trk[["time_s"] + available].copy()

    # Rename track Speed_kmh to Track_Speed_kmh to avoid collision with telemetry GPS_Speed_kmh
    if "Speed_kmh" in trk_subset.columns:
        trk_subset = trk_subset.rename(columns={"Speed_kmh": "Track_Speed_kmh"})

    merged = pd.merge_asof(
        tel, trk_subset,
        on="time_s",
        direction="nearest",
        tolerance=0.1,
    )
    return merged


def _compute_lap_summaries(merged_df: pd.DataFrame, laps_df: pd.DataFrame, session_id: str) -> list[dict]:
    """Per-lap aggregated metrics for the anomaly pipeline."""
    summaries = []

    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifiers/timestamps from aggregation
    exclude = {"timestamp_ms", "time_s", "lap", "sample"}
    agg_cols = [c for c in numeric_cols if c not in exclude]

    for _, lap_row in laps_df.iterrows():
        lap_num = int(lap_row["lap_number"])
        lap_data = merged_df[merged_df["lap"] == lap_num]

        if lap_data.empty:
            continue

        summary: dict = {
            "session_id": session_id,
            "lap_number": lap_num,
            "lap_time_s": float(lap_row["lap_time_s"]),
            "sample_count": len(lap_data),
        }

        for col in agg_cols:
            series = lap_data[col].dropna()
            if series.empty:
                continue
            summary[f"avg_{col}"] = round(float(series.mean()), 4)
            summary[f"max_{col}"] = round(float(series.max()), 4)
            summary[f"min_{col}"] = round(float(series.min()), 4)
            summary[f"std_{col}"] = round(float(series.std()), 4)

        # Brake application percentage
        if "FrontBrake_bar" in lap_data.columns:
            brake_series = lap_data["FrontBrake_bar"].dropna()
            if len(brake_series) > 0:
                summary["brake_application_pct"] = round(
                    float((brake_series > 1.0).sum() / len(brake_series) * 100), 2
                )

        # Distance covered in this lap
        if "Distance_m" in lap_data.columns:
            dist = lap_data["Distance_m"].dropna()
            if len(dist) > 1:
                summary["distance_m"] = round(float(dist.max() - dist.min()), 1)

        summaries.append(summary)

    return summaries


def _build_session_summary(merged_df: pd.DataFrame, laps_df: pd.DataFrame) -> dict:
    """Top-level session summary with avg/max for key channels."""
    summary: dict = {}
    key_channels = [
        "GPS_Speed_kmh", "Throttle_pct", "FrontBrake_bar", "RearBrake_bar",
        "OilTemp_C", "WaterTemp_C", "OilPressure_bar", "FuelPressure_bar",
        "Lambda", "AFR", "ExternalVoltage_V", "LateralAcc_g", "InlineAcc_g",
        "VerticalAcc_g", "YawRate_dps", "RollRate_dps", "PitchRate_dps",
    ]
    for col in key_channels:
        if col in merged_df.columns:
            series = merged_df[col].dropna()
            if not series.empty:
                summary[f"avg_{col}"] = round(float(series.mean()), 3)
                summary[f"max_{col}"] = round(float(series.max()), 3)

    # Alert counts
    alert_cols = ["OilTempAlert", "OilPressureAlert", "WaterTempAlert",
                  "WaterLowTempAlert", "LowBattery"]
    alert_count = 0
    for col in alert_cols:
        if col in merged_df.columns:
            alert_count += int(merged_df[col].dropna().sum())
    summary["alert_count"] = alert_count

    # Best lap
    if not laps_df.empty:
        summary["best_lap_time_s"] = round(float(laps_df["lap_time_s"].min()), 3)
        summary["best_lap_number"] = int(laps_df.loc[laps_df["lap_time_s"].idxmin(), "lap_number"])

    return summary


def _push_to_mongo(
    session_doc: dict,
    telemetry_docs: list[dict],
    gps_docs: list[dict],
    lap_summaries: list[dict],
) -> None:
    """Insert session + telemetry + GPS + lap summaries into MongoDB."""
    db = _get_db()
    session_id = session_doc["session_id"]

    # ── aim_sessions ──
    coll = db["aim_sessions"]
    coll.create_index("session_id", unique=True)
    coll.create_index("driver")
    coll.create_index("track")
    coll.replace_one({"session_id": session_id}, session_doc, upsert=True)
    logger.info("aim_sessions: upserted %s", session_id)

    # ── aim_telemetry — batch insert ──
    coll = db["aim_telemetry"]
    coll.create_index([("session_id", 1), ("timestamp_ms", 1)])
    coll.create_index([("session_id", 1), ("lap", 1)])
    coll.delete_many({"session_id": session_id})
    batch_size = 10_000
    for i in range(0, len(telemetry_docs), batch_size):
        coll.insert_many(telemetry_docs[i : i + batch_size])
    logger.info("aim_telemetry: inserted %d docs for %s", len(telemetry_docs), session_id)

    # ── aim_gps_raw ──
    if gps_docs:
        coll = db["aim_gps_raw"]
        coll.create_index([("session_id", 1), ("time_s", 1)])
        coll.delete_many({"session_id": session_id})
        for i in range(0, len(gps_docs), batch_size):
            coll.insert_many(gps_docs[i : i + batch_size])
        logger.info("aim_gps_raw: inserted %d docs for %s", len(gps_docs), session_id)

    # ── aim_lap_summary ──
    if lap_summaries:
        coll = db["aim_lap_summary"]
        coll.create_index([("session_id", 1), ("lap_number", 1)], unique=True)
        coll.create_index("driver")
        coll.delete_many({"session_id": session_id})
        coll.insert_many(lap_summaries)
        logger.info("aim_lap_summary: inserted %d docs for %s", len(lap_summaries), session_id)


# ── RS3 CSV column mapping ───────────────────────────────────────────────

_RS3_COL_MAP: dict[str, str] = {
    "Time": "time_s",
    "GPS Speed": "GPS_Speed_kmh",
    "GPS Latitude": "GPS_Lat",
    "GPS Longitude": "GPS_Lon",
    "GPS Altitude": "GPS_Altitude_m",
    "GPS Heading": "GPS_Heading_deg",
    "GPS Nsat": "GPS_Satellites",
    "GPS LatAcc": "GPS_LatAcc_g",
    "GPS LonAcc": "GPS_LonAcc_g",
    "GPS Slope": "GPS_Slope_deg",
    "GPS Gyro": "GPS_Gyro_dps",
    "GPS PosAccuracy": "GPS_PosAccuracy_mm",
    "GPS SpdAccuracy": "GPS_SpdAccuracy_kmh",
    "GPS Radius": "GPS_Radius_m",
    "front brake": "FrontBrake_bar",
    "Rear brake": "RearBrake_bar",
    "oil press": "OilPressure_bar",
    "fuel press": "FuelPressure_bar",
    "water temp": "WaterTemp_C",
    "Oil temp": "OilTemp_C",
    "Lambda Analog": "Lambda",
    "AFR": "AFR",
    "TPS analog": "TPS_raw",
    "External Voltage": "ExternalVoltage_V",
    "LoggerTemp": "LoggerTemp_C",
    "Luminosity": "Luminosity",
    "InlineAcc": "InlineAcc_g",
    "LateralAcc": "LateralAcc_g",
    "VerticalAcc": "VerticalAcc_g",
    "RollRate": "RollRate_dps",
    "PitchRate": "PitchRate_dps",
    "YawRate": "YawRate_dps",
    "Predictive Time": "PredictiveTime",
    "Ref Lap Diff": "RefLapDiff",
    "Bias w Thrs": "BiasWithThrottle_pct",
    "SV presion aceite": "SV_OilPressure",
    "SV pres gasolina": "SV_FuelPressure",
    "SV temp agua": "SV_WaterTemp",
    "SDS RPM": "RPM",
    "SDS TPS": "Throttle_pct",
    "SDS GEAR": "Gear",
    "SDS BATT VOLT": "BatteryVoltage_V",
    "Distance on GPS Speed": "Distance_m",
}


def _parse_rs3_header(lines: list[str]) -> dict:
    """Parse the 13-line RS3 CSV metadata header.

    Returns dict with keys: driver, vehicle, championship, date, time,
    sample_rate, duration_s, beacon_markers, segment_times.
    """
    import csv as csv_mod
    from io import StringIO

    def _parse_row(line: str) -> list[str]:
        return list(csv_mod.reader(StringIO(line)))[0]

    meta: dict = {}
    for line in lines[:13]:
        row = _parse_row(line)
        if not row:
            continue
        key = row[0].strip().lower()
        if key == "racer":
            meta["driver"] = row[1].strip() if len(row) > 1 else ""
        elif key == "vehicle":
            meta["vehicle"] = row[1].strip() if len(row) > 1 else ""
        elif key == "championship":
            meta["championship"] = row[1].strip() if len(row) > 1 else ""
        elif key == "session":
            meta["session_name"] = row[1].strip() if len(row) > 1 else ""
        elif key == "date":
            meta["date"] = row[1].strip() if len(row) > 1 else ""
        elif key == "time":
            meta["time"] = row[1].strip() if len(row) > 1 else ""
        elif key == "sample rate":
            meta["sample_rate"] = int(row[1]) if len(row) > 1 else 20
        elif key == "duration":
            meta["duration_s"] = float(row[1]) if len(row) > 1 else 0.0
        elif key == "beacon markers":
            meta["beacon_markers"] = [float(v) for v in row[1:] if v.strip()]
        elif key == "segment times":
            meta["segment_times"] = row[1:]

    return meta


def _parse_segment_time(s: str) -> float:
    """Convert RS3 segment time like '1:57.580' or '1:08.948' to seconds."""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        return float(parts[0]) * 60 + float(parts[1])
    return float(s)


def _parse_rs3_date(date_str: str) -> str:
    """Convert RS3 date 'Tuesday, October 21, 2025' → '2025-10-21'."""
    for fmt in ("%A, %B %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str.strip()


def _parse_rs3_time(time_str: str) -> str:
    """Convert RS3 time '1:00 PM' → '130000'."""
    for fmt in ("%I:%M %p", "%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(time_str.strip(), fmt).strftime("%H%M%S")
        except ValueError:
            continue
    return "000000"


# ── Public API ───────────────────────────────────────────────────────────


def ingest_from_rs3_csv(csv_path: str | Path) -> str:
    """Ingest a single RS3-exported CSV (preferred path). Returns session_id."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"RS3 CSV not found: {csv_path}")

    logger.info("Loading RS3 CSV from %s ...", csv_path)

    # ── Read raw lines for header parsing ──
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        raw_lines = f.readlines()

    meta = _parse_rs3_header(raw_lines)
    logger.info(
        "RS3 header: driver=%s, date=%s, sample_rate=%s, duration=%s",
        meta.get("driver"), meta.get("date"),
        meta.get("sample_rate"), meta.get("duration_s"),
    )

    # ── Read data table (skip header rows) ──
    # Row 15 = column names (0-indexed line 14), row 16 = units, row 17 = blank, row 18+ = data
    df = pd.read_csv(csv_path, skiprows=14, header=0)
    # Drop the units row (first data row) and any blank rows
    # The units row has non-numeric values in numeric columns
    # Find first row where Time can be parsed as float
    first_data_idx = None
    for idx in df.index:
        try:
            float(df.loc[idx, df.columns[0]])
            first_data_idx = idx
            break
        except (ValueError, TypeError):
            continue

    if first_data_idx is None:
        raise ValueError("No valid data rows found in RS3 CSV")

    df = df.loc[first_data_idx:].reset_index(drop=True)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Rename columns ──
    rename_map = {}
    for orig_col in df.columns:
        stripped = orig_col.strip()
        if stripped in _RS3_COL_MAP:
            rename_map[orig_col] = _RS3_COL_MAP[stripped]
        else:
            # Keep as-is but sanitize for Mongo (no dots)
            rename_map[orig_col] = stripped.replace(".", "_").replace(" ", "_")
    df = df.rename(columns=rename_map)

    logger.info("RS3 data: %d rows, %d channels", len(df), len(df.columns))

    # ── Assign lap numbers from beacon markers ──
    beacon_markers = meta.get("beacon_markers", [])
    if beacon_markers:
        boundaries = [0.0] + beacon_markers
        df["lap"] = 0
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            mask = (df["time_s"] >= start) & (df["time_s"] < end)
            df.loc[mask, "lap"] = i + 1
        # Last marker edge: assign rows exactly at last boundary to last lap
        df.loc[df["lap"] == 0, "lap"] = len(boundaries) - 1
    else:
        df["lap"] = 1

    # ── Build laps DataFrame ──
    segment_times_raw = meta.get("segment_times", [])
    if segment_times_raw:
        lap_records = []
        for i, st in enumerate(segment_times_raw):
            try:
                lap_records.append({
                    "lap_number": i + 1,
                    "lap_time_s": round(_parse_segment_time(st), 3),
                })
            except (ValueError, IndexError):
                continue
        laps_df = pd.DataFrame(lap_records)
    else:
        laps_df = pd.DataFrame(columns=["lap_number", "lap_time_s"])

    # ── Add timestamp_ms column (used by downstream) ──
    if "time_s" in df.columns:
        df["timestamp_ms"] = (df["time_s"] * 1000).round().astype(int)

    # ── Session ID ──
    parsed_date = _parse_rs3_date(meta.get("date", ""))
    parsed_time = _parse_rs3_time(meta.get("time", ""))
    session_id = _build_session_id({
        "driver": meta.get("driver", "unknown"),
        "date": parsed_date,
        "time": parsed_time,
    })

    # ── Compute summaries ──
    lap_summaries = _compute_lap_summaries(df, laps_df, session_id)
    session_summary = _build_session_summary(df, laps_df)

    # ── Build session doc ──
    session_doc = {
        "session_id": session_id,
        "driver": meta.get("driver", ""),
        "track": meta.get("session_name", ""),
        "date": parsed_date,
        "time": meta.get("time", ""),
        "vehicle": meta.get("vehicle", ""),
        "championship": meta.get("championship", ""),
        "sample_rate_hz": meta.get("sample_rate", 20),
        "lap_count": len(laps_df),
        "duration_s": round(meta.get("duration_s", 0.0), 3),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "source": "rs3_csv",
        "source_file": csv_path.name,
        "summary": session_summary,
        "laps": [
            {"lap_number": int(r["lap_number"]), "lap_time_s": round(float(r["lap_time_s"]), 3)}
            for _, r in laps_df.iterrows()
        ],
    }

    # ── Build telemetry docs (drop NaN) ──
    telemetry_docs = []
    for row in df.to_dict(orient="records"):
        doc = {"session_id": session_id}
        for k, v in row.items():
            if pd.notna(v):
                doc[k] = v
        telemetry_docs.append(doc)

    # ── Build GPS docs ──
    gps_cols = [c for c in df.columns if c.startswith("GPS_")]
    gps_docs = []
    if gps_cols and "time_s" in df.columns:
        extra = [c for c in ["lap"] if c in df.columns and c not in gps_cols]
        gps_subset = df[["time_s"] + extra + gps_cols].dropna(subset=["GPS_Lat", "GPS_Lon"], how="any")
        for row in gps_subset.to_dict(orient="records"):
            doc = {"session_id": session_id}
            for k, v in row.items():
                if pd.notna(v):
                    doc[k] = v
            gps_docs.append(doc)

    # ── Add driver to lap summaries ──
    for ls in lap_summaries:
        ls["driver"] = meta.get("driver", "")

    # ── Push to MongoDB ──
    _push_to_mongo(session_doc, telemetry_docs, gps_docs, lap_summaries)

    logger.info("RS3 ingestion complete: session_id=%s (%d telemetry docs)", session_id, len(telemetry_docs))
    return session_id


def ingest_from_csvs(csv_dir: str | Path) -> str:
    """Primary ingest path — reads 5 pre-exported CSVs, pushes to MongoDB.

    Returns: session_id
    """
    csv_dir = Path(csv_dir)
    files = _find_csvs(csv_dir)

    required = {"telemetry", "metadata", "laps"}
    missing = required - files.keys()
    if missing:
        raise FileNotFoundError(f"Missing CSVs in {csv_dir}: {missing}")

    logger.info("Loading CSVs from %s ...", csv_dir)

    # ── Read CSVs ──
    meta_df = pd.read_csv(files["metadata"])
    meta = meta_df.iloc[0].to_dict()

    telemetry_df = pd.read_csv(files["telemetry"])
    laps_df = pd.read_csv(files["laps"])

    track_df = pd.read_csv(files["track"]) if "track" in files else pd.DataFrame()
    gps_df = pd.read_csv(files["gps_raw"]) if "gps_raw" in files else pd.DataFrame()

    logger.info(
        "Loaded: telemetry=%d rows, track=%d rows, gps=%d rows, laps=%d",
        len(telemetry_df), len(track_df), len(gps_df), len(laps_df),
    )

    # ── Session ID ──
    session_id = _build_session_id(meta)

    # ── Merge track into telemetry ──
    if not track_df.empty:
        merged_df = _merge_track_into_telemetry(telemetry_df, track_df)
    else:
        merged_df = telemetry_df

    # ── Compute summaries ──
    lap_summaries = _compute_lap_summaries(merged_df, laps_df, session_id)
    session_summary = _build_session_summary(merged_df, laps_df)

    # ── Build session doc ──
    session_doc = {
        "session_id": session_id,
        "driver": meta.get("driver", ""),
        "track": meta.get("track", ""),
        "date": meta.get("date", ""),
        "time": meta.get("time", ""),
        "vehicle": meta.get("vehicle", ""),
        "ecu": meta.get("ecu", ""),
        "firmware": meta.get("firmware", ""),
        "session_file": meta.get("session_file", ""),
        "lap_count": len(laps_df),
        "duration_s": round(float(laps_df["lap_time_s"].sum()), 3) if not laps_df.empty else 0,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "summary": session_summary,
        "laps": [
            {"lap_number": int(r["lap_number"]), "lap_time_s": round(float(r["lap_time_s"]), 3)}
            for _, r in laps_df.iterrows()
        ],
    }

    # ── Build telemetry docs ──
    # Drop NaN values — they aren't valid JSON and waste space in MongoDB
    telemetry_docs = []
    for row in merged_df.to_dict(orient="records"):
        doc = {"session_id": session_id}
        for k, v in row.items():
            if pd.notna(v):
                doc[k] = v
        telemetry_docs.append(doc)

    # ── Build GPS docs ──
    gps_docs = []
    if not gps_df.empty:
        for row in gps_df.to_dict(orient="records"):
            doc = {"session_id": session_id}
            for k, v in row.items():
                if pd.notna(v):
                    doc[k] = v
            gps_docs.append(doc)

    # Add driver to lap summaries
    for ls in lap_summaries:
        ls["driver"] = meta.get("driver", "")

    # ── Push ──
    _push_to_mongo(session_doc, telemetry_docs, gps_docs, lap_summaries)

    logger.info("Ingestion complete: session_id=%s", session_id)
    return session_id


def ingest_xrk_session(
    xrk_path: str | Path,
    gpk_path: Optional[str | Path] = None,
    rrk_path: Optional[str | Path] = None,
    drk_path: Optional[str | Path] = None,
) -> str:
    """Convert binary AiM files to CSVs, then ingest.

    Returns: session_id
    """
    import tempfile
    from pipeline.xrk_to_csv import convert_xrk_to_csv

    xrk_path = Path(xrk_path)
    if not xrk_path.exists():
        raise FileNotFoundError(f"XRK file not found: {xrk_path}")

    with tempfile.TemporaryDirectory(prefix="aim_ingest_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        convert_xrk_to_csv(str(xrk_path), str(tmpdir_path))

        # TODO: parse GPK/RRK/DRK when parsers are available
        if gpk_path:
            logger.info("GPK parsing not yet implemented: %s", gpk_path)
        if rrk_path:
            logger.info("RRK parsing not yet implemented: %s", rrk_path)
        if drk_path:
            logger.info("DRK parsing not yet implemented: %s", drk_path)

        return ingest_from_csvs(tmpdir_path)


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    target = sys.argv[1] if len(sys.argv) > 1 else "/home/pedrad/Desktop/f1/RealRacingData/1.csv"
    target_path = Path(target)

    if target_path.is_file() and target_path.suffix.lower() == ".csv":
        sid = ingest_from_rs3_csv(target_path)
    elif target_path.is_dir():
        sid = ingest_from_csvs(target_path)
    else:
        print(f"Error: {target} is not a CSV file or directory")
        sys.exit(1)

    print(f"Done — session_id: {sid}")
