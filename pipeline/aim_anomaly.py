"""AiM RaceStudio 3 — anomaly detection & health scoring.

Maps AiM telemetry channels to the 7 OmniHealth systems, then runs
the existing AnomalyEnsemble + health scoring pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── AiM channel → OmniHealth system mapping ─────────────────────────────

AIM_SYSTEM_FEATURES: dict[str, list[str]] = {
    "Power Unit": [
        "avg_Throttle_pct", "avg_AFR", "avg_Lambda", "avg_TPS_raw",
        "max_Throttle_pct", "max_AFR",
    ],
    "Brakes": [
        "avg_FrontBrake_bar", "max_FrontBrake_bar",
        "avg_RearBrake_bar", "max_RearBrake_bar",
        "brake_application_pct",
    ],
    "Drivetrain": [
        "avg_GPS_Speed_kmh", "max_GPS_Speed_kmh",
        "avg_InlineAcc_g", "max_InlineAcc_g",
        "distance_m",
    ],
    "Suspension": [
        "avg_LateralAcc_g", "max_LateralAcc_g",
        "avg_RollRate_dps", "max_RollRate_dps",
        "avg_PitchRate_dps", "max_PitchRate_dps",
        "avg_VerticalAcc_g", "max_VerticalAcc_g",
    ],
    "Thermal": [
        "avg_OilTemp_C", "max_OilTemp_C",
        "avg_WaterTemp_C", "max_WaterTemp_C",
        "min_OilPressure_bar",
    ],
    "Electronics": [
        "avg_ExternalVoltage_V", "min_ExternalVoltage_V",
        "avg_FuelPressure_bar", "min_FuelPressure_bar",
        "avg_Luminosity",
    ],
    "Tyre Management": [
        "avg_YawRate_dps", "max_YawRate_dps",
        "avg_LateralAcc_g",
        "lap_time_s",
    ],
}


def _get_db():
    from pipeline.updater._db import get_db
    return get_db()


# ── Data loaders ─────────────────────────────────────────────────────────


def load_aim_session_data(session_id: str) -> pd.DataFrame:
    """Load per-lap summary data from aim_lap_summary for anomaly analysis."""
    db = _get_db()
    docs = list(db["aim_lap_summary"].find(
        {"session_id": session_id},
        {"_id": 0},
    ).sort("lap_number", 1))

    if not docs:
        raise ValueError(f"No lap summary data for session {session_id}")

    return pd.DataFrame(docs)


def load_aim_session_meta(session_id: str) -> dict:
    """Load session metadata."""
    db = _get_db()
    doc = db["aim_sessions"].find_one({"session_id": session_id}, {"_id": 0})
    if not doc:
        raise ValueError(f"Session not found: {session_id}")
    return doc


# ── Anomaly scoring ─────────────────────────────────────────────────────


def run_aim_anomaly(session_id: str) -> dict:
    """Run the full anomaly + health pipeline on an AiM session.

    Returns the same JSON shape as the existing fleet anomaly endpoint:
    {
        session_id, driver, track, systems: [...],
        overall_health, severity, action
    }
    """
    from omnianalytics.anomaly import AnomalyEnsemble
    from omnianalytics._types import SeverityLevel
    from omnihealth.health import score_health, assess_component
    from omnidata._types import TabularDataset, DatasetProfile, ColumnProfile, ColumnRole, DType

    df = load_aim_session_data(session_id)
    meta = load_aim_session_meta(session_id)

    systems_result = []
    overall_scores = []

    for system_name, feature_list in AIM_SYSTEM_FEATURES.items():
        # Filter to features that exist in the data
        available = [f for f in feature_list if f in df.columns]
        if len(available) < 2:
            logger.warning("System %s: only %d features available, skipping", system_name, len(available))
            systems_result.append({
                "system": system_name,
                "health_pct": 100.0,
                "severity": "NORMAL",
                "action": "NONE",
                "features_used": available,
                "anomaly_scores": [],
            })
            overall_scores.append(100.0)
            continue

        subset = df[available].copy()

        # Build a minimal TabularDataset for the ensemble
        col_profiles = [
            ColumnProfile(name=c, dtype=DType.FLOAT, role=ColumnRole.METRIC)
            for c in available
        ]
        profile = DatasetProfile(
            row_count=len(subset),
            column_count=len(available),
            columns=col_profiles,
            metric_cols=available,
        )
        dataset = TabularDataset(df=subset, profile=profile, source=f"aim:{session_id}")

        try:
            ensemble = AnomalyEnsemble()
            result = ensemble.run(dataset, columns=available)

            # Aggregate to system-level health
            row_scores = [s.score_mean for s in result.scores]
            mean_score = float(np.mean(row_scores)) if row_scores else 0.0
            health_pct = score_health(mean_score)

            # SHAP features live in model_scores, not on AnomalyScore directly
            shap_map = result.scores[0].model_scores.get("shap_explanations", {}) if result.scores else {}

            # Per-lap anomaly scores
            per_lap = []
            for i, score_obj in enumerate(result.scores):
                lap_num = int(df.iloc[i].get("lap_number", i + 1))
                # Extract SHAP features for this row if available
                row_shap = shap_map.get(i, [])
                if isinstance(row_shap, dict):
                    row_shap = row_shap.get("features", [])
                sev_val = score_obj.severity.value if hasattr(score_obj.severity, "value") else str(score_obj.severity)
                per_lap.append({
                    "lap": lap_num,
                    "score": round(score_obj.score_mean, 4),
                    "severity": sev_val.upper(),
                    "top_features": [
                        {"feature": f.get("feature", ""), "importance": round(f.get("importance", 0), 4)}
                        for f in (row_shap if isinstance(row_shap, list) else [])
                    ],
                })

            severity_order = ["normal", "low", "medium", "high", "critical"]
            worst_severity = max(
                (s.severity for s in result.scores),
                key=lambda sv: severity_order.index(
                    sv.value if hasattr(sv, "value") else str(sv)
                ),
                default=SeverityLevel.NORMAL,
            )
            severity_str = (worst_severity.value if hasattr(worst_severity, "value") else str(worst_severity)).upper()

            systems_result.append({
                "system": system_name,
                "health_pct": round(health_pct, 1),
                "severity": severity_str,
                "action": _action_from_severity(severity_str),
                "features_used": available,
                "anomaly_scores": per_lap,
            })
            overall_scores.append(health_pct)

        except Exception as e:
            logger.error("Anomaly failed for %s/%s: %s", session_id, system_name, e)
            systems_result.append({
                "system": system_name,
                "health_pct": 100.0,
                "severity": "NORMAL",
                "action": "NONE",
                "features_used": available,
                "anomaly_scores": [],
                "error": str(e),
            })
            overall_scores.append(100.0)

    overall_health = round(float(np.mean(overall_scores)), 1) if overall_scores else 100.0
    _sev_order = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    worst_overall = max(
        (s["severity"] for s in systems_result),
        key=lambda sv: _sev_order.index(sv) if sv in _sev_order else 0,
        default="NORMAL",
    )

    return {
        "session_id": session_id,
        "driver": meta.get("driver", ""),
        "track": meta.get("track", ""),
        "vehicle": meta.get("vehicle", ""),
        "systems": systems_result,
        "overall_health": overall_health,
        "severity": worst_overall,
        "action": _action_from_severity(worst_overall),
    }


def _action_from_severity(severity: str) -> str:
    """Map severity to maintenance action."""
    return {
        "CRITICAL": "ALERT_AND_REMEDIATE",
        "HIGH": "ALERT",
        "MEDIUM": "LOG_AND_MONITOR",
        "LOW": "LOG",
        "NORMAL": "NONE",
    }.get(severity, "NONE")


# ── Track-level anomaly detection (statistical thresholds) ────────────


def _detect_threshold_anomalies(session_id: str) -> tuple[list[dict], list[dict]]:
    """Run statistical threshold rules on raw telemetry.

    Returns (anomaly_zones, point_events).
    """
    db = _get_db()

    docs = list(db["aim_telemetry"].find(
        {"session_id": session_id},
        {"_id": 0, "time_s": 1, "lap": 1,
         "GPS_Lat": 1, "GPS_Lon": 1, "GPS_Speed_kmh": 1,
         "FrontBrake_bar": 1, "RearBrake_bar": 1,
         "OilTemp_C": 1, "WaterTemp_C": 1, "OilPressure_bar": 1,
         "FuelPressure_bar": 1, "Throttle_pct": 1,
         "LateralAcc_g": 1, "ExternalVoltage_V": 1},
    ).sort("time_s", 1))

    if not docs:
        return [], []

    df = pd.DataFrame(docs)

    # Session-wide stats
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "p5": float(series.quantile(0.05)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
        }

    zones: list[dict] = []
    points: list[dict] = []

    # Identify outlap/inlap — exclude from analysis
    lap_times = df.groupby("lap")["time_s"].agg(["min", "max"])
    lap_times["duration"] = lap_times["max"] - lap_times["min"]
    median_duration = lap_times["duration"].median()
    hot_laps = set(
        lap_times[lap_times["duration"] < median_duration * 1.5].index
    )

    for lap_num in hot_laps:
        lap_df = df[df["lap"] == lap_num].copy()
        if lap_df.empty:
            continue

        # ── Rule 1: Brake pressure drop during braking zone ──
        if "FrontBrake_bar" in stats:
            braking = lap_df[lap_df["FrontBrake_bar"] > 2.0]
            if len(braking) > 5:
                brake_mean = float(braking["FrontBrake_bar"].mean())
                brake_std = float(braking["FrontBrake_bar"].std()) or 1.0
                drops = braking[braking["FrontBrake_bar"] < brake_mean - 2 * brake_std]
                if len(drops) >= 3:
                    start = drops.iloc[0]
                    end = drops.iloc[-1]
                    zones.append({
                        "type": "segment",
                        "system": "Brakes",
                        "severity": "HIGH" if len(drops) > 5 else "MEDIUM",
                        "channel": "FrontBrake_bar",
                        "reason": f"Brake pressure dropped {((brake_mean - float(drops['FrontBrake_bar'].mean())) / brake_mean * 100):.0f}% below braking zone mean",
                        "lap": int(lap_num),
                        "start_time_s": float(start["time_s"]),
                        "end_time_s": float(end["time_s"]),
                        "start_gps": {"lat": float(start.get("GPS_Lat", 0)), "lon": float(start.get("GPS_Lon", 0))},
                        "end_gps": {"lat": float(end.get("GPS_Lat", 0)), "lon": float(end.get("GPS_Lon", 0))},
                    })

        # ── Rule 2: Oil temp spike (>95th percentile) ──
        if "OilTemp_C" in stats:
            threshold = stats["OilTemp_C"]["p95"]
            spikes = lap_df[lap_df["OilTemp_C"] > threshold]
            if len(spikes) > 0:
                worst = spikes.loc[spikes["OilTemp_C"].idxmax()]
                points.append({
                    "type": "point",
                    "system": "Thermal",
                    "severity": "HIGH" if float(worst["OilTemp_C"]) > stats["OilTemp_C"]["p99"] else "MEDIUM",
                    "channel": "OilTemp_C",
                    "reason": f"Oil temp {float(worst['OilTemp_C']):.1f}°C (95th pct = {threshold:.1f}°C)",
                    "lap": int(lap_num),
                    "time_s": float(worst["time_s"]),
                    "gps": {"lat": float(worst.get("GPS_Lat", 0)), "lon": float(worst.get("GPS_Lon", 0))},
                    "value": round(float(worst["OilTemp_C"]), 1),
                })

        # ── Rule 3: Water temp spike (>95th percentile) ──
        if "WaterTemp_C" in stats:
            threshold = stats["WaterTemp_C"]["p95"]
            spikes = lap_df[lap_df["WaterTemp_C"] > threshold]
            if len(spikes) > 0:
                worst = spikes.loc[spikes["WaterTemp_C"].idxmax()]
                points.append({
                    "type": "point",
                    "system": "Thermal",
                    "severity": "HIGH" if float(worst["WaterTemp_C"]) > stats["WaterTemp_C"]["p99"] else "MEDIUM",
                    "channel": "WaterTemp_C",
                    "reason": f"Water temp {float(worst['WaterTemp_C']):.1f}°C (95th pct = {threshold:.1f}°C)",
                    "lap": int(lap_num),
                    "time_s": float(worst["time_s"]),
                    "gps": {"lat": float(worst.get("GPS_Lat", 0)), "lon": float(worst.get("GPS_Lon", 0))},
                    "value": round(float(worst["WaterTemp_C"]), 1),
                })

        # ── Rule 4: Oil pressure drop (<5th percentile) ──
        if "OilPressure_bar" in stats:
            threshold = stats["OilPressure_bar"]["p5"]
            drops = lap_df[lap_df["OilPressure_bar"] < threshold]
            if len(drops) > 0:
                worst = drops.loc[drops["OilPressure_bar"].idxmin()]
                points.append({
                    "type": "point",
                    "system": "Thermal",
                    "severity": "HIGH",
                    "channel": "OilPressure_bar",
                    "reason": f"Oil pressure {float(worst['OilPressure_bar']):.2f} bar (5th pct = {threshold:.2f} bar)",
                    "lap": int(lap_num),
                    "time_s": float(worst["time_s"]),
                    "gps": {"lat": float(worst.get("GPS_Lat", 0)), "lon": float(worst.get("GPS_Lon", 0))},
                    "value": round(float(worst["OilPressure_bar"]), 2),
                })

        # ── Rule 5: Lateral G spike (>99th percentile) ──
        if "LateralAcc_g" in stats:
            threshold = stats["LateralAcc_g"]["p99"]
            spikes = lap_df[lap_df["LateralAcc_g"].abs() > abs(threshold)]
            if len(spikes) > 0:
                worst = spikes.loc[spikes["LateralAcc_g"].abs().idxmax()]
                points.append({
                    "type": "point",
                    "system": "Suspension",
                    "severity": "MEDIUM",
                    "channel": "LateralAcc_g",
                    "reason": f"Lateral G spike: {float(worst['LateralAcc_g']):.2f}g (99th pct = {abs(threshold):.2f}g)",
                    "lap": int(lap_num),
                    "time_s": float(worst["time_s"]),
                    "gps": {"lat": float(worst.get("GPS_Lat", 0)), "lon": float(worst.get("GPS_Lon", 0))},
                    "value": round(float(worst["LateralAcc_g"]), 2),
                })

        # ── Rule 6: Voltage drop (<11.5V) ──
        if "ExternalVoltage_V" in lap_df.columns:
            drops = lap_df[lap_df["ExternalVoltage_V"] < 11.5]
            if len(drops) > 0:
                worst = drops.loc[drops["ExternalVoltage_V"].idxmin()]
                points.append({
                    "type": "point",
                    "system": "Electronics",
                    "severity": "HIGH" if float(worst["ExternalVoltage_V"]) < 10.5 else "MEDIUM",
                    "channel": "ExternalVoltage_V",
                    "reason": f"Battery voltage drop: {float(worst['ExternalVoltage_V']):.1f}V",
                    "lap": int(lap_num),
                    "time_s": float(worst["time_s"]),
                    "gps": {"lat": float(worst.get("GPS_Lat", 0)), "lon": float(worst.get("GPS_Lon", 0))},
                    "value": round(float(worst["ExternalVoltage_V"]), 1),
                })

    return zones, points


def _detect_degradation(session_id: str) -> list[dict]:
    """Detect per-channel degradation trends across laps using linear regression."""
    from scipy.stats import linregress

    db = _get_db()
    lap_docs = list(db["aim_lap_summary"].find(
        {"session_id": session_id},
        {"_id": 0},
    ).sort("lap_number", 1))

    if len(lap_docs) < 3:
        return []

    df = pd.DataFrame(lap_docs)

    # Exclude outlaps
    if "lap_time_s" in df.columns and len(df) > 2:
        median_time = df["lap_time_s"].median()
        df = df[df["lap_time_s"] < median_time * 1.5].reset_index(drop=True)

    if len(df) < 3:
        return []

    channels = [
        ("avg_OilPressure_bar", "Oil Pressure", "Thermal", "bar"),
        ("avg_OilTemp_C", "Oil Temperature", "Thermal", "°C"),
        ("avg_WaterTemp_C", "Water Temperature", "Thermal", "°C"),
        ("avg_FuelPressure_bar", "Fuel Pressure", "Electronics", "bar"),
        ("avg_ExternalVoltage_V", "Battery Voltage", "Electronics", "V"),
        ("avg_FrontBrake_bar", "Front Brake Pressure", "Brakes", "bar"),
    ]

    degradations = []
    x = np.arange(len(df))

    for col, label, system, unit in channels:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if len(values) < 3:
            continue

        slope, intercept, r_value, p_value, std_err = linregress(x[:len(values)], values)

        mean_val = float(values.mean())
        if mean_val == 0:
            continue

        rate_pct = (slope / mean_val) * 100

        if p_value < 0.1 and abs(rate_pct) > 1.0:
            direction = "increasing" if slope > 0 else "decreasing"

            if abs(rate_pct) > 5:
                severity = "HIGH"
            elif abs(rate_pct) > 2:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            bad_increasing = col in ("avg_OilTemp_C", "avg_WaterTemp_C")
            bad_decreasing = col in ("avg_OilPressure_bar", "avg_FuelPressure_bar", "avg_ExternalVoltage_V", "avg_FrontBrake_bar")
            is_concerning = (direction == "increasing" and bad_increasing) or (direction == "decreasing" and bad_decreasing)

            if not is_concerning:
                continue

            message = f"{label} {direction} {abs(rate_pct):.1f}% per lap"
            if severity in ("HIGH", "MEDIUM"):
                message += " — investigate before next session"

            degradations.append({
                "system": system,
                "channel": col.replace("avg_", ""),
                "label": label,
                "unit": unit,
                "trend": direction,
                "rate_per_lap": round(float(slope), 4),
                "rate_pct_per_lap": round(rate_pct, 1),
                "r_squared": round(float(r_value ** 2), 3),
                "message": message,
                "severity": severity,
                "lap_values": [round(float(v), 2) for v in values.tolist()],
            })

    return degradations


def _compute_lap_deltas(session_id: str) -> list[dict]:
    """Compare consecutive hot laps to find where time was gained/lost."""
    db = _get_db()
    session = db["aim_sessions"].find_one({"session_id": session_id}, {"_id": 0, "laps": 1})
    if not session or not session.get("laps"):
        return []

    laps = session["laps"]
    if len(laps) < 2:
        return []

    median_time = np.median([l["lap_time_s"] for l in laps])
    hot_laps = [l for l in laps if l["lap_time_s"] < median_time * 1.5]
    if len(hot_laps) < 2:
        return []

    summaries = {
        doc["lap_number"]: doc
        for doc in db["aim_lap_summary"].find(
            {"session_id": session_id},
            {"_id": 0},
        )
    }

    deltas = []
    for i in range(1, len(hot_laps)):
        prev = hot_laps[i - 1]
        curr = hot_laps[i]
        delta_s = curr["lap_time_s"] - prev["lap_time_s"]
        faster = delta_s < 0

        reasons = []
        prev_sum = summaries.get(prev["lap_number"], {})
        curr_sum = summaries.get(curr["lap_number"], {})

        comparisons = [
            ("avg_FrontBrake_bar", "Brake pressure", "bar"),
            ("max_FrontBrake_bar", "Peak brake force", "bar"),
            ("avg_GPS_Speed_kmh", "Avg speed", "km/h"),
            ("max_GPS_Speed_kmh", "Top speed", "km/h"),
            ("avg_Throttle_pct", "Throttle application", "%"),
            ("avg_LateralAcc_g", "Cornering G", "g"),
            ("brake_application_pct", "Brake usage", "%"),
        ]

        for col, label, unit in comparisons:
            prev_val = prev_sum.get(col)
            curr_val = curr_sum.get(col)
            if prev_val is None or curr_val is None:
                continue
            diff = curr_val - prev_val
            pct_diff = (diff / prev_val * 100) if prev_val != 0 else 0

            if abs(pct_diff) > 2:
                direction = "more" if diff > 0 else "less"
                reasons.append({
                    "channel": col,
                    "detail": f"{abs(pct_diff):.1f}% {direction} {label.lower()} ({curr_val:.1f} vs {prev_val:.1f} {unit})",
                })

        deltas.append({
            "lap": curr["lap_number"],
            "delta_s": round(delta_s, 3),
            "vs_lap": prev["lap_number"],
            "faster": faster,
            "reasons": reasons[:4],
        })

    return deltas


def run_track_anomalies(session_id: str) -> dict:
    """Run the full track-level intelligence pipeline."""
    meta = load_aim_session_meta(session_id)

    zones, point_events = _detect_threshold_anomalies(session_id)
    degradation = _detect_degradation(session_id)
    lap_deltas = _compute_lap_deltas(session_id)

    return {
        "session_id": session_id,
        "driver": meta.get("driver", ""),
        "track": meta.get("track", ""),
        "anomaly_zones": zones,
        "point_events": point_events,
        "degradation": degradation,
        "lap_deltas": lap_deltas,
        "summary": {
            "total_anomalies": len(zones) + len(point_events),
            "degradation_count": len(degradation),
            "worst_severity": max(
                [z.get("severity", "NORMAL") for z in zones] +
                [p.get("severity", "NORMAL") for p in point_events] +
                ["NORMAL"],
                key=lambda s: ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"].index(s)
                if s in ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"] else 0,
            ),
        },
    }
