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
