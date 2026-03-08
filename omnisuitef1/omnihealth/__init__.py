"""omnihealth — Predictive Maintenance & Maintenance Scheduling.

Combines best of DataSense (forecasting, risk assessment, scheduling) and
F1 (health scoring, severity classification, temporal feature engineering).

Builds on top of omnianalytics (anomaly detection + forecasting).
Part of omnisuite.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from omnianalytics._types import SeverityLevel
from omnihealth._types import (
    PRIORITY_HOURS,
    SENSOR_UNITS,
    SEVERITY_TO_ACTION,
    HealthReport,
    HealthScore,
    MaintenanceAction,
    MaintenancePriority,
    MaintenanceSchedule,
    MaintenanceTask,
    RiskAssessment,
    RiskLevel,
    TimeSeriesAnalysis,
    TrendDirection,
)
from omnihealth.health import (
    add_lifecycle_context,
    add_temporal_features,
    assess_component,
    assess_components,
    score_health,
)
from omnihealth.risk import (
    assess_all_risks,
    assess_feature_risk,
    assess_risk,
    detect_degradation,
)
from omnihealth.scheduler import (
    determine_priority,
    estimate_completion_hours,
    generate_schedule,
    generate_task,
    infer_sampling_hours,
)
from omnihealth.timeseries import (
    analyze as analyze_timeseries,
    analyze_trend,
    classify_operational_zone,
    detect_anomalies_ts,
    detect_seasonality,
    score_forecastability,
    test_stationarity,
)


def assess(
    data: pd.DataFrame,
    component_map: Dict[str, List[str]],
    *,
    horizon: int = 10,
    forecast_method: str = "auto",
    include_schedule: bool = True,
    include_timeseries: bool = False,
    anomaly_weights: Optional[Dict[str, float]] = None,
    session_key: Optional[int] = None,
    driver_number: Optional[int] = None,
    db=None,
) -> HealthReport:
    """One-call pipeline: health scoring -> risk assessment -> scheduling.

    Parameters
    ----------
    data : DataFrame with numeric sensor/telemetry columns.
    component_map : {"Motor": ["vibration", "rpm"], "Thermal": ["temperature"], ...}
    horizon : Forecast horizon for risk assessment.
    forecast_method : "auto", "arima", "linear", or "lightgbm".
    include_schedule : Generate maintenance schedule.
    include_timeseries : Include full time series analysis in risk assessments.
    anomaly_weights : Model weights for AnomalyEnsemble.

    Returns
    -------
    HealthReport with components, risk assessments, schedule, and overall metrics.
    """
    # ── Feature store cache check ──
    if session_key is not None and driver_number is not None and db is not None:
        from omnianalytics import feature_store
        cached = feature_store.get(db, session_key, driver_number, "health_report")
        if cached is not None:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "feature_store HIT: health_report session=%s driver=%s", session_key, driver_number
            )
            return HealthReport.from_dict(cached)

    # Step 1: Health scoring
    health_scores = assess_components(data, component_map, weights=anomaly_weights)

    # Step 2: Risk assessment — prioritize SHAP top features from critical anomalies
    #
    # Strategy: SHAP explanations tell us WHICH features drove HIGH/CRITICAL
    # anomalies. Forecast those first with full time series analysis, then
    # forecast remaining columns from MEDIUM components with lighter analysis.

    risk_assessments: List[RiskAssessment] = []

    sev_order = {SeverityLevel.NORMAL: 0, SeverityLevel.LOW: 1, SeverityLevel.MEDIUM: 2,
                 SeverityLevel.HIGH: 3, SeverityLevel.CRITICAL: 4}

    # 2a: Extract SHAP top features from HIGH/CRITICAL components (most pressing)
    critical_features: List[str] = []   # ordered by SHAP importance
    secondary_features: List[str] = []  # remaining features from MEDIUM+ components

    for hs in health_scores:
        sev = sev_order.get(hs.severity, 0)
        if sev < 2:
            continue

        component_cols = component_map.get(hs.component, [])
        available_cols = [c for c in component_cols if c in data.columns]

        if sev >= 3:  # HIGH or CRITICAL
            # Only SHAP top features get forecasted (ordered by importance)
            shap_names = [f["feature"] for f in hs.top_features if f.get("feature")]
            for feat in shap_names:
                if feat in available_cols and feat not in critical_features:
                    critical_features.append(feat)
        else:  # MEDIUM
            for col in available_cols:
                if col not in critical_features and col not in secondary_features:
                    secondary_features.append(col)

    # 2b: Build TabularDataset once for all forecasting
    all_risk_columns = critical_features + secondary_features

    if all_risk_columns:
        from omnidata._types import TabularDataset, DatasetProfile, ColumnProfile, ColumnRole, DType

        numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        col_profiles = [
            ColumnProfile(
                name=c,
                dtype=DType.FLOAT,
                role=ColumnRole.METRIC,
                null_count=int(data[c].isna().sum()),
                unique_count=int(data[c].nunique()),
            )
            for c in numeric_cols
        ]

        ts_col = None
        for c in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[c]):
                ts_col = c
                break

        profile = DatasetProfile(
            row_count=len(data),
            column_count=len(numeric_cols),
            columns=col_profiles,
            metric_cols=numeric_cols,
            timestamp_col=ts_col,
        )
        dataset = TabularDataset(df=data, profile=profile)

        # 2c: Forecast critical features FIRST — always with full time series analysis
        critical_set = set(critical_features)
        for col in critical_features:
            try:
                ra = assess_feature_risk(
                    dataset, col,
                    horizon=horizon, method=forecast_method,
                    include_analysis=True,  # always deep-analyze critical features
                )
                risk_assessments.append(ra)
            except Exception:
                pass

        # 2d: Forecast secondary features — time series analysis only if requested
        for col in secondary_features:
            try:
                ra = assess_feature_risk(
                    dataset, col,
                    horizon=horizon, method=forecast_method,
                    include_analysis=include_timeseries,
                )
                risk_assessments.append(ra)
            except Exception:
                pass

    # Step 3: Schedule (pass data for sampling frequency inference)
    if include_schedule:
        schedule = generate_schedule(health_scores, risk_assessments, data=data)
    else:
        schedule = MaintenanceSchedule(
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_tasks=0,
            priority_breakdown={},
            tasks=[],
            summary="Schedule generation skipped.",
        )

    # Step 4: Overall metrics
    if health_scores:
        overall_health = sum(h.health_pct for h in health_scores) / len(health_scores)
    else:
        overall_health = 100.0

    # Worst risk
    risk_order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
    if risk_assessments:
        overall_risk = max(risk_assessments, key=lambda r: risk_order.get(r.risk_level, 0)).risk_level
    elif health_scores:
        overall_risk = max(health_scores, key=lambda h: risk_order.get(h.risk_level, 0)).risk_level
    else:
        overall_risk = RiskLevel.LOW

    report = HealthReport(
        components=health_scores,
        risk_assessments=risk_assessments,
        schedule=schedule,
        overall_health=round(overall_health, 1),
        overall_risk=overall_risk,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    # ── Cache result in feature store ──
    if session_key is not None and driver_number is not None and db is not None:
        from omnianalytics import feature_store as _fs
        try:
            _fs.put(db, session_key, driver_number, "health_report", report.to_dict())
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "Failed to cache health_report for session=%s driver=%s", session_key, driver_number
            )

    return report


__all__ = [
    # Types
    "HealthScore",
    "TimeSeriesAnalysis",
    "RiskAssessment",
    "MaintenanceTask",
    "MaintenanceSchedule",
    "HealthReport",
    "RiskLevel",
    "MaintenancePriority",
    "MaintenanceAction",
    "TrendDirection",
    "SENSOR_UNITS",
    "SEVERITY_TO_ACTION",
    "PRIORITY_HOURS",
    # Health
    "score_health",
    "assess_component",
    "assess_components",
    "add_temporal_features",
    "add_lifecycle_context",
    # Time Series
    "analyze_timeseries",
    "analyze_trend",
    "test_stationarity",
    "detect_seasonality",
    "detect_anomalies_ts",
    "score_forecastability",
    "classify_operational_zone",
    # Risk
    "assess_risk",
    "assess_feature_risk",
    "assess_all_risks",
    "detect_degradation",
    # Scheduler
    "determine_priority",
    "estimate_completion_hours",
    "generate_schedule",
    "generate_task",
    # Convenience
    "assess",
]
