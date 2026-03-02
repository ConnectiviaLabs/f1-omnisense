"""OmniAnalytics — unified anomaly detection, forecasting, and background jobs.

Quick start:
    from omnidata import load, profile, preprocess
    from omnianalytics import AnomalyEnsemble, forecast, forecast_anomaly_features

    dataset = load("data.csv")
    profile(dataset)
    dataset = preprocess(dataset)
    result = AnomalyEnsemble().run(dataset)
    prediction = forecast(dataset, column="temperature", horizon=24)
"""

from omnianalytics.anomaly import AnomalyEnsemble
from omnianalytics.forecast import forecast, forecast_anomaly_features
from omnianalytics.jobs import get_job_manager

from omnianalytics._types import (
    AnomalyResult, AnomalyScore, SeverityLevel,
    ForecastResult, JobState, JobStatus,
)

__all__ = [
    "AnomalyEnsemble",
    "forecast", "forecast_anomaly_features",
    "get_job_manager",
    "AnomalyResult", "AnomalyScore", "SeverityLevel",
    "ForecastResult", "JobState", "JobStatus",
]
