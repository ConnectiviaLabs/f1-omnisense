"""FastAPI server exposing all OmniAnalytics capabilities.

Usage:
    uvicorn omnianalytics.server:app --host 0.0.0.0 --port 8300
"""

from __future__ import annotations

import io
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OmniAnalytics",
    description="Unified analytics service — anomaly detection, forecasting, feature engineering.",
    version="0.1.0",
)


# ── Helpers ──────────────────────────────────────────────────────────

def _load_upload(file: UploadFile, format: Optional[str] = None, sample: Optional[int] = None):
    from omnidata.loader import load
    data = file.file.read()
    return load(data, filename=file.filename or "", format=format, sample=sample)


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ── Load & Profile ──────────────────────────────────────────────────

@app.post("/load")
async def load_endpoint(
    file: UploadFile = File(...),
    format: Optional[str] = Form(None),
    sample: Optional[int] = Form(None),
):
    """Load a tabular file, return profile + data preview."""
    dataset = _load_upload(file, format, sample)

    from omnidata.profiler import profile
    profile(dataset)

    return dataset.to_dict(include_data=True, max_rows=50)


@app.post("/profile")
async def profile_endpoint(
    file: UploadFile = File(...),
    format: Optional[str] = Form(None),
):
    """Load and compute full statistical profile."""
    dataset = _load_upload(file, format)

    from omnidata.profiler import profile
    result = profile(dataset)
    return result.to_dict()


# ── Preprocessing ───────────────────────────────────────────────────

@app.post("/preprocess")
async def preprocess_endpoint(
    file: UploadFile = File(...),
    coerce_types: bool = Form(True),
    normalize_time: bool = Form(True),
    fill_strategy: str = Form("median"),
):
    """Load, preprocess, return cleaned data + profile."""
    dataset = _load_upload(file)

    from omnidata.preprocessing import preprocess
    preprocess(dataset, coerce_types=coerce_types, normalize_time=normalize_time, fill_strategy=fill_strategy)

    return dataset.to_dict(include_data=True, max_rows=50)


# ── Anomaly Detection ───────────────────────────────────────────────

@app.post("/anomaly/detect")
async def anomaly_detect_endpoint(
    file: UploadFile = File(...),
    columns: Optional[str] = Form(None),
    do_preprocess: bool = Form(True),
    explain_critical: bool = Form(True),
    top_k_features: int = Form(3),
):
    """Run anomaly ensemble. HIGH/CRITICAL get SHAP explanations."""
    dataset = _load_upload(file)

    from omnidata.profiler import profile
    from omnidata.preprocessing import preprocess as pp
    from omnianalytics.anomaly import AnomalyEnsemble

    profile(dataset)
    if do_preprocess:
        pp(dataset)

    cols = [c.strip() for c in columns.split(",")] if columns else None
    ensemble = AnomalyEnsemble()
    result = ensemble.run(
        dataset, columns=cols,
        explain_critical=explain_critical, top_k_features=top_k_features,
    )
    return result.to_dict()


@app.post("/anomaly/detect/async")
async def anomaly_detect_async_endpoint(
    file: UploadFile = File(...),
    columns: Optional[str] = Form(None),
    do_preprocess: bool = Form(True),
    explain_critical: bool = Form(True),
):
    """Submit anomaly detection as background job."""
    from omnianalytics.jobs import get_job_manager

    data = file.file.read()
    filename = file.filename or ""
    cols = [c.strip() for c in columns.split(",")] if columns else None

    def _run_anomaly():
        from omnidata.loader import load
        from omnidata.profiler import profile
        from omnidata.preprocessing import preprocess as pp
        from omnianalytics.anomaly import AnomalyEnsemble

        dataset = load(data, filename=filename)
        profile(dataset)
        if do_preprocess:
            pp(dataset)
        return AnomalyEnsemble().run(
            dataset, columns=cols, explain_critical=explain_critical,
        )

    manager = get_job_manager()
    job_id = manager.submit(_run_anomaly)
    return {"job_id": job_id, "status": "queued"}


# ── Anomaly + Forecast (integrated) ────────────────────────────────

@app.post("/anomaly/analyze")
async def anomaly_analyze_endpoint(
    file: UploadFile = File(...),
    columns: Optional[str] = Form(None),
    horizon: int = Form(10),
    forecast_method: str = Form("auto"),
):
    """Full pipeline: anomaly detection + SHAP on HIGH/CRITICAL + forecast their top features."""
    dataset = _load_upload(file)

    from omnidata.profiler import profile
    from omnidata.preprocessing import preprocess as pp
    from omnianalytics.anomaly import AnomalyEnsemble
    from omnianalytics.forecast import forecast_anomaly_features

    profile(dataset)
    pp(dataset)

    cols = [c.strip() for c in columns.split(",")] if columns else None
    ensemble = AnomalyEnsemble()
    anomaly_result = ensemble.run(dataset, columns=cols, explain_critical=True)

    # Forecast the top features from HIGH/CRITICAL anomalies
    forecasts = forecast_anomaly_features(
        dataset, anomaly_result, horizon=horizon, method=forecast_method,
    )

    return {
        "anomaly": anomaly_result.to_dict(),
        "forecasts": {
            severity: [f.to_dict() for f in fcs]
            for severity, fcs in forecasts.items()
        },
    }


# ── Feature Engineering ─────────────────────────────────────────────

@app.post("/features/engineer")
async def features_endpoint(
    file: UploadFile = File(...),
    temporal: bool = Form(True),
    rolling_window: int = Form(3),
    sequence_context: bool = Form(True),
):
    """Load, engineer features, return enriched profile + preview."""
    dataset = _load_upload(file)

    from omnidata.profiler import profile
    from omnidata.features import engineer

    profile(dataset)
    dataset, new_cols = engineer(
        dataset, temporal=temporal, rolling_window=rolling_window,
        sequence_context=sequence_context,
    )

    return {
        "new_features": new_cols,
        "dataset": dataset.to_dict(include_data=True, max_rows=50),
    }


# ── Forecasting ─────────────────────────────────────────────────────

@app.post("/forecast")
async def forecast_endpoint(
    file: UploadFile = File(...),
    column: str = Form(...),
    horizon: int = Form(10),
    method: str = Form("auto"),
):
    """Forecast a single metric column."""
    dataset = _load_upload(file)

    from omnidata.profiler import profile
    from omnidata.preprocessing import preprocess as pp
    from omnianalytics.forecast import forecast

    profile(dataset)
    pp(dataset)

    result = forecast(dataset, column, horizon=horizon, method=method)
    return result.to_dict()


# ── ML Prep ─────────────────────────────────────────────────────────

@app.post("/ml/split")
async def ml_split_endpoint(
    file: UploadFile = File(...),
    columns: Optional[str] = Form(None),
    test_ratio: float = Form(0.2),
    scale: bool = Form(True),
):
    """Prepare train/test split with scaling."""
    dataset = _load_upload(file)

    from omnidata.profiler import profile
    from omnidata.features import train_test_split

    profile(dataset)
    cols = [c.strip() for c in columns.split(",")] if columns else None

    result = train_test_split(
        dataset.df, columns=cols, test_ratio=test_ratio, scale=scale,
    )
    return {
        "feature_names": result["feature_names"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "scaled": scale,
    }


# ── Jobs ────────────────────────────────────────────────────────────

@app.get("/jobs")
async def list_jobs(status: Optional[str] = Query(None)):
    from omnianalytics.jobs import get_job_manager, JobStatus
    manager = get_job_manager()
    st = JobStatus(status) if status else None
    jobs = manager.list_jobs(status=st)
    return {"jobs": [j.to_dict() for j in jobs]}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    from omnianalytics.jobs import get_job_manager
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job.to_dict()
