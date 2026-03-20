"""FastAPI server for omnihealth — predictive maintenance & scheduling.

Follows omnianalytics/omnidapt server patterns.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OmniHealth",
    description="Predictive Maintenance & Maintenance Scheduling — part of omnisuite",
    version="0.1.0",
)

_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_dataset(file_bytes: bytes, filename: str):
    """Load uploaded file into TabularDataset."""
    from omnidata.loader import load
    return load(file_bytes, filename=filename)


# ── Health assessment ────────────────────────────────────────────────────────

@app.post("/assess")
async def full_assessment(
    file: UploadFile = File(...),
    component_map: str = Form(...),
    horizon: int = Form(10),
    forecast_method: str = Form("auto"),
    include_schedule: bool = Form(True),
    include_timeseries: bool = Form(False),
):
    """Full pipeline: health scoring -> risk assessment -> scheduling.

    component_map should be JSON string: {"Motor": ["vibration", "rpm"], ...}
    """
    from omnihealth import assess

    try:
        cmap = json.loads(component_map)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid component_map JSON"})

    dataset = _load_dataset(await file.read(), file.filename)

    report = assess(
        dataset.df, cmap,
        horizon=horizon,
        forecast_method=forecast_method,
        include_schedule=include_schedule,
        include_timeseries=include_timeseries,
    )
    return report.to_dict()


@app.post("/health")
async def health_only(
    file: UploadFile = File(...),
    component_map: str = Form(...),
):
    """Health scoring only (no forecasting or scheduling)."""
    from omnihealth.health import assess_components

    try:
        cmap = json.loads(component_map)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid component_map JSON"})

    dataset = _load_dataset(await file.read(), file.filename)
    scores = assess_components(dataset.df, cmap)
    return {"components": [s.to_dict() for s in scores]}


@app.post("/risk")
async def risk_assessment(
    file: UploadFile = File(...),
    column: str = Form(...),
    horizon: int = Form(10),
    method: str = Form("auto"),
    include_analysis: bool = Form(True),
):
    """Risk assessment for a single column."""
    from omnihealth.risk import assess_feature_risk

    dataset = _load_dataset(await file.read(), file.filename)
    result = assess_feature_risk(
        dataset, column,
        horizon=horizon, method=method,
        include_analysis=include_analysis,
    )
    return result.to_dict()


@app.post("/schedule")
async def schedule_from_health(
    health_scores: str = Form(...),
    risk_assessments: str = Form("[]"),
):
    """Generate schedule from pre-computed health scores and risk assessments."""
    from omnihealth._types import HealthScore, RiskAssessment
    from omnihealth.scheduler import generate_schedule

    try:
        hs_list = [HealthScore.from_dict(h) for h in json.loads(health_scores)]
    except (json.JSONDecodeError, Exception) as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid health_scores: {e}"})

    try:
        ra_data = json.loads(risk_assessments)
        ra_list = [RiskAssessment.from_dict(r) for r in ra_data] if ra_data else None
    except (json.JSONDecodeError, Exception):
        ra_list = None

    schedule = generate_schedule(hs_list, ra_list)
    return schedule.to_dict()


@app.post("/timeseries/analyze")
async def analyze_ts(
    file: UploadFile = File(...),
    column: str = Form(...),
    seasonal_periods: str = Form("[7, 24]"),
):
    """Time series analysis for a single column."""
    from omnihealth.timeseries import analyze

    dataset = _load_dataset(await file.read(), file.filename)
    series = dataset.df[column].dropna()

    try:
        periods = json.loads(seasonal_periods)
    except json.JSONDecodeError:
        periods = [7, 24]

    result = analyze(series, seasonal_periods=periods)
    return result.to_dict()


# ── Async job support ────────────────────────────────────────────────────────

@app.post("/assess/async")
async def assess_async(
    file: UploadFile = File(...),
    component_map: str = Form(...),
    horizon: int = Form(10),
    forecast_method: str = Form("auto"),
    include_schedule: bool = Form(True),
    include_timeseries: bool = Form(False),
):
    """Background full assessment — returns job_id for polling."""
    from omnianalytics.jobs import get_job_manager
    from omnihealth import assess

    try:
        cmap = json.loads(component_map)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid component_map JSON"})

    file_bytes = await file.read()
    filename = file.filename

    def _run():
        dataset = _load_dataset(file_bytes, filename)
        report = assess(
            dataset.df, cmap,
            horizon=horizon,
            forecast_method=forecast_method,
            include_schedule=include_schedule,
            include_timeseries=include_timeseries,
        )
        return report.to_dict()

    manager = get_job_manager()
    job_id = manager.submit(_run)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Poll background job status."""
    from omnianalytics.jobs import get_job_manager

    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": f"Job '{job_id}' not found"})
    return job.to_dict()


@app.get("/health-check")
async def health_check():
    """Simple health check."""
    return {"status": "ok", "service": "omnihealth", "version": "0.1.0"}
