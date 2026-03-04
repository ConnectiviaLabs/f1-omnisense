"""FastAPI endpoints for omniKeX.

Follows the omnihealth/server.py pattern:
  - File upload with CSV/JSON parsing
  - JSON body for structured inputs
  - Returns JSON responses
"""

from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from omnikex import extract, extract_report, AutoExtractor
from omnikex._types import (
    ExtractionResult,
    InsightPillar,
    KexLLMConfig,
    LLMProvider,
)
from omnikex.llm import list_available_providers

app = FastAPI(
    title="omniKeX",
    description="Knowledge Extraction from Analytics Data using WISE Framework",
    version="1.0.0",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_upload(file: UploadFile) -> pd.DataFrame:
    """Parse an uploaded CSV or JSON file into a DataFrame."""
    content = file.file.read()
    filename = file.filename or ""

    if filename.endswith(".json"):
        data = json.loads(content)
        return pd.DataFrame(data)
    else:
        return pd.read_csv(io.BytesIO(content))


def _make_config(
    provider: str = "auto",
    model: str = "",
    persona: Optional[str] = None,
) -> KexLLMConfig:
    """Build a KexLLMConfig from request parameters."""
    return KexLLMConfig(
        provider=LLMProvider(provider) if provider else LLMProvider.AUTO,
        model=model or "",
        persona=persona,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/extract")
async def extract_endpoint(
    file: UploadFile = File(...),
    question: str = Form("Analyze this data and provide key insights."),
    pillar: str = Form("realtime"),
    provider: str = Form("auto"),
    model: str = Form(""),
    persona: Optional[str] = Form(None),
    response_length: str = Form("medium"),
    verify_grounding: bool = Form(True),
):
    """Single pillar extraction from uploaded data."""
    try:
        df = _parse_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    try:
        insight = extract(
            df,
            question,
            pillar=InsightPillar(pillar),
            llm_provider=LLMProvider(provider),
            model=model,
            persona=persona,
            response_length=response_length,
            verify_grounding=verify_grounding,
        )
        return JSONResponse(content=insight.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/report")
async def extract_report_endpoint(
    file: UploadFile = File(...),
    health_report_json: str = Form(...),
    question: Optional[str] = Form(None),
    provider: str = Form("auto"),
    model: str = Form(""),
    persona: Optional[str] = Form(None),
):
    """Full autonomous extraction from a health report."""
    try:
        df = _parse_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    try:
        from omnihealth._types import HealthReport
        report_dict = json.loads(health_report_json)
        # Reconstruct HealthReport from dict
        report = _reconstruct_health_report(report_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse health report: {e}")

    try:
        result = extract_report(
            df, report,
            question=question,
            llm_provider=LLMProvider(provider),
            model=model,
            persona=persona,
        )
        return JSONResponse(content=result.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/realtime")
async def extract_realtime_endpoint(
    file: UploadFile = File(...),
    question: str = Form("Analyze this data and provide key insights."),
    provider: str = Form("auto"),
    model: str = Form(""),
    persona: Optional[str] = Form(None),
    response_length: str = Form("medium"),
):
    """Realtime-only extraction."""
    try:
        df = _parse_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    try:
        insight = extract(
            df, question,
            pillar=InsightPillar.REALTIME,
            llm_provider=LLMProvider(provider),
            model=model,
            persona=persona,
            response_length=response_length,
        )
        return JSONResponse(content=insight.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/anomaly")
async def extract_anomaly_endpoint(
    file: UploadFile = File(...),
    anomaly_result_json: str = Form(...),
    question: str = Form("Analyze the detected anomalies and their implications."),
    provider: str = Form("auto"),
    model: str = Form(""),
    persona: Optional[str] = Form(None),
    response_length: str = Form("medium"),
):
    """Anomaly-only extraction (requires anomaly_result JSON)."""
    try:
        df = _parse_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    try:
        from omnianalytics._types import AnomalyResult
        ar_dict = json.loads(anomaly_result_json)
        anomaly_result = AnomalyResult.from_dict(ar_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse anomaly result: {e}")

    try:
        insight = extract(
            df, question,
            anomaly_result=anomaly_result,
            llm_provider=LLMProvider(provider),
            model=model,
            persona=persona,
            response_length=response_length,
        )
        return JSONResponse(content=insight.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/forecast")
async def extract_forecast_endpoint(
    file: UploadFile = File(...),
    forecast_results_json: str = Form(...),
    question: str = Form("Analyze the forecast results and their implications."),
    provider: str = Form("auto"),
    model: str = Form(""),
    persona: Optional[str] = Form(None),
    response_length: str = Form("medium"),
):
    """Forecast-only extraction (requires forecast_results JSON)."""
    try:
        df = _parse_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    try:
        from omnianalytics._types import ForecastResult
        fr_list = json.loads(forecast_results_json)
        forecast_results = [ForecastResult.from_dict(fr) for fr in fr_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse forecast results: {e}")

    try:
        insight = extract(
            df, question,
            forecast_results=forecast_results,
            llm_provider=LLMProvider(provider),
            model=model,
            persona=persona,
            response_length=response_length,
        )
        return JSONResponse(content=insight.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/providers")
async def providers_endpoint():
    """List available LLM providers."""
    return {"providers": list_available_providers()}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _reconstruct_health_report(d: Dict[str, Any]):
    """Reconstruct a HealthReport from a dictionary."""
    from omnihealth._types import (
        HealthReport,
        HealthScore,
        MaintenanceSchedule,
        MaintenanceTask,
        RiskAssessment,
    )

    components = [HealthScore.from_dict(c) for c in d.get("components", [])]
    risk_assessments = [RiskAssessment.from_dict(r) for r in d.get("risk_assessments", [])]

    schedule_dict = d.get("schedule", {})
    tasks = []
    for t in schedule_dict.get("tasks", []):
        from omnihealth._types import MaintenancePriority, MaintenanceAction
        tasks.append(MaintenanceTask(
            task_id=t["task_id"],
            component=t["component"],
            feature=t["feature"],
            priority=MaintenancePriority(t["priority"]),
            action=MaintenanceAction(t["action"]),
            description=t["description"],
            reason=t["reason"],
            estimated_hours=t["estimated_hours"],
        ))

    schedule = MaintenanceSchedule(
        generated_at=schedule_dict.get("generated_at", ""),
        total_tasks=schedule_dict.get("total_tasks", 0),
        priority_breakdown=schedule_dict.get("priority_breakdown", {}),
        tasks=tasks,
        summary=schedule_dict.get("summary", ""),
    )

    from omnihealth._types import RiskLevel
    return HealthReport(
        components=components,
        risk_assessments=risk_assessments,
        schedule=schedule,
        overall_health=d.get("overall_health", 100.0),
        overall_risk=RiskLevel(d.get("overall_risk", "low")),
        generated_at=d.get("generated_at", ""),
    )
