"""omniKeX — Knowledge Extraction from Analytics Data.

Turns numeric analytics outputs into LLM-generated natural language insights
using the WISE framework (Weight/Infer/Show/Exercise) with anti-fabrication
grounding, multi-LLM routing, and optional RL-tuned emphasis weights.

Quick start::

    from omnikex import extract, InsightPillar

    insight = extract(df, "What are the key patterns?")
    print(insight.text)

Autonomous extraction from a full HealthReport::

    from omnikex import extract_report
    from omnihealth import assess

    report = assess(df, component_map)
    result = extract_report(df, report)
    for insight in result.insights:
        print(f"[{insight.pillar.value}] {insight.text[:100]}...")
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from omnikex._types import (
    DataFingerprint,
    DataProfile,
    ExtractionResult,
    GroundingResult,
    Insight,
    InsightPillar,
    InsightStatus,
    KexLLMConfig,
    LLMProvider,
    PILLAR_BASE_WEIGHTS,
    TASK_TEMPERATURES,
    WISEConfig,
    WISEWeights,
)
from omnikex.auto import AutoExtractor
from omnikex.pillars import extract_anomaly, extract_forecast, extract_realtime


def extract(
    data: pd.DataFrame,
    question: str = "Analyze this data and provide key insights.",
    *,
    pillar: InsightPillar = InsightPillar.REALTIME,
    anomaly_result=None,
    forecast_results=None,
    llm_provider: LLMProvider = LLMProvider.AUTO,
    model: str = "",
    persona: Optional[str] = None,
    response_length: str = "medium",
    verify_grounding: bool = True,
) -> Insight:
    """One-call extraction. Auto-selects pillar if anomaly/forecast data provided.

    Args:
        data: DataFrame to analyze.
        question: Question or analysis prompt.
        pillar: Which pillar to use (auto-selected if anomaly/forecast data given).
        anomaly_result: AnomalyResult from omnianalytics (triggers anomaly pillar).
        forecast_results: List of ForecastResult (triggers forecast pillar).
        llm_provider: LLM provider to use (default: auto-detect).
        model: Specific model name override.
        persona: Persona context string (e.g. "You are a CEO...").
        response_length: "short", "medium", or "long".
        verify_grounding: Whether to verify LLM claims against source data.

    Returns:
        Insight with generated text, status, and grounding results.
    """
    cfg = KexLLMConfig(provider=llm_provider, model=model, persona=persona)

    # Auto-select pillar based on provided data
    if anomaly_result is not None:
        pillar = InsightPillar.ANOMALY
    elif forecast_results is not None:
        pillar = InsightPillar.FORECAST

    if pillar == InsightPillar.ANOMALY and anomaly_result is not None:
        return extract_anomaly(
            data, anomaly_result, question,
            llm_config=cfg,
            persona_context=persona,
            response_length=response_length,
            verify=verify_grounding,
        )
    elif pillar == InsightPillar.FORECAST and forecast_results is not None:
        return extract_forecast(
            data, forecast_results, question,
            llm_config=cfg,
            persona_context=persona,
            response_length=response_length,
            verify=verify_grounding,
        )
    else:
        return extract_realtime(
            data, question,
            llm_config=cfg,
            persona_context=persona,
            response_length=response_length,
            verify=verify_grounding,
        )


def extract_report(
    data: pd.DataFrame,
    health_report,
    *,
    question: Optional[str] = None,
    llm_provider: LLMProvider = LLMProvider.AUTO,
    model: str = "",
    persona: Optional[str] = None,
) -> ExtractionResult:
    """Generate insights from a full HealthReport (autonomous).

    Args:
        data: Source DataFrame.
        health_report: HealthReport from omnihealth.assess().
        question: Optional question/focus for insights.
        llm_provider: LLM provider to use.
        model: Specific model name override.
        persona: Persona context string.

    Returns:
        ExtractionResult with insights for each pillar.
    """
    cfg = KexLLMConfig(provider=llm_provider, model=model, persona=persona)
    extractor = AutoExtractor(llm_config=cfg)
    return extractor.from_health_report(data, health_report, question=question)


__all__ = [
    # Types
    "Insight",
    "ExtractionResult",
    "InsightPillar",
    "InsightStatus",
    "WISEWeights",
    "WISEConfig",
    "DataProfile",
    "DataFingerprint",
    "GroundingResult",
    "LLMProvider",
    "KexLLMConfig",
    # Constants
    "PILLAR_BASE_WEIGHTS",
    "TASK_TEMPERATURES",
    # Functions
    "extract",
    "extract_report",
    "extract_realtime",
    "extract_anomaly",
    "extract_forecast",
    # Classes
    "AutoExtractor",
]
