"""Autonomous extraction triggers — NEW, not in DataSense.

AutoExtractor generates insights automatically from analytics outputs
based on severity thresholds, without requiring user interaction.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from omnianalytics._types import AnomalyResult, ForecastResult, SeverityLevel
from omnikex._types import (
    ExtractionResult,
    Insight,
    InsightPillar,
    KexLLMConfig,
    LLMProvider,
)
from omnikex.pillars import extract_anomaly, extract_forecast, extract_realtime
from omnikex.profiler import profile_data

logger = logging.getLogger(__name__)

# Severity ordering for threshold comparison
_SEVERITY_ORDER = {
    SeverityLevel.NORMAL: 0,
    SeverityLevel.LOW: 1,
    SeverityLevel.MEDIUM: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.CRITICAL: 4,
}


class AutoExtractor:
    """Autonomous insight extraction triggered by severity thresholds."""

    def __init__(
        self,
        llm_config: Optional[KexLLMConfig] = None,
        severity_threshold: SeverityLevel = SeverityLevel.MEDIUM,
        verify_grounding: bool = True,
    ):
        self._llm_config = llm_config
        self._severity_threshold = severity_threshold
        self._verify = verify_grounding

    def from_anomaly_result(
        self,
        data: pd.DataFrame,
        anomaly_result: AnomalyResult,
        *,
        question: Optional[str] = None,
    ) -> Optional[ExtractionResult]:
        """Auto-trigger if anomaly_result has anomalies above severity threshold.

        Returns None if no anomalies meet the threshold.
        """
        threshold_val = _SEVERITY_ORDER[self._severity_threshold]

        # Check if any scores meet the threshold
        above_threshold = [
            s for s in anomaly_result.scores
            if _SEVERITY_ORDER.get(s.severity, 0) >= threshold_val
        ]
        if not above_threshold:
            logger.info("No anomalies above %s threshold — skipping extraction",
                       self._severity_threshold.value)
            return None

        insights: List[Insight] = []

        # Generate anomaly insight
        q = question or "Analyze the detected anomalies, their severity, root causes, and recommended actions."
        insight = extract_anomaly(
            data, anomaly_result, q,
            llm_config=self._llm_config,
            verify=self._verify,
        )
        insights.append(insight)

        profile = profile_data(data)
        return ExtractionResult(
            insights=insights,
            source_profile=profile,
        )

    def from_forecast_results(
        self,
        data: pd.DataFrame,
        forecast_results: List[ForecastResult],
        *,
        question: Optional[str] = None,
    ) -> Optional[ExtractionResult]:
        """Auto-trigger if any forecast shows significant trends.

        Returns None if no forecasts warrant extraction.
        """
        if not forecast_results:
            return None

        # Check if any forecast has predictions
        has_predictions = any(fr.predictions for fr in forecast_results)
        if not has_predictions:
            logger.info("No forecast predictions available — skipping extraction")
            return None

        q = question or "Analyze the forecast results, model agreement, and recommended preparations."
        insight = extract_forecast(
            data, forecast_results, q,
            llm_config=self._llm_config,
            verify=self._verify,
        )

        profile = profile_data(data)
        return ExtractionResult(
            insights=[insight],
            source_profile=profile,
        )

    def from_health_report(
        self,
        data: pd.DataFrame,
        health_report: Any,
        *,
        question: Optional[str] = None,
    ) -> ExtractionResult:
        """Full autonomous pipeline from a HealthReport.

        Generates:
          1. Realtime insight from raw data
          2. Anomaly insight for components with severity >= threshold
          3. Forecast insight from risk assessments
        """
        insights: List[Insight] = []
        threshold_val = _SEVERITY_ORDER[self._severity_threshold]

        # 1. Realtime insight
        realtime_q = question or "Provide an overall analysis of this data: key patterns, trends, and notable observations."
        try:
            realtime_insight = extract_realtime(
                data, realtime_q,
                llm_config=self._llm_config,
                verify=self._verify,
            )
            insights.append(realtime_insight)
        except Exception as e:
            logger.warning("Realtime extraction failed: %s", e)

        # 2. Check components for anomaly severity
        components_above = [
            c for c in health_report.components
            if _SEVERITY_ORDER.get(c.severity, 0) >= threshold_val
        ]
        if components_above:
            # Build a synthetic anomaly result from health scores
            from omnianalytics._types import AnomalyScore
            scores = []
            for comp in health_report.components:
                scores.append(AnomalyScore(
                    index=0,
                    composite_score=comp.anomaly_score,
                    severity=comp.severity,
                    vote_count=comp.vote_count,
                    total_models=comp.total_models,
                    top_features=comp.top_features,
                    model_scores={},
                ))
            anomaly_result = AnomalyResult(
                scores=scores,
                method="health_report",
                feature_names=list(data.select_dtypes(include="number").columns),
            )

            anomaly_q = (
                f"Analyze anomalies in {len(components_above)} components "
                f"(severity >= {self._severity_threshold.value}): "
                + ", ".join(c.component for c in components_above)
            )
            try:
                anomaly_insight = extract_anomaly(
                    data, anomaly_result, anomaly_q,
                    llm_config=self._llm_config,
                    verify=self._verify,
                )
                insights.append(anomaly_insight)
            except Exception as e:
                logger.warning("Anomaly extraction failed: %s", e)

        # 3. Forecast insight from risk assessments
        if health_report.risk_assessments:
            # Build synthetic forecast results from risk assessments
            forecast_results = []
            for ra in health_report.risk_assessments:
                fr = ForecastResult(
                    model_name=f"risk_{ra.feature}",
                    predictions=[{
                        "value": ra.forecast_value,
                        "step": 1,
                    }],
                    feature=ra.feature,
                )
                forecast_results.append(fr)

            forecast_q = "Analyze the risk-based forecasts and recommended preparatory actions."
            try:
                forecast_insight = extract_forecast(
                    data, forecast_results, forecast_q,
                    llm_config=self._llm_config,
                    verify=self._verify,
                )
                insights.append(forecast_insight)
            except Exception as e:
                logger.warning("Forecast extraction failed: %s", e)

        profile = profile_data(data)
        return ExtractionResult(
            insights=insights,
            source_profile=profile,
        )

    def from_pipeline(
        self,
        data: pd.DataFrame,
        component_map: Dict[str, List[str]],
        *,
        horizon: int = 10,
        question: Optional[str] = None,
    ) -> ExtractionResult:
        """Complete autonomous pipeline: data in → insights out.

        Runs omnihealth.assess() → from_health_report().
        """
        from omnihealth import assess

        report = assess(data, component_map, horizon=horizon)
        return self.from_health_report(data, report, question=question)
