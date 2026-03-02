"""Three pillar extractors: realtime, anomaly, forecast.

Each pillar follows the same flow:
  1. Sample data if needed
  2. Compute data fingerprint
  3. Build WISE pillar prompt with pillar-specific context
  4. Generate via LLM
  5. Verify grounding
  6. Return Insight
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from omnianalytics._types import AnomalyResult, AnomalyScore, ForecastResult, SeverityLevel
from omnikex._types import (
    DataFingerprint,
    Insight,
    InsightPillar,
    InsightStatus,
    KexLLMConfig,
    LLMProvider,
    WISEConfig,
)
from omnikex import llm as kex_llm
from omnikex.profiler import (
    anomaly_aware_sampling,
    build_outlier_summary,
    compute_fingerprint,
    intelligent_sampling,
    verify_grounding,
)
from omnikex.wise import build_pillar_prompt


# ── Realtime extraction ──────────────────────────────────────────────────────

def extract_realtime(
    data: pd.DataFrame,
    question: str = "Analyze this data and provide key insights.",
    *,
    wise_config: Optional[WISEConfig] = None,
    llm_config: Optional[KexLLMConfig] = None,
    collection_name: str = "",
    persona_context: Optional[str] = None,
    response_length: str = "medium",
    verify: bool = True,
) -> Insight:
    """Extract realtime insights from raw data."""
    cfg = llm_config or KexLLMConfig(task_type="realtime")
    cfg.task_type = "realtime"

    # 1. Sample if large
    sampled = intelligent_sampling(data, max_rows=15000)

    # 2. Fingerprint
    fingerprint = compute_fingerprint(sampled, mode="realtime")

    # 3. Build prompt
    prompt = build_pillar_prompt(
        data=sampled,
        user_question=question,
        pillar=InsightPillar.REALTIME,
        wise_config=wise_config,
        persona_context=persona_context,
        response_length=response_length,
        collection_name=collection_name,
    )

    # 4. Generate
    start = time.time()
    text, model_name, provider_name = kex_llm.generate(prompt, cfg)
    gen_time = time.time() - start

    # 5. Grounding
    grounding = verify_grounding(text, fingerprint) if verify else None
    status = InsightStatus.VERIFIED if grounding and grounding.grounding_score >= 0.5 else InsightStatus.GENERATED

    return Insight(
        pillar=InsightPillar.REALTIME,
        text=text,
        status=status,
        model_used=model_name,
        provider_used=provider_name,
        prompt_length=len(prompt),
        generation_time_s=gen_time,
        wise_config=wise_config,
        fingerprint=fingerprint,
        grounding=grounding,
    )


# ── Anomaly extraction ───────────────────────────────────────────────────────

def extract_anomaly(
    data: pd.DataFrame,
    anomaly_result: AnomalyResult,
    question: str = "Analyze the detected anomalies and their implications.",
    *,
    wise_config: Optional[WISEConfig] = None,
    llm_config: Optional[KexLLMConfig] = None,
    collection_name: str = "",
    persona_context: Optional[str] = None,
    response_length: str = "medium",
    verify: bool = True,
) -> Insight:
    """Extract anomaly insights from anomaly detection results."""
    cfg = llm_config or KexLLMConfig(task_type="anomaly")
    cfg.task_type = "anomaly"

    # 1. Sample with anomaly bias
    sampled = anomaly_aware_sampling(data, max_rows=8000)

    # 2. Build anomaly context
    anomaly_context = _build_anomaly_context(anomaly_result, data)

    # 3. Fingerprint with extra anomaly info
    severity_dist = _severity_distribution(anomaly_result)
    anomaly_count = sum(1 for s in anomaly_result.scores if s.severity != SeverityLevel.NORMAL)
    fingerprint = compute_fingerprint(
        sampled,
        mode="anomaly",
        extra={
            "severity_distribution": severity_dist,
            "total_anomalies": anomaly_count,
            "total_records": len(anomaly_result.scores),
            "anomaly_percentage": round(anomaly_count / max(len(anomaly_result.scores), 1) * 100, 1),
        },
    )

    # 4. Build prompt
    prompt = build_pillar_prompt(
        data=sampled,
        user_question=question,
        pillar=InsightPillar.ANOMALY,
        wise_config=wise_config,
        persona_context=persona_context,
        response_length=response_length,
        collection_name=collection_name,
        pillar_context=anomaly_context,
    )

    # 5. Generate
    start = time.time()
    text, model_name, provider_name = kex_llm.generate(prompt, cfg)
    gen_time = time.time() - start

    # 6. Grounding
    grounding = verify_grounding(text, fingerprint) if verify else None
    status = InsightStatus.VERIFIED if grounding and grounding.grounding_score >= 0.5 else InsightStatus.GENERATED

    return Insight(
        pillar=InsightPillar.ANOMALY,
        text=text,
        status=status,
        model_used=model_name,
        provider_used=provider_name,
        prompt_length=len(prompt),
        generation_time_s=gen_time,
        wise_config=wise_config,
        fingerprint=fingerprint,
        grounding=grounding,
    )


def _severity_distribution(anomaly_result: AnomalyResult) -> Dict[str, int]:
    """Count scores per severity level."""
    dist: Dict[str, int] = {}
    for score in anomaly_result.scores:
        key = score.severity.value
        dist[key] = dist.get(key, 0) + 1
    return dist


def _build_anomaly_context(anomaly_result: AnomalyResult, data: pd.DataFrame) -> str:
    """Build anomaly-specific context for the LLM prompt."""
    lines = ["[ANOMALY DETECTION CONTEXT]"]

    # Severity distribution
    severity_dist = _severity_distribution(anomaly_result)
    lines.append("\n**Severity Distribution:**")
    for level in ["critical", "high", "medium", "low", "normal"]:
        count = severity_dist.get(level, 0)
        if count > 0:
            lines.append(f"- {level.upper()}: {count}")

    total = len(anomaly_result.scores)
    anomaly_count = sum(1 for s in anomaly_result.scores if s.severity != SeverityLevel.NORMAL)
    lines.append(f"\nTotal records: {total}")
    lines.append(f"Anomalies detected: {anomaly_count} ({anomaly_count / max(total, 1) * 100:.1f}%)")

    # Top features from high-severity scores
    high_scores = [s for s in anomaly_result.scores
                   if s.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH)]
    if high_scores:
        feature_importance: Dict[str, float] = {}
        for score in high_scores:
            if score.top_features:
                for f in score.top_features:
                    name = f.get("feature", f.get("name", "unknown"))
                    imp = f.get("importance", f.get("score", 0))
                    feature_importance[name] = feature_importance.get(name, 0) + imp

        if feature_importance:
            lines.append("\n**Top Contributing Features (HIGH/CRITICAL):**")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for fname, imp in sorted_features[:5]:
                lines.append(f"- {fname}: importance={imp:.3f}")

    # Outlier summary
    outlier_summary = build_outlier_summary(data)
    if outlier_summary:
        lines.append(f"\n{outlier_summary}")

    return "\n".join(lines)


# ── Forecast extraction ──────────────────────────────────────────────────────

def extract_forecast(
    data: pd.DataFrame,
    forecast_results: List[ForecastResult],
    question: str = "Analyze the forecast results and their implications.",
    *,
    wise_config: Optional[WISEConfig] = None,
    llm_config: Optional[KexLLMConfig] = None,
    collection_name: str = "",
    persona_context: Optional[str] = None,
    response_length: str = "medium",
    verify: bool = True,
) -> Insight:
    """Extract forecast insights from forecast model results."""
    cfg = llm_config or KexLLMConfig(task_type="forecast")
    cfg.task_type = "forecast"

    # 1. Build forecast context
    forecast_context, model_predictions, consensus = _build_forecast_context(forecast_results, data)

    # 2. Fingerprint
    fingerprint = compute_fingerprint(
        data,
        mode="forecast",
        extra={
            "model_predictions": model_predictions,
            "consensus_mean": consensus.get("mean", 0),
            "model_count": len(forecast_results),
            "agreement_score": consensus.get("agreement_pct", 0),
        },
    )

    # 3. Build prompt
    prompt = build_pillar_prompt(
        data=data,
        user_question=question,
        pillar=InsightPillar.FORECAST,
        wise_config=wise_config,
        persona_context=persona_context,
        response_length=response_length,
        collection_name=collection_name,
        pillar_context=forecast_context,
    )

    # 4. Generate
    start = time.time()
    text, model_name, provider_name = kex_llm.generate(prompt, cfg)
    gen_time = time.time() - start

    # 5. Grounding
    grounding = verify_grounding(text, fingerprint) if verify else None
    status = InsightStatus.VERIFIED if grounding and grounding.grounding_score >= 0.5 else InsightStatus.GENERATED

    return Insight(
        pillar=InsightPillar.FORECAST,
        text=text,
        status=status,
        model_used=model_name,
        provider_used=provider_name,
        prompt_length=len(prompt),
        generation_time_s=gen_time,
        wise_config=wise_config,
        fingerprint=fingerprint,
        grounding=grounding,
    )


def _build_forecast_context(
    forecast_results: List[ForecastResult],
    data: pd.DataFrame,
) -> tuple:
    """Build forecast-specific context and extract model predictions.

    Returns:
        (context_string, model_predictions_dict, consensus_dict)
    """
    lines = ["[FORECAST ANALYSIS CONTEXT]"]
    model_predictions: Dict[str, Dict[str, float]] = {}

    for fr in forecast_results:
        model_name = fr.model_name
        preds = fr.predictions
        if not preds:
            continue

        values = [p["value"] for p in preds if "value" in p]
        if not values:
            continue

        import numpy as np
        stats = {
            "mean": round(float(np.mean(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "std": round(float(np.std(values)), 4) if len(values) > 1 else 0.0,
        }
        model_predictions[model_name] = stats

        lines.append(f"\n**Model: {model_name}**")
        lines.append(f"  Predictions: {len(values)} steps")
        lines.append(f"  Mean: {stats['mean']:.4f}, Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        # Error metrics if available
        if hasattr(fr, "metrics") and fr.metrics:
            for metric_name, metric_val in fr.metrics.items():
                lines.append(f"  {metric_name}: {metric_val:.4f}")

    # Consensus metrics
    consensus: Dict[str, float] = {}
    if model_predictions:
        all_means = [s["mean"] for s in model_predictions.values()]
        import numpy as np
        consensus["mean"] = round(float(np.mean(all_means)), 4)
        spread = max(all_means) - min(all_means) if len(all_means) > 1 else 0
        consensus["spread"] = round(spread, 4)
        if len(all_means) > 1 and consensus["mean"] != 0:
            agreement = max(0, 100 - (spread / abs(consensus["mean"]) * 100))
        else:
            agreement = 100.0
        consensus["agreement_pct"] = round(agreement, 1)

        lines.append(f"\n**Consensus:**")
        lines.append(f"  Mean of means: {consensus['mean']:.4f}")
        lines.append(f"  Model spread: {consensus['spread']:.4f}")
        lines.append(f"  Agreement: {consensus['agreement_pct']:.1f}%")

    # Trend from historical data
    numeric_cols = data.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        from omnikex.profiler import compute_trend_label
        lines.append("\n**Historical Trends:**")
        for col in numeric_cols[:5]:
            s = data[col].dropna()
            if len(s) >= 8:
                trend = compute_trend_label(s)
                lines.append(f"  {col}: {trend}")

    return "\n".join(lines), model_predictions, consensus
