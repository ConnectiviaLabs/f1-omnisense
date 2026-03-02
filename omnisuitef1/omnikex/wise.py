"""WISE prompt building and anti-fabrication rules.

Ported from DataSense:
  - wise_kex_framework.WISEKnowledgeExtractor
  - PromptRegistry fallback template
  - Anti-fabrication grounding rules
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from omnikex._types import (
    InsightPillar,
    PERSONA_TEMPERATURES,
    DEFAULT_TEMPERATURE,
    WISEConfig,
    WISEWeights,
)
from omnikex.profiler import (
    build_data_context,
    build_stats_table,
    compute_trend_label,
    find_entity_columns,
    profile_data,
)


# ── Anti-fabrication rules ───────────────────────────────────────────────────

GROUNDING_RULE = (
    "CRITICAL GROUNDING RULES:\n"
    "1. Every numeric value you cite MUST come from the Column Statistics table above.\n"
    "2. Do NOT invent confidence percentages, p-values, or correlation coefficients.\n"
    "3. Do NOT compute percentage changes from min/max ranges — use the Trend column.\n"
    "4. Do NOT fabricate entity names, sensor IDs, or group labels not in the data.\n"
    "5. If the data is insufficient to answer, say 'insufficient data' explicitly.\n"
    "6. Do NOT mention AI model names, LLM names, or machine learning model names.\n"
    "7. Focus exclusively on the data, results, patterns, and actionable insights.\n"
)

DIRECT_ANSWER_RULE = (
    "Your ENTIRE response must directly answer the above question. "
    "Do NOT provide a generic data overview. If the question asks for a ranking, "
    "provide a ranked list. If it asks for predictions, provide predictions."
)

LLM_GUARDRAIL = (
    "IMPORTANT: Never mention AI model names, LLM names, or machine learning model names "
    "(e.g. Gemma, GPT, Llama, Qwen, BERT, etc.) in your response. "
    "Do not discuss the AI system, ML pipelines, or how the analysis was generated. "
    "Focus exclusively on the data, results, patterns, and actionable business insights.\n\n"
)

WISE_TEMPLATE = """[WISE KNOWLEDGE EXTRACTION]

{persona_role}

{dataset_context}

## Data Profile
{data_profile}

{grounding_rule}

## Question
{user_question}

{direct_answer_rule}

## Analysis Framework (WISE)
Structure your response using this framework:

### 1. Direct Answer
Answer the question directly with specific numbers from the data profile above.

### 2. Evidence & Patterns (Infer)
Surface hidden patterns, correlations, and trends visible in the data.

### 3. Structured Summary (Show)
Present key findings in a clear table format.

### 4. Verification
Cross-reference your claims against the Column Statistics table.

### 5. Decision Support (Weight)
Score findings by business impact and actionability.

### 6. Action Plan (Exercise)
Provide 2-3 actionable options with measurable triggers and thresholds.

{context}

Response length: {response_length}
"""


# ── Prompt building ──────────────────────────────────────────────────────────

def build_wise_prompt(
    data: pd.DataFrame,
    user_question: str,
    *,
    pillar: InsightPillar = InsightPillar.REALTIME,
    wise_config: Optional[WISEConfig] = None,
    persona_context: Optional[str] = None,
    response_length: str = "medium",
    collection_name: str = "",
) -> str:
    """Build a WISE-framework prompt with inline stats and grounding rules."""
    length_guidance = {
        "short": "3-4 paragraphs",
        "medium": "5-6 paragraphs",
        "long": "comprehensive report",
    }

    # Build persona role
    if persona_context:
        persona_role = (
            f"{persona_context}\n\n"
            f"Apply the WISE framework (Weight, Infer, Show, Exercise) to structure your response "
            f"according to the role instructions above."
        )
    else:
        persona_role = "You are an expert data analyst. Extract actionable insights using the WISE framework."

    # Build data profile with inline stats table
    profile_lines = [
        f"Dataset: {collection_name or 'unnamed'}",
        f"Rows: {len(data):,}",
        f"Columns: {len(data.columns)}",
    ]

    # Data quality
    total_cells = len(data) * len(data.columns)
    if total_cells > 0:
        missing = int(data.isnull().sum().sum())
        completeness = round((1 - missing / total_cells) * 100, 2)
        profile_lines.append(f"Completeness: {completeness}%")

    # Entity columns
    entity_cols = find_entity_columns(data)
    if entity_cols:
        profile_lines.append(f"Entity columns (grouping keys): {', '.join(entity_cols)}")

    # Stats table
    stats_table = build_stats_table(data)
    if stats_table:
        profile_lines.append("")
        profile_lines.append(stats_table)

    # Categorical + datetime info
    cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()[:5]
    dt_cols = data.select_dtypes(include="datetime").columns.tolist()
    if cat_cols:
        profile_lines.append(f"Categorical: {', '.join(cat_cols)}")
    if dt_cols:
        profile_lines.append(f"Time Columns: {', '.join(dt_cols)}")

    data_profile_str = "\n".join(profile_lines)

    # Build emphasis instruction if dynamic weights
    emphasis_instruction = ""
    if wise_config and wise_config.is_dynamic:
        emphasis_instruction = build_emphasis_instruction(wise_config.wise_weights)

    # Dataset context from column names
    all_cols = data.columns.tolist()[:20]
    dataset_context = f"Columns: {', '.join(all_cols)}"

    # Context block
    context_parts = []
    if emphasis_instruction:
        context_parts.append(f"[RL-OPTIMIZED WISE FRAMEWORK]\n\n{emphasis_instruction}")
    if wise_config and wise_config.additional_instructions:
        context_parts.append(wise_config.additional_instructions)

    # Detect opportunities
    opportunities = detect_insight_opportunities(data, user_question)
    if opportunities:
        context_parts.append(f"Focus areas: {', '.join(opp.replace('_', ' ') for opp in opportunities[:4])}")

    context_str = "\n\n".join(context_parts)

    return WISE_TEMPLATE.format(
        persona_role=persona_role,
        dataset_context=dataset_context,
        data_profile=data_profile_str,
        grounding_rule=GROUNDING_RULE,
        user_question=user_question,
        direct_answer_rule=DIRECT_ANSWER_RULE,
        context=context_str,
        response_length=length_guidance.get(response_length, length_guidance["medium"]),
    )


def build_emphasis_instruction(wise_weights: WISEWeights) -> str:
    """Build instruction text from WISE emphasis weights."""
    w = wise_weights
    items = [
        ("W (Weight) - Score by business impact", w.weight),
        ("I (Infer) - Surface hidden patterns", w.infer),
        ("S (Show) - Present in tables with confidence", w.show),
        ("E (Exercise) - Provide decision options", w.exercise),
    ]
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

    lines = ["**WISE Framework Emphasis (RL-Optimized):**"]
    for name, value in sorted_items:
        pct = int(value * 100)
        level = "PRIMARY" if value >= 0.3 else "Secondary" if value >= 0.2 else "Light"
        lines.append(f"- {name}: {pct}% ({level})")

    return "\n".join(lines)


def build_pillar_prompt(
    data: pd.DataFrame,
    user_question: str,
    *,
    pillar: InsightPillar = InsightPillar.REALTIME,
    wise_config: Optional[WISEConfig] = None,
    persona_context: Optional[str] = None,
    response_length: str = "medium",
    collection_name: str = "",
    pillar_context: str = "",
    max_sample_rows: int = 30,
) -> str:
    """Build a complete KeX prompt: WISE base + data context + pillar context.

    Every pillar (realtime, anomaly, forecast) should call this function.
    """
    # 1. WISE base prompt (data profile + stats + grounding rules)
    base = build_wise_prompt(
        data=data,
        user_question=user_question,
        pillar=pillar,
        wise_config=wise_config,
        persona_context=persona_context,
        response_length=response_length,
        collection_name=collection_name,
    )

    # 2. Structured data context (per-group stats + sample)
    data_context = build_data_context(data, max_sample_rows=max_sample_rows)

    # 3-4. Assemble
    sections = [base, data_context]
    if pillar_context:
        sections.append(pillar_context)
    sections.append("BEGIN ANALYSIS:\n")

    return "\n\n".join(sections)


def detect_insight_opportunities(data: pd.DataFrame, question: str) -> List[str]:
    """Detect what types of insights the LLM should focus on."""
    opportunities: List[str] = []
    q = question.lower()

    dt_cols = data.select_dtypes(include="datetime").columns
    if len(dt_cols) > 0 or any(kw in q for kw in ["trend", "time", "when", "pattern", "over time"]):
        opportunities.append("temporal_patterns")

    numeric_cols = data.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        opportunities.append("statistical_analysis")
        if len(numeric_cols) >= 2:
            opportunities.append("correlations")

    cat_cols = data.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        opportunities.append("segmentation")

    if any(kw in q for kw in ["anomaly", "outlier", "unusual", "abnormal", "spike"]):
        opportunities.append("anomaly_detection")

    if any(kw in q for kw in ["compare", "difference", "vs", "versus", "between"]):
        opportunities.append("comparative_analysis")

    if any(kw in q for kw in ["predict", "forecast", "future", "trend", "will"]):
        opportunities.append("predictive_insights")

    if any(kw in q for kw in ["why", "cause", "reason", "explain"]):
        opportunities.append("causal_analysis")

    if any(kw in q for kw in ["optimize", "improve", "reduce", "increase", "maximize", "minimize"]):
        opportunities.append("optimization_opportunities")

    return opportunities


def get_persona_temperature(persona_context: Optional[str]) -> float:
    """Return the appropriate LLM temperature for a given persona."""
    if not persona_context:
        return DEFAULT_TEMPERATURE
    ctx_lower = persona_context.lower()
    for key, temp in PERSONA_TEMPERATURES.items():
        if key in ctx_lower:
            return temp
    return DEFAULT_TEMPERATURE
