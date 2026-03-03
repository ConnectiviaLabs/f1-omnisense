"""OmniKeX APIRouter — NL insight extraction from analytics data via WISE framework.

Endpoints:
    POST /api/omni/kex/extract               — extract insight from uploaded CSV or driver data
    POST /api/omni/kex/extract/driver/{code}  — extract insight from driver telemetry
    POST /api/omni/kex/report/{driver_code}   — full autonomous extraction from HealthReport
    GET  /api/omni/kex/mclaren-briefing/{year} — McLaren dashboard WISE insights per year
    GET  /api/omni/kex/providers              — list available LLM providers
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/omni/kex", tags=["OmniKeX"])

# ── Lazy singletons ─────────────────────────────────────────────────────

_health_chain_ready = False



def _load_driver_data(driver_code: str):
    """Load and merge car + bio telemetry for a driver."""
    from pipeline.anomaly.run_f1_anomaly import (
        load_car_race_data,
        load_bio_race_data,
        merge_telemetry,
    )
    car_df = load_car_race_data(driver_code)
    bio_df = load_bio_race_data(driver_code)
    return merge_telemetry(car_df, bio_df)


# ── Request/Response models ─────────────────────────────────────────────

class ExtractRequest(BaseModel):
    question: Optional[str] = None
    pillar: Optional[str] = None  # "realtime" | "anomaly" | "forecast"
    provider: Optional[str] = None  # "groq" | "ollama" | "openai" | "anthropic" | "auto"
    persona: Optional[str] = None  # "CEO" | "analyst" | etc.
    response_length: str = "medium"  # "short" | "medium" | "long"
    verify_grounding: bool = True


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/extract/driver/{driver_code}")
def extract_driver(driver_code: str, req: ExtractRequest):
    """Extract NL insights from a driver's telemetry data."""
    from omnikex import extract

    drivers = {"NOR": "Lando Norris", "PIA": "Oscar Piastri"}
    if driver_code not in drivers:
        raise HTTPException(404, f"Unknown driver: {driver_code}")

    df = _load_driver_data(driver_code)

    kwargs = {
        "data": df,
        "question": req.question or f"Analyze {drivers[driver_code]}'s telemetry performance patterns",
        "response_length": req.response_length,
        "verify_grounding": req.verify_grounding,
    }
    if req.pillar:
        from omnikex import InsightPillar
        kwargs["pillar"] = InsightPillar(req.pillar)
    if req.provider:
        from omnikex import LLMProvider
        kwargs["llm_provider"] = LLMProvider(req.provider)
    if req.persona:
        kwargs["persona"] = req.persona

    insight = extract(**kwargs)
    return insight.to_dict()


@router.post("/report/{driver_code}")
def extract_report(driver_code: str, req: ExtractRequest):
    """Full autonomous extraction: health assess → 3 pillars (realtime, anomaly, forecast)."""
    from omnikex import extract_report as _extract_report
    from omnihealth import assess

    drivers = {"NOR": "Lando Norris", "PIA": "Oscar Piastri"}
    if driver_code not in drivers:
        raise HTTPException(404, f"Unknown driver: {driver_code}")

    df = _load_driver_data(driver_code)

    # Build component map matching omni_health_router
    component_map = {
        "Power Unit": [c for c in ["RPM_mean", "RPM_max", "RPM_std", "nGear_mean"] if c in df.columns],
        "Brakes": [c for c in ["Brake_pct", "Speed_mean", "Speed_std"] if c in df.columns],
        "Drivetrain": [c for c in ["Throttle_mean", "Throttle_max", "DRS_pct"] if c in df.columns],
        "Suspension": [c for c in ["Speed_mean", "Speed_max", "Distance_mean"] if c in df.columns],
        "Thermal": [c for c in ["HeartRate_bpm_mean", "CockpitTemp_C_mean", "AirTemp_C_mean", "TrackTemp_C_mean"] if c in df.columns],
        "Electronics": [c for c in ["DRS_pct", "RPM_mean", "nGear_mean"] if c in df.columns],
    }

    health_report = assess(df, component_map, horizon=10)

    kwargs = {
        "data": df,
        "health_report": health_report,
    }
    if req.question:
        kwargs["question"] = req.question
    if req.provider:
        from omnikex import LLMProvider
        kwargs["llm_provider"] = LLMProvider(req.provider)

    result = _extract_report(**kwargs)
    return result.to_dict()


@router.post("/extract")
async def extract_uploaded(
    file: UploadFile = File(...),
    question: str = Form(None),
    pillar: str = Form(None),
    provider: str = Form(None),
    persona: str = Form(None),
    response_length: str = Form("medium"),
):
    """Extract NL insight from an uploaded CSV/JSON file."""
    import io
    import pandas as pd
    from omnikex import extract

    content = await file.read()
    filename = file.filename or "data.csv"

    if filename.endswith(".json"):
        df = pd.read_json(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content))

    kwargs = {
        "data": df,
        "question": question or "Analyze the key patterns and anomalies in this data",
        "response_length": response_length,
        "verify_grounding": True,
    }
    if pillar:
        from omnikex import InsightPillar
        kwargs["pillar"] = InsightPillar(pillar)
    if provider:
        from omnikex import LLMProvider
        kwargs["llm_provider"] = LLMProvider(provider)
    if persona:
        kwargs["persona"] = persona

    insight = extract(**kwargs)
    return insight.to_dict()


def _enrich_nlp(text: str) -> Dict[str, Any]:
    """Run sentiment analysis, NER, and topic modeling on insight text."""
    import re as _re

    # ── Sentiment (keyword-based, tuned for F1 engineering context) ────
    pos_words = {
        "improved", "improvement", "strong", "fastest", "consistent", "gain",
        "excellent", "dominant", "peak", "optimal", "smooth", "reliable",
        "impressive", "competitive", "advantage", "podium", "win", "better",
        "increase", "increased", "positive", "stable", "clean", "efficient",
    }
    neg_words = {
        "slow", "slower", "decline", "declined", "degradation", "issue",
        "inconsistent", "poor", "worst", "deficit", "loss", "lost", "problem",
        "struggle", "struggled", "penalty", "damage", "risk", "gap", "behind",
        "decrease", "decreased", "negative", "unstable", "erratic", "dnf",
    }
    words = set(_re.findall(r"[a-z]+", text.lower()))
    pos_count = len(words & pos_words)
    neg_count = len(words & neg_words)
    total = pos_count + neg_count or 1
    polarity = (pos_count - neg_count) / total  # -1 to +1
    if polarity > 0.15:
        sentiment = {"label": "positive", "score": round(polarity, 2)}
    elif polarity < -0.15:
        sentiment = {"label": "negative", "score": round(polarity, 2)}
    else:
        sentiment = {"label": "neutral", "score": round(polarity, 2)}

    # ── NER (spaCy) ───────────────────────────────────────────────────
    entities: list[Dict[str, str]] = []
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:3000])  # cap for perf
        seen = set()
        for ent in doc.ents:
            key = (ent.text.strip(), ent.label_)
            if key not in seen and len(ent.text.strip()) > 1:
                seen.add(key)
                entities.append({"text": ent.text.strip(), "label": ent.label_})
    except Exception:
        logger.debug("spaCy NER unavailable, skipping")

    # ── Topic modeling (LDA) ────────────────────────────────────────────
    topics: list[str] = []
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        import numpy as np

        # Split into sentences as pseudo-documents for LDA
        sentences = [s.strip() for s in _re.split(r"[.\n]+", text) if len(s.strip()) > 20]
        if len(sentences) >= 3:
            vectorizer = CountVectorizer(
                stop_words="english", max_features=200, ngram_range=(1, 2),
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{2,}\b",
            )
            doc_term = vectorizer.fit_transform(sentences)
            n_topics = min(3, len(sentences))
            lda = LatentDirichletAllocation(
                n_components=n_topics, max_iter=15, random_state=42,
            )
            lda.fit(doc_term)
            feature_names = vectorizer.get_feature_names_out()
            # Collect top words from each topic, deduplicated
            seen: set[str] = set()
            for topic_weights in lda.components_:
                top_idx = topic_weights.argsort()[::-1][:3]
                for idx in top_idx:
                    word = str(feature_names[idx])
                    if word not in seen:
                        seen.add(word)
                        topics.append(word)
            topics = topics[:6]
    except Exception:
        logger.debug("Topic modeling unavailable, skipping")

    return {"sentiment": sentiment, "entities": entities, "topics": topics}


@router.get("/mclaren-briefing/{year}")
def mclaren_briefing(year: int, force: bool = False):
    """Generate WISE insights for McLaren drivers in a given season.

    Persists to MongoDB so each historical year is only generated once.
    Pass ?force=true to regenerate.
    """
    from updater._db import get_db
    from pipeline.anomaly.mongo_loader import load_driver_race_telemetry
    from omnikex.pillars import extract_realtime
    from omnikex._types import KexLLMConfig, LLMProvider

    db = get_db()
    coll = db["kex_briefings"]

    # Check persisted briefing first
    if not force:
        existing = coll.find_one({"year": year})
        if existing:
            existing.pop("_id", None)
            return existing

    # Dynamically detect McLaren drivers for this year
    pipeline = [
        {"$match": {"season": year, "constructor_id": {"$regex": "mclaren", "$options": "i"}}},
        {"$group": {"_id": "$driver_code"}},
    ]
    mclaren_codes = [doc["_id"] for doc in db["jolpica_race_results"].aggregate(pipeline) if doc["_id"]]

    if not mclaren_codes:
        raise HTTPException(404, f"No McLaren drivers found for {year}")

    persona = (
        "You are a McLaren F1 race engineer providing a concise performance briefing. "
        "Focus on pace trends, tyre management, race craft, and areas for improvement. "
        "Be specific with data references."
    )

    insights = []
    for code in sorted(mclaren_codes):
        df = load_driver_race_telemetry(code, year)
        if df.empty:
            continue

        # Realtime pillar extraction (Ollama + ministral-3)
        llm_cfg = KexLLMConfig(
            provider=LLMProvider.OLLAMA,
            model="ministral-3:8b",
            task_type="realtime",
        )
        try:
            insight = extract_realtime(
                df,
                f"Analyze {code}'s {year} season telemetry — pace consistency, speed trap trends, and stint management.",
                llm_config=llm_cfg,
                persona_context=persona,
                response_length="short",
                verify=True,
            )
            grounding_score = insight.grounding.grounding_score if insight.grounding else 0.0
            nlp_meta = _enrich_nlp(insight.text)
            insights.append({
                "pillar": insight.pillar.value,
                "driver": code,
                "text": insight.text,
                "grounding_score": round(grounding_score, 2),
                "model_used": insight.model_used,
                **nlp_meta,
            })
        except Exception:
            logger.exception("WISE realtime extraction failed for %s/%s", code, year)

    now = time.time()
    result = {"year": year, "insights": insights, "generated_at": now}

    # Persist: upsert so only one doc per year
    coll.replace_one({"year": year}, result, upsert=True)

    # Strip _id for JSON response
    result.pop("_id", None)
    return result


@router.get("/providers")
def list_providers():
    """List available LLM providers for insight generation."""
    providers = []
    try:
        if os.getenv("GROQ_API_KEY"):
            providers.append("groq")
    except Exception:
        pass
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.ok:
            providers.append("ollama")
    except Exception:
        pass
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    return {"providers": providers}
