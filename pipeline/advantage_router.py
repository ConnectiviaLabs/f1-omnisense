"""Advantage router — Trident convergence reports + Crossover entity similarity.

Trident: On-demand 4-section reports synthesized from KeX, anomaly, and forecast data.
Crossover: Entity similarity matrix and clustering using VectorProfiles / VictoryProfiles embeddings.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/advantage", tags=["Advantage"])

GROQ_MODEL = os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile")
STALE_SECONDS = 1800  # 30 minutes


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_db():
    from pipeline.chat_server import get_data_db
    return get_data_db()


def _get_groq():
    from pipeline.chat_server import get_groq
    return get_groq()


def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


def _sanitize(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.floating):
        return _sanitize(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def _truncate_prompt(prompt: str, max_chars: int = 12000) -> str:
    """Truncate prompt to stay within LLM context limits."""
    if len(prompt) <= max_chars:
        return prompt
    return prompt[:max_chars] + "\n\n[... truncated for context limit ...]"


def _llm_synthesize(prompt: str, system: str = "") -> str:
    """Call Groq to synthesize text."""
    groq = _get_groq()
    prompt = _truncate_prompt(prompt)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        completion = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        raise HTTPException(502, f"LLM provider error: {str(e)[:200]}")


# ═══════════════════════════════════════════════════════════════════════
#  TRIDENT — Convergence Reports
# ═══════════════════════════════════════════════════════════════════════

TRIDENT_SYSTEM = (
    "You are an elite F1 intelligence analyst for McLaren. "
    "Synthesize data into confident, actionable insights. "
    "Do NOT include disclaimers, caveats, or limitations. "
    "Present all findings directly. Use specific numbers and driver codes."
)


class TridentGenerateRequest(BaseModel):
    scope: str = "grid"
    entity: Optional[str] = None
    force: bool = False


class TridentFeedbackRequest(BaseModel):
    report_id: str
    section: str
    rating: str  # "up" or "down"
    comment: Optional[str] = None


class TridentDbSynthesisRequest(BaseModel):
    force: bool = False


from pipeline.advantage_data import (
    gather_anomaly_data,
    gather_forecast_data,
    gather_kex_data,
    gather_structured_context,
)


@router.post("/trident/generate")
async def trident_generate(body: TridentGenerateRequest):
    """Generate a Trident convergence report."""
    db = _get_db()
    scope = body.scope
    entity = body.entity
    now = time.time()

    # Check cache
    if not body.force:
        cached = db["trident_reports"].find_one(
            {"scope": scope, "entity": entity, "stale_after": {"$gt": now}},
            {"_id": 0},
        )
        if cached:
            cached["from_cache"] = True
            return _sanitize(cached)

    t0 = time.time()

    # Gather raw data + structured metrics
    structured = gather_structured_context(db, scope, entity)
    kex_raw = gather_kex_data(db, scope, entity)
    anomaly_raw = gather_anomaly_data(db, scope, entity)
    forecast_raw = gather_forecast_data(db, scope, entity)

    scope_label = "the entire F1 grid" if scope == "grid" else f"{entity}"
    metrics_block = f"\n\nSTRUCTURED METRICS:\n{structured}" if structured else ""

    # Build prompts for the 3 content sections — each gets structured metrics for comparative reasoning
    key_insights_prompt = (
        f"Based on the following KeX briefings and driver intelligence for {scope_label}, "
        f"write 3-5 key insights about current performance patterns. "
        f"Use the structured metrics to make specific numerical comparisons "
        f"(e.g. 'McLaren's undercut aggression of 0.70 vs Red Bull's 0.45 means...').\n\n"
        f"{kex_raw}{metrics_block}"
    )
    anomaly_prompt = (
        f"Based on the following anomaly detection data and car health metrics for {scope_label}, "
        f"identify 3-5 significant anomaly patterns and health trends. "
        f"Reference specific system health scores and compare across teams/drivers where possible.\n\n"
        f"{anomaly_raw}{metrics_block}"
    )
    forecast_prompt = (
        f"Based on the following telemetry trends and strategy data for {scope_label}, "
        f"identify 3-5 forecast signals and predicted performance trends. "
        f"Use the structured strategy and telemetry metrics to support predictions with data.\n\n"
        f"{forecast_raw}{metrics_block}"
    )

    # Synthesize 3 content sections in parallel
    key_insights, anomaly_patterns, forecast_signals = await asyncio.gather(
        asyncio.to_thread(_llm_synthesize, key_insights_prompt, TRIDENT_SYSTEM),
        asyncio.to_thread(_llm_synthesize, anomaly_prompt, TRIDENT_SYSTEM),
        asyncio.to_thread(_llm_synthesize, forecast_prompt, TRIDENT_SYSTEM),
    )

    # Synthesize recommendations sequentially (depends on the 3 sections above)
    recommendations_prompt = (
        f"Based on these three intelligence pillars for {scope_label}, "
        f"provide 3-5 actionable strategic recommendations:\n\n"
        f"KEY INSIGHTS:\n{key_insights}\n\n"
        f"ANOMALY PATTERNS:\n{anomaly_patterns}\n\n"
        f"FORECAST SIGNALS:\n{forecast_signals}"
    )
    recommendations = await asyncio.to_thread(_llm_synthesize, recommendations_prompt, TRIDENT_SYSTEM)

    gen_time = round(time.time() - t0, 2)
    report_id = f"trident_{scope}_{entity or 'all'}_{int(now)}"

    report = {
        "report_id": report_id,
        "scope": scope,
        "entity": entity,
        "generated_at": now,
        "stale_after": now + STALE_SECONDS,
        "sections": {
            "key_insights": {"title": "Key Insights", "content": key_insights},
            "recommendations": {"title": "Recommendations", "content": recommendations},
            "anomaly_patterns": {"title": "Anomaly Patterns", "content": anomaly_patterns},
            "forecast_signals": {"title": "Forecast Signals", "content": forecast_signals},
        },
        "metadata": {
            "model_used": GROQ_MODEL,
            "generation_time_s": gen_time,
        },
    }

    # Upsert by scope + entity
    db["trident_reports"].update_one(
        {"scope": scope, "entity": entity},
        {"$set": report},
        upsert=True,
    )
    # Also insert into history
    db["trident_reports_history"].insert_one({**report})

    report.pop("_id", None)
    return _sanitize(report)


@router.get("/trident/latest")
async def trident_latest(scope: str = "grid", entity: str | None = None):
    """Get the latest Trident report for a scope."""
    db = _get_db()
    doc = db["trident_reports"].find_one(
        {"scope": scope, "entity": entity},
        {"_id": 0},
    )
    if not doc:
        raise HTTPException(404, "No Trident report found for this scope")
    return _sanitize(doc)


@router.get("/trident/history")
async def trident_history(scope: str = "grid", entity: str | None = None, limit: int = 10):
    """List past Trident reports (metadata only)."""
    db = _get_db()
    cursor = db["trident_reports_history"].find(
        {"scope": scope, "entity": entity},
        {"_id": 0, "sections": 0},
    ).sort("generated_at", -1).limit(limit)
    return _sanitize(list(cursor))


DB_STALE_SECONDS = 600  # 10 min cache for database-wide reports


@router.post("/trident/database-synthesis")
async def trident_database_synthesis(body: TridentDbSynthesisRequest):
    """Cross-collection synthesis report scanning all intelligence collections."""
    db = _get_db()
    now = time.time()

    if not body.force:
        cached = db["trident_reports"].find_one(
            {"scope": "database", "stale_after": {"$gt": now}},
            {"_id": 0},
        )
        if cached:
            cached["from_cache"] = True
            return _sanitize(cached)

    t0 = time.time()

    # Scan across all agent output collections
    agent_collections = [
        "agent_telemetry_anomalies",
        "agent_weather_alerts",
        "agent_pit_windows",
        "agent_knowledge_fused",
        "agent_visual_incidents",
        "agent_predictive_maintenance",
    ]
    cross_data_parts = []
    for coll_name in agent_collections:
        recent = list(db[coll_name].find({}, {"_id": 0}).sort("_timestamp", -1).limit(3))
        if recent:
            summaries = []
            for doc in recent:
                s = doc.get("summary") or doc.get("insight") or doc.get("briefing", "")
                if s:
                    summaries.append(str(s)[:200])
            if summaries:
                cross_data_parts.append(f"[{coll_name}]\n" + "\n".join(summaries))

    # Also pull standard Trident sources
    kex_raw = gather_kex_data(db, "grid", None)
    anomaly_raw = gather_anomaly_data(db, "grid", None)
    forecast_raw = gather_forecast_data(db, "grid", None)

    all_data = (
        f"CROSS-COLLECTION AGENT DATA:\n" + "\n\n".join(cross_data_parts) + "\n\n"
        f"KEX BRIEFINGS:\n{kex_raw}\n\n"
        f"ANOMALY DATA:\n{anomaly_raw}\n\n"
        f"FORECAST DATA:\n{forecast_raw}"
    )

    key_insights_prompt = (
        "Synthesize the following cross-collection data from the entire McLaren intelligence platform. "
        "Identify 3-5 key insights that emerge from correlating data across multiple collections.\n\n"
        f"{all_data}"
    )
    anomaly_prompt = (
        "From the following cross-collection data, identify 3-5 anomaly patterns "
        "that span multiple data sources or correlate across collections.\n\n"
        f"{all_data}"
    )
    forecast_prompt = (
        "From the following cross-collection data, identify 3-5 forecast signals "
        "that are supported by evidence from multiple data sources.\n\n"
        f"{all_data}"
    )

    key_insights, anomaly_patterns, forecast_signals = await asyncio.gather(
        asyncio.to_thread(_llm_synthesize, key_insights_prompt, TRIDENT_SYSTEM),
        asyncio.to_thread(_llm_synthesize, anomaly_prompt, TRIDENT_SYSTEM),
        asyncio.to_thread(_llm_synthesize, forecast_prompt, TRIDENT_SYSTEM),
    )

    recommendations_prompt = (
        "Based on these cross-collection findings, provide 3-5 actionable recommendations:\n\n"
        f"KEY INSIGHTS:\n{key_insights}\n\n"
        f"ANOMALY PATTERNS:\n{anomaly_patterns}\n\n"
        f"FORECAST SIGNALS:\n{forecast_signals}"
    )
    recommendations = await asyncio.to_thread(_llm_synthesize, recommendations_prompt, TRIDENT_SYSTEM)

    gen_time = round(time.time() - t0, 2)
    report_id = f"trident_database_all_{int(now)}"

    report = {
        "report_id": report_id,
        "scope": "database",
        "entity": None,
        "generated_at": now,
        "stale_after": now + DB_STALE_SECONDS,
        "sections": {
            "key_insights": {"title": "Key Insights", "content": key_insights},
            "recommendations": {"title": "Recommendations", "content": recommendations},
            "anomaly_patterns": {"title": "Anomaly Patterns", "content": anomaly_patterns},
            "forecast_signals": {"title": "Forecast Signals", "content": forecast_signals},
        },
        "metadata": {
            "model_used": GROQ_MODEL,
            "generation_time_s": gen_time,
            "collections_scanned": len(agent_collections),
        },
    }

    db["trident_reports"].update_one(
        {"scope": "database"},
        {"$set": report},
        upsert=True,
    )
    db["trident_reports_history"].insert_one({**report})

    report.pop("_id", None)
    return _sanitize(report)


@router.post("/trident/feedback")
async def trident_submit_feedback(body: TridentFeedbackRequest):
    """Submit user feedback for a Trident report section."""
    db = _get_db()
    doc = {
        "report_id": body.report_id,
        "section": body.section,
        "rating": body.rating,
        "comment": body.comment,
        "created_at": time.time(),
    }
    db["trident_report_feedback"].insert_one(doc)
    return {"status": "ok"}


@router.get("/trident/feedback/{report_id}")
async def trident_get_feedback(report_id: str):
    """Get all feedback for a specific report."""
    db = _get_db()
    docs = list(db["trident_report_feedback"].find(
        {"report_id": report_id}, {"_id": 0}
    ))
    return {"feedback": docs, "count": len(docs)}


@router.get("/trident/report/{report_id}")
async def trident_report(report_id: str):
    """Get a specific historical report."""
    db = _get_db()
    doc = db["trident_reports_history"].find_one(
        {"report_id": report_id},
        {"_id": 0},
    )
    if not doc:
        raise HTTPException(404, f"Report {report_id} not found")
    return _sanitize(doc)


# ═══════════════════════════════════════════════════════════════════════
#  CROSSOVER — Entity Similarity
# ═══════════════════════════════════════════════════════════════════════

COLLECTION_MAP = {
    "VectorProfiles": {"key": "driver_code", "team_key": "team", "embed_key": "embedding"},
    "victory_driver_profiles": {"key": "driver_code", "team_key": "team", "embed_key": "embedding"},
    "victory_car_profiles": {"key": "team", "team_key": "team", "embed_key": "embedding"},
    "victory_team_kb": {"key": "team", "team_key": "team", "embed_key": "embedding"},
}

# Metric paths to extract from each source for feature attribution
METRIC_PATHS = {
    "VectorProfiles": {
        "throttle_smoothness": "metrics.throttle_smoothness",
        "late_race_pace": "metrics.late_race_pace",
        "top_speed": "metrics.top_speed",
        "consistency": "metrics.consistency",
        "overtake_success_rate": "metrics.overtake_success_rate",
        "defensive_rating": "metrics.defensive_rating",
    },
    "victory_driver_profiles": {
        "overall_health": "health.overall_health",
        "power_unit": "health.systems.Power Unit",
        "brakes": "health.systems.Brakes",
        "drivetrain": "health.systems.Drivetrain",
        "suspension": "health.systems.Suspension",
        "thermal": "health.systems.Thermal",
        "electronics": "health.systems.Electronics",
        "tyre_mgmt": "health.systems.Tyre Management",
    },
    "victory_car_profiles": {
        "overall_health": "health.overall_health",
        "avg_speed": "telemetry.avg_speed",
        "avg_throttle": "telemetry.avg_throttle",
        "brake_pct": "telemetry.brake_pct",
        "total_wins": "constructor.total_wins",
        "total_points": "constructor.total_points",
        "avg_finish": "constructor.avg_finish_position",
        "dnf_rate": "constructor.dnf_rate",
        "power_unit": "health.systems.Power Unit",
        "brakes": "health.systems.Brakes",
        "suspension": "health.systems.Suspension",
        "thermal": "health.systems.Thermal",
    },
    "victory_team_kb": {
        "overall_health": "metadata.car.health.overall_health",
        "avg_speed": "metadata.car.telemetry.avg_speed",
        "total_wins": "metadata.car.constructor.total_wins",
        "total_points": "metadata.car.constructor.total_points",
        "avg_finish": "metadata.car.constructor.avg_finish_position",
        "dnf_rate": "metadata.car.constructor.dnf_rate",
        "undercut_aggression": "metadata.strategy.team_undercut_aggression",
        "one_stop_freq": "metadata.strategy.team_one_stop_freq",
        "avg_tyre_life": "metadata.strategy.team_avg_tyre_life",
    },
}


def _extract_nested(doc: dict, path: str):
    """Extract a value from a nested dict using dot-notation path."""
    parts = path.split(".")
    current = doc
    for p in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(p)
    return current if isinstance(current, (int, float)) else None


def _fetch_metrics(db, source: str, labels: list[str], key_field: str, season: int | None) -> dict[str, dict[str, float]]:
    """Fetch structured metrics for each entity label. Returns {label: {metric: value}}."""
    paths = METRIC_PATHS.get(source, {})
    if not paths:
        return {}

    filt: dict = {}
    if season is not None:
        filt["season"] = season

    docs = list(db[source].find(filt, {"_id": 0, "embedding": 0, "narrative": 0}))
    doc_map = {d.get(key_field, "?"): d for d in docs}

    result = {}
    for label in labels:
        doc = doc_map.get(label, {})
        metrics = {}
        for name, path in paths.items():
            val = _extract_nested(doc, path)
            if val is not None:
                metrics[name] = float(val)
        result[label] = metrics
    return result


class CrossoverMatrixRequest(BaseModel):
    entity_type: str = "driver"
    season: Optional[int] = None
    source: str = "VectorProfiles"


class CrossoverClusterRequest(BaseModel):
    entity_type: str = "driver"
    season: Optional[int] = None
    n_clusters: int = 4
    source: str = "VectorProfiles"


class CrossoverInsightRequest(BaseModel):
    entity_type: str = "driver"
    entities: list[str] = []
    season: Optional[int] = None
    source: str = "VectorProfiles"


def _fetch_embeddings(db, source: str, entity_type: str, season: int | None) -> tuple[list[str], list[str], np.ndarray]:
    """Fetch entity labels, teams, and embedding matrix from a collection.

    Returns (labels, teams, embedding_matrix).
    """
    conf = COLLECTION_MAP.get(source)
    if not conf:
        raise HTTPException(400, f"Unknown source: {source}")

    # Validate entity_type against source
    if source in ("victory_car_profiles", "victory_team_kb") and entity_type not in ("team", "car"):
        raise HTTPException(400, f"Source {source} does not support entity_type '{entity_type}'")

    filt: dict = {}
    if season is not None:
        filt["season"] = season

    projection = {"_id": 0, conf["key"]: 1, conf["embed_key"]: 1}
    if conf.get("team_key"):
        projection[conf["team_key"]] = 1

    docs = list(db[source].find(filt, projection))
    docs = [d for d in docs if conf["embed_key"] in d]

    if not docs:
        raise HTTPException(404, f"No embeddings found in {source}")

    if entity_type == "team" and source == "VectorProfiles":
        # Average driver embeddings per team
        from collections import defaultdict
        team_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
        for d in docs:
            team = d.get(conf["team_key"], "Unknown")
            team_vecs[team].append(np.array(d[conf["embed_key"]]))
        labels = sorted(team_vecs.keys())
        teams = labels
        matrix = np.array([np.mean(team_vecs[t], axis=0) for t in labels])
    else:
        labels = [d.get(conf["key"], "?") for d in docs]
        teams = [d.get(conf["team_key"], "") for d in docs]
        matrix = np.array([d[conf["embed_key"]] for d in docs])

    # Normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    return labels, teams, matrix


@router.post("/crossover/matrix")
async def crossover_matrix(body: CrossoverMatrixRequest):
    """Compute full cosine similarity matrix for entities."""
    db = _get_db()
    labels, teams, matrix = _fetch_embeddings(db, body.source, body.entity_type, body.season)

    # NxN cosine similarity (already normalized)
    sim = matrix @ matrix.T

    return _sanitize({
        "entity_type": body.entity_type,
        "entities": labels,
        "teams": teams,
        "matrix": sim.tolist(),
        "count": len(labels),
        "source": body.source,
    })


@router.post("/crossover/cluster")
async def crossover_cluster(body: CrossoverClusterRequest):
    """PCA 3D projection + KMeans clustering."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    db = _get_db()
    labels, teams, matrix = _fetch_embeddings(db, body.source, body.entity_type, body.season)

    n = len(labels)
    n_components = min(3, n, matrix.shape[1])
    n_clusters = min(body.n_clusters, n)

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)

    entities = []
    for i in range(n):
        entry = {
            "code": labels[i],
            "team": teams[i] if i < len(teams) else "",
            "x": float(coords[i][0]) if n_components > 0 else 0,
            "y": float(coords[i][1]) if n_components > 1 else 0,
            "z": float(coords[i][2]) if n_components > 2 else 0,
            "cluster": int(cluster_labels[i]),
        }
        entities.append(entry)

    # Fetch structured metrics for feature attribution
    conf = COLLECTION_MAP[body.source]
    entity_metrics = _fetch_metrics(db, body.source, labels, conf["key"], body.season)

    clusters = []
    for c in range(n_clusters):
        members = [labels[i] for i in range(n) if cluster_labels[i] == c]
        centroid = kmeans.cluster_centers_[c].tolist()
        clusters.append({"id": c, "members": members, "centroid": centroid})

    # Compute per-cluster metric averages and find discriminating features
    cluster_profiles: list[dict] = []
    all_metric_names = set()
    for c_info in clusters:
        member_metrics = [entity_metrics.get(m, {}) for m in c_info["members"]]
        avg_metrics: dict[str, float] = {}
        for mm in member_metrics:
            for k, v in mm.items():
                all_metric_names.add(k)
                avg_metrics.setdefault(k, []).append(v)  # type: ignore
        avg_metrics = {k: float(np.mean(v)) for k, v in avg_metrics.items()}  # type: ignore
        cluster_profiles.append(avg_metrics)

    # Discriminating features: highest variance of cluster means across clusters
    discriminators: list[dict] = []
    for metric in all_metric_names:
        cluster_means = [cp.get(metric) for cp in cluster_profiles if metric in cp]
        if len(cluster_means) >= 2:
            spread = float(np.std(cluster_means))
            mean_val = float(np.mean(cluster_means))
            # Normalize spread by mean to get relative discrimination power
            rel_spread = spread / abs(mean_val) if mean_val != 0 else spread
            discriminators.append({
                "metric": metric,
                "spread": round(rel_spread, 4),
                "cluster_values": {f"C{i}": round(cp.get(metric, 0), 2) for i, cp in enumerate(cluster_profiles)},
            })
    discriminators.sort(key=lambda x: x["spread"], reverse=True)

    return _sanitize({
        "entity_type": body.entity_type,
        "entities": entities,
        "clusters": clusters,
        "cluster_profiles": cluster_profiles,
        "discriminators": discriminators[:10],  # top 10 discriminating features
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "source": body.source,
    })


@router.post("/crossover/insight")
async def crossover_insight(body: CrossoverInsightRequest):
    """LLM synthesis of cross-entity similarity patterns."""
    db = _get_db()
    labels, teams, matrix = _fetch_embeddings(db, body.source, body.entity_type, body.season)

    # Filter to requested entities if specified
    if body.entities:
        upper = [e.upper() for e in body.entities]
        indices = [i for i, l in enumerate(labels) if l.upper() in upper]
        if len(indices) < 2:
            raise HTTPException(400, "Need at least 2 entities for comparison")
        labels = [labels[i] for i in indices]
        teams = [teams[i] for i in indices]
        matrix = matrix[indices]

    sim = matrix @ matrix.T
    n = len(labels)

    # Find top-5 most and least similar pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((labels[i], labels[j], float(sim[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)

    most_similar = [{"a": p[0], "b": p[1], "score": round(p[2], 4)} for p in pairs[:5]]
    most_dissimilar = [{"a": p[0], "b": p[1], "score": round(p[2], 4)} for p in pairs[-5:]]

    # Fetch structured metrics so the LLM can explain WHY pairs are similar/different
    conf = COLLECTION_MAP[body.source]
    entity_metrics = _fetch_metrics(db, body.source, labels, conf["key"], body.season)

    metrics_block = "\nSTRUCTURED METRICS PER ENTITY:"
    for label in labels:
        m = entity_metrics.get(label, {})
        if m:
            metrics_str = ", ".join(f"{k}={v:.2f}" for k, v in m.items())
            metrics_block += f"\n  {label}: {metrics_str}"

    prompt = (
        f"Analyze these F1 {body.entity_type} similarity patterns:\n\n"
        f"MOST SIMILAR PAIRS:\n"
        + "\n".join(f"  {p['a']} ↔ {p['b']}: {p['score']}" for p in most_similar)
        + f"\n\nMOST DISSIMILAR PAIRS:\n"
        + "\n".join(f"  {p['a']} ↔ {p['b']}: {p['score']}" for p in most_dissimilar)
        + f"\n{metrics_block}"
        + f"\n\nUsing the structured metrics above, explain specifically which metrics drive the similarity/dissimilarity. "
        f"For each pair, identify the 2-3 metrics where they converge (similar) or diverge (dissimilar). "
        f"Be precise — cite the actual numbers."
    )

    insight = _llm_synthesize(prompt, TRIDENT_SYSTEM)

    return _sanitize({
        "insight": insight,
        "model_used": GROQ_MODEL,
        "entity_type": body.entity_type,
        "count": n,
        "pairs": {"most_similar": most_similar, "most_dissimilar": most_dissimilar},
    })


# ═══════════════════════════════════════════════════════════════════════
#  CROSSOVER — Compare + Cross-Entity Intelligence
# ═══════════════════════════════════════════════════════════════════════

class CrossoverCompareRequest(BaseModel):
    entity_type: str = "driver"
    entities: list[str]
    source: str = "VectorProfiles"
    season: Optional[int] = None


class CrossoverCrossInsightRequest(BaseModel):
    entity_type: str = "driver"
    entities: list[str]
    query: str
    source: str = "VectorProfiles"
    season: Optional[int] = None


def _generate_suggested_questions(
    entity_type: str,
    entities: list[str],
    highest_pair: dict | None,
    lowest_pair: dict | None,
) -> list[str]:
    """Generate 8 deterministic suggested questions from comparison data."""
    a_hi = highest_pair["a"] if highest_pair else entities[0]
    b_hi = highest_pair["b"] if highest_pair else entities[-1]
    a_lo = lowest_pair["a"] if lowest_pair else entities[-1]
    b_lo = lowest_pair["b"] if lowest_pair else entities[0]
    elist = ", ".join(entities[:4]) + ("..." if len(entities) > 4 else "")

    return [
        f"What makes {a_hi} and {b_hi} so similar?",
        f"What fundamentally differentiates {a_lo} from {b_lo}?",
        f"What anomalies exist between {entities[0]} and {entities[-1]}?",
        f"How do performance trends in {a_hi} correlate with {b_hi}?",
        f"Which strategic patterns are shared across {elist}?",
        f"Where does {a_lo} diverge from the group?",
        f"What {entity_type} traits unite the most similar pair?",
        f"How would swapping {a_hi}'s approach benefit {b_lo}?",
    ]


def _find_metric_correlations(
    entity_metrics: dict[str, dict[str, float]],
    pairs: list[dict],
) -> list[dict]:
    """Identify converging/diverging metrics for each pair."""
    all_vals: dict[str, list[float]] = {}
    for m in entity_metrics.values():
        for k, v in m.items():
            all_vals.setdefault(k, []).append(v)
    ranges = {k: (max(vs) - min(vs)) if len(vs) > 1 else 1.0 for k, vs in all_vals.items()}

    correlations = []
    for pair in pairs:
        a_m = entity_metrics.get(pair["a"], {})
        b_m = entity_metrics.get(pair["b"], {})
        converging, diverging = [], []
        for k in set(a_m) & set(b_m):
            r = ranges.get(k, 1.0) or 1.0
            diff = abs(a_m[k] - b_m[k]) / r
            if diff <= 0.10:
                converging.append(k)
            elif diff >= 0.50:
                diverging.append(k)
        correlations.append({
            "pair": [pair["a"], pair["b"]],
            "converging": converging[:5],
            "diverging": diverging[:5],
        })
    return correlations


@router.get("/crossover/entities")
def crossover_entities(
    entity_type: str = "driver",
    source: str = "VectorProfiles",
    season: int | None = None,
):
    """Lightweight endpoint returning available entity labels."""
    db = _get_db()
    if source not in COLLECTION_MAP:
        raise HTTPException(400, f"Unknown source: {source}")
    conf = COLLECTION_MAP[source]
    key = conf["key"]
    filt: dict = {}
    if season is not None:
        filt["season"] = season
    docs = list(db[source].find(filt, {"_id": 0, key: 1, conf["team_key"]: 1}))
    entities = []
    seen = set()
    for d in docs:
        label = d.get(key, "")
        if label and label not in seen:
            seen.add(label)
            entities.append({"code": label, "team": d.get(conf["team_key"], "")})
    entities.sort(key=lambda x: x["code"])
    return {"entities": entities, "entity_type": entity_type, "source": source}


@router.post("/crossover/compare")
def crossover_compare(body: CrossoverCompareRequest):
    """Pairwise comparison of 2-8 selected entities."""
    if not (2 <= len(body.entities) <= 8):
        raise HTTPException(400, "Select 2-8 entities to compare")

    db = _get_db()
    if body.source not in COLLECTION_MAP:
        raise HTTPException(400, f"Unknown source: {body.source}")

    labels, teams, matrix = _fetch_embeddings(db, body.source, body.entity_type, body.season)
    if len(labels) == 0:
        raise HTTPException(404, "No embeddings found")

    # Filter to requested entities
    idx_map = {l: i for i, l in enumerate(labels)}
    valid = [e for e in body.entities if e in idx_map]
    if len(valid) < 2:
        raise HTTPException(400, f"Only {len(valid)} of {len(body.entities)} entities found in {body.source}")

    idxs = [idx_map[e] for e in valid]
    sub = matrix[np.ix_(idxs, idxs)]
    sim = sub @ sub.T

    # Fetch structured metrics
    conf = COLLECTION_MAP[body.source]
    entity_metrics = _fetch_metrics(db, body.source, valid, conf["key"], body.season)

    # Build pairwise results
    pairs = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            pairs.append({
                "a": valid[i],
                "b": valid[j],
                "similarity": float(sim[i, j]),
                "a_metrics": entity_metrics.get(valid[i], {}),
                "b_metrics": entity_metrics.get(valid[j], {}),
            })
    pairs.sort(key=lambda x: x["similarity"], reverse=True)

    sims = [p["similarity"] for p in pairs]
    highest = {"a": pairs[0]["a"], "b": pairs[0]["b"], "similarity": pairs[0]["similarity"]} if pairs else None
    lowest = {"a": pairs[-1]["a"], "b": pairs[-1]["b"], "similarity": pairs[-1]["similarity"]} if pairs else None

    questions = _generate_suggested_questions(body.entity_type, valid, highest, lowest)

    return _sanitize({
        "pairs": pairs,
        "statistics": {
            "avg": float(np.mean(sims)) if sims else 0,
            "max": float(max(sims)) if sims else 0,
            "min": float(min(sims)) if sims else 0,
        },
        "highest_pair": highest,
        "lowest_pair": lowest,
        "entity_type": body.entity_type,
        "source": body.source,
        "entities": valid,
        "suggested_questions": questions,
    })


@router.post("/crossover/cross_insights")
def crossover_cross_insights(body: CrossoverCrossInsightRequest):
    """LLM synthesis with entity context + semantic metric correlations."""
    if len(body.entities) < 2:
        raise HTTPException(400, "Need at least 2 entities")

    db = _get_db()
    if body.source not in COLLECTION_MAP:
        raise HTTPException(400, f"Unknown source: {body.source}")

    labels, teams, matrix = _fetch_embeddings(db, body.source, body.entity_type, body.season)
    idx_map = {l: i for i, l in enumerate(labels)}
    valid = [e for e in body.entities if e in idx_map]
    if len(valid) < 2:
        raise HTTPException(400, f"Only {len(valid)} entities found")

    # Compute pairwise similarities
    idxs = [idx_map[e] for e in valid]
    sub = matrix[np.ix_(idxs, idxs)]
    sim = sub @ sub.T

    pair_list = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            pair_list.append({"a": valid[i], "b": valid[j], "similarity": float(sim[i, j])})

    # Fetch metrics + correlations
    conf = COLLECTION_MAP[body.source]
    entity_metrics = _fetch_metrics(db, body.source, valid, conf["key"], body.season)
    correlations = _find_metric_correlations(entity_metrics, pair_list)

    # Build enhanced prompt
    metrics_block = "\nENTITY METRICS:"
    for label in valid:
        m = entity_metrics.get(label, {})
        if m:
            metrics_str = ", ".join(f"{k}={v:.2f}" for k, v in m.items())
            metrics_block += f"\n  {label}: {metrics_str}"

    sim_block = "\nPAIRWISE SIMILARITY:"
    for p in pair_list:
        sim_block += f"\n  {p['a']} ↔ {p['b']}: {p['similarity']:.4f}"

    corr_block = "\nMETRIC CORRELATIONS:"
    for c in correlations:
        pair_label = f"{c['pair'][0]} ↔ {c['pair'][1]}"
        if c["converging"]:
            corr_block += f"\n  {pair_label} converge on: {', '.join(c['converging'])}"
        if c["diverging"]:
            corr_block += f"\n  {pair_label} diverge on: {', '.join(c['diverging'])}"

    prompt = (
        f"USER QUESTION: {body.query}\n\n"
        f"CONTEXT: Comparing these F1 {body.entity_type}s: {', '.join(valid)}\n"
        f"{metrics_block}\n{sim_block}\n{corr_block}\n\n"
        f"Answer the user's question using the metrics, similarity scores, and correlations above. "
        f"Cite specific numbers. Be precise and insightful."
    )

    insight = _llm_synthesize(prompt, TRIDENT_SYSTEM)

    # Store to shared context
    from datetime import datetime
    from uuid import uuid4
    db["crossover_insights"].insert_one({
        "uid": str(uuid4()),
        "agent": "cross-entity-analysis",
        "entities": valid,
        "query": body.query,
        "insight": insight,
        "llm_provider": "groq",
        "model_used": GROQ_MODEL,
        "entity_type": body.entity_type,
        "source": body.source,
        "generated_at": datetime.utcnow(),
    })

    # Generate suggested questions for follow-up
    pair_list.sort(key=lambda x: x["similarity"], reverse=True)
    highest = pair_list[0] if pair_list else None
    lowest = pair_list[-1] if pair_list else None
    questions = _generate_suggested_questions(body.entity_type, valid, highest, lowest)

    return _sanitize({
        "insight": insight,
        "model_used": GROQ_MODEL,
        "entities": valid,
        "query": body.query,
        "correlations_found": correlations,
        "suggested_questions": questions,
    })


# ══════════════════════════════════════════════════════════════════════════
# Prep Mode — Pre-Race Intelligence Package
# ══════════════════════════════════════════════════════════════════════════


class PrepModeRequest(BaseModel):
    """Request body for generating a pre-race intelligence package."""
    race_name: str                  # e.g. "Austrian Grand Prix"
    season: int = 2024
    team: str = "McLaren"
    drivers: list[str] = []        # e.g. ["NOR", "PIA"] — empty = auto-detect
    force: bool = False


def _gather_prep_anomaly(db, drivers: list[str], race_name: str) -> dict:
    """Gather anomaly health data for target drivers."""
    results = {}
    snapshot = db["anomaly_scores_snapshot"].find_one({}, {"_id": 0})
    if not snapshot:
        return results

    for d in snapshot.get("drivers", []):
        code = d.get("code", "")
        if code not in drivers:
            continue
        systems = {}
        for race in d.get("races", []):
            for sys_name, sys_data in race.get("systems", {}).items():
                systems[sys_name] = {
                    "health": sys_data.get("health", 0),
                    "level": sys_data.get("level", "unknown"),
                }
        results[code] = {
            "overall_health": d.get("overall_health", 0),
            "overall_level": d.get("overall_level", "unknown"),
            "last_race": d.get("last_race", ""),
            "race_count": d.get("race_count", 0),
            "systems": systems,
        }
    return results


def _gather_prep_elt(db, race_name: str, season: int, drivers: list[str]) -> dict:
    """Load ELT pace predictions for each driver."""
    results = {}
    for driver in drivers:
        bl = db["elt_parameters"].find_one(
            {"type": "circuit_baseline", "circuit": race_name},
            {"_id": 0},
            sort=[("year", -1)],
        )
        if not bl:
            continue
        dd = db["elt_parameters"].find_one(
            {"type": "driver_year_delta", "driver": driver, "year": season},
            {"_id": 0},
        )
        if not dd:
            dd = db["elt_parameters"].find_one(
                {"type": "driver_delta", "driver": driver},
                {"_id": 0},
            )
        delta = dd["avg_delta"] if dd else 0.0
        results[driver] = {
            "baseline_pace_s": round(bl["baseline_lap_time"], 3),
            "driver_advantage_s": round(delta, 3),
            "predicted_pace_s": round(bl["baseline_lap_time"] + delta, 3),
        }
    return results


def _gather_prep_degradation(db, race_name: str) -> list[dict]:
    """Load tyre degradation curves for the circuit."""
    curves = []
    for doc in db["tyre_degradation_curves"].find(
        {"circuit": race_name}, {"_id": 0}
    ):
        curves.append({
            "compound": doc.get("compound", ""),
            "temp_band": doc.get("temp_band", "all"),
            "coefficients": doc.get("coefficients", []),
            "deg_per_lap_s": doc.get("deg_per_lap_s"),
            "r_squared": doc.get("r_squared"),
            "n_stints": doc.get("n_stints"),
        })
    return curves


def _gather_prep_strategy(db, race_name: str) -> dict | None:
    """Load pre-computed strategy simulation for the circuit."""
    doc = db["race_strategy_simulations"].find_one(
        {"race_name": {"$regex": race_name, "$options": "i"}},
        {"_id": 0},
    )
    return doc


def _gather_prep_circuit(db, race_name: str) -> dict:
    """Load circuit intelligence and pit loss data."""
    circuit = {}
    ci = db["circuit_intelligence"].find_one(
        {"race_name": {"$regex": race_name, "$options": "i"}},
        {"_id": 0},
    )
    if ci:
        circuit["intelligence"] = ci

    pit = db["circuit_pit_loss_times"].find_one(
        {"circuit": {"$regex": race_name.lower().replace(" grand prix", ""), "$options": "i"}},
        {"_id": 0},
    )
    if pit:
        circuit["pit_loss_s"] = pit.get("pit_loss_s") or pit.get("avg_pit_loss_s")

    air = db["race_air_density"].find_one(
        {"race_name": {"$regex": race_name, "$options": "i"}},
        {"_id": 0},
    )
    if air:
        circuit["air_density"] = {
            "density_kg_m3": air.get("density_kg_m3"),
            "temperature_c": air.get("temperature_c"),
            "humidity_pct": air.get("humidity_pct"),
            "pressure_hpa": air.get("pressure_hpa"),
        }

    return circuit


@router.post("/prep-mode/generate")
async def prep_mode_generate(body: PrepModeRequest):
    """Generate a pre-race intelligence package.

    Combines anomaly health, ELT pace, tyre degradation, strategy,
    circuit intelligence, and a Trident synthesis into a single report.
    This is the Wednesday briefing for the strategy group.
    """
    db = _get_db()
    t0 = time.time()

    race_name = body.race_name
    season = body.season
    team = body.team

    # Auto-detect drivers if not provided
    drivers = body.drivers
    if not drivers:
        drivers = sorted(db["jolpica_race_results"].distinct(
            "driver_code",
            {"season": season, "constructor_name": {"$regex": team, "$options": "i"}},
        ))
    if not drivers:
        raise HTTPException(400, f"No drivers found for {team} in {season}")

    # Gather all data pillars in parallel
    anomaly_data = _gather_prep_anomaly(db, drivers, race_name)
    elt_data = _gather_prep_elt(db, race_name, season, drivers)
    deg_curves = _gather_prep_degradation(db, race_name)
    strategy_sim = _gather_prep_strategy(db, race_name)
    circuit_data = _gather_prep_circuit(db, race_name)

    # Build context for LLM synthesis
    context_parts = [f"PRE-RACE INTELLIGENCE PACKAGE: {race_name} ({season})"]
    context_parts.append(f"Team: {team} | Drivers: {', '.join(drivers)}")

    if anomaly_data:
        context_parts.append("\n--- ANOMALY HEALTH ---")
        for drv, data in anomaly_data.items():
            systems_str = ", ".join(
                f"{s}: {d['health']}/100 ({d['level']})"
                for s, d in data.get("systems", {}).items()
            )
            context_parts.append(
                f"{drv}: Overall {data['overall_health']}/100 ({data['overall_level']}). "
                f"Systems: {systems_str}"
            )

    if elt_data:
        context_parts.append("\n--- PACE PREDICTION (ELT) ---")
        for drv, data in elt_data.items():
            context_parts.append(
                f"{drv}: Predicted {data['predicted_pace_s']:.3f}s "
                f"(baseline {data['baseline_pace_s']:.3f}s, "
                f"advantage {data['driver_advantage_s']:+.3f}s)"
            )

    if deg_curves:
        context_parts.append("\n--- TYRE DEGRADATION CURVES ---")
        for c in deg_curves[:6]:
            deg = c.get("deg_per_lap_s")
            r2 = c.get("r_squared")
            context_parts.append(
                f"{c['compound']} ({c['temp_band']}): "
                f"{deg:.4f} s/lap degradation" + (f" (R²={r2:.3f})" if r2 else "")
                if deg else f"{c['compound']}: no curve data"
            )

    if circuit_data:
        context_parts.append("\n--- CIRCUIT DATA ---")
        if circuit_data.get("pit_loss_s"):
            context_parts.append(f"Pit loss: {circuit_data['pit_loss_s']:.1f}s")
        ad = circuit_data.get("air_density", {})
        if ad.get("temperature_c"):
            context_parts.append(
                f"Conditions: {ad['temperature_c']}°C, "
                f"{ad.get('humidity_pct', '?')}% humidity, "
                f"{ad.get('density_kg_m3', '?')} kg/m³ air density"
            )

    full_context = "\n".join(context_parts)

    # Generate LLM synthesis — pre-race briefing
    briefing_prompt = (
        f"You are a Formula 1 strategy analyst preparing the pre-race intelligence "
        f"briefing for {team}'s strategy group. Based on the following data, write a "
        f"concise pre-race report with these sections:\n\n"
        f"1. RISK ASSESSMENT — Overall risk level for each driver, key concerns\n"
        f"2. PACE OUTLOOK — Expected performance relative to the grid\n"
        f"3. STRATEGY RECOMMENDATION — Pit stop strategy, tyre selection, key windows\n"
        f"4. WATCH ITEMS — Specific things to monitor during FP1/FP2\n\n"
        f"Be specific with numbers. Reference the data. Keep it actionable.\n\n"
        f"{full_context}"
    )

    briefing = await asyncio.to_thread(
        _llm_synthesize, briefing_prompt, TRIDENT_SYSTEM
    )

    gen_time = round(time.time() - t0, 2)

    report = {
        "report_type": "prep_mode",
        "race_name": race_name,
        "season": season,
        "team": team,
        "drivers": drivers,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_time_s": gen_time,
        "pillars": {
            "anomaly": anomaly_data,
            "elt_pace": elt_data,
            "degradation_curves": deg_curves,
            "strategy_simulation": _sanitize(strategy_sim) if strategy_sim else None,
            "circuit": circuit_data,
        },
        "briefing": briefing,
        "model_used": GROQ_MODEL,
    }

    # Store in MongoDB
    db["prep_mode_reports"].update_one(
        {"race_name": race_name, "season": season, "team": team},
        {"$set": report},
        upsert=True,
    )

    return _sanitize(report)


@router.get("/prep-mode/latest")
async def prep_mode_latest(
    race_name: str | None = None,
    season: int = 2024,
    team: str = "McLaren",
):
    """Get the latest prep mode report."""
    db = _get_db()
    filt: dict = {"season": season, "team": {"$regex": team, "$options": "i"}}
    if race_name:
        filt["race_name"] = {"$regex": race_name, "$options": "i"}

    doc = db["prep_mode_reports"].find_one(filt, {"_id": 0}, sort=[("generated_at", -1)])
    if not doc:
        raise HTTPException(404, "No prep mode report found")
    return _sanitize(doc)


@router.get("/prep-mode/list")
async def prep_mode_list(season: int = 2024, team: str = "McLaren"):
    """List available prep mode reports."""
    db = _get_db()
    cursor = db["prep_mode_reports"].find(
        {"season": season, "team": {"$regex": team, "$options": "i"}},
        {"_id": 0, "pillars": 0, "briefing": 0},
    ).sort("generated_at", -1)
    return _sanitize(list(cursor))


# ══════════════════════════════════════════════════════════════════════════
# Strategy Simulator — Interactive Monte Carlo Race Simulation
# ══════════════════════════════════════════════════════════════════════════


class StrategySimRequest(BaseModel):
    """Request body for interactive strategy simulation."""
    race_name: str               # e.g. "Austrian Grand Prix"
    total_laps: int              # race distance
    pit_loss_s: float = 22.0     # pit-lane time loss (seconds)
    fuel_effect_s: float = 0.03  # seconds-per-lap lighter per lap burned
    compounds: list[str] = ["SOFT", "MEDIUM", "HARD"]
    max_stops: int = 3           # explore 1-stop through N-stop


def _load_deg_lookup(db, race_name: str) -> dict[str, float]:
    """Load tyre degradation rates from MongoDB for a given circuit.

    Returns {compound: deg_per_lap_s} mapping.
    Falls back to sensible defaults if no data found.
    """
    lookup: dict[str, float] = {}
    for doc in db["tyre_degradation_curves"].find(
        {"circuit": {"$regex": race_name, "$options": "i"}},
        {"_id": 0, "compound": 1, "deg_per_lap_s": 1},
    ):
        compound = doc.get("compound", "").upper()
        deg = doc.get("deg_per_lap_s")
        if compound and deg is not None:
            lookup[compound] = float(deg)

    # Defaults if no curve data
    defaults = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.035}
    for c, d in defaults.items():
        if c not in lookup:
            lookup[c] = d
    return lookup


def _get_deg(compound: str, deg_lookup: dict[str, float]) -> float:
    """Get degradation rate for a compound."""
    return deg_lookup.get(compound.upper(), 0.05)


def _fuel_effect(lap: int, total_laps: int, fuel_effect_s: float) -> float:
    """Compute fuel-mass time benefit at a given lap.

    Car gets lighter each lap → faster. Linear model.
    """
    return -fuel_effect_s * lap


def _generate_splits(total_laps: int, n_stops: int) -> list[list[int]]:
    """Generate reasonable stint-length splits for n_stops.

    Each stint must be at least 5 laps. Returns a list of splits,
    where each split is a list of stint lengths summing to total_laps.
    """
    if n_stops == 0:
        return [[total_laps]]

    n_stints = n_stops + 1
    min_stint = 5
    if min_stint * n_stints > total_laps:
        return []

    splits = []
    remaining = total_laps - min_stint * n_stints

    # Generate evenly-spaced splits with variations
    base = total_laps // n_stints
    for offset in range(-8, 9, 2):
        split = []
        left = total_laps
        for s in range(n_stints - 1):
            length = max(min_stint, base + offset * (1 if s == 0 else -1 if s == n_stints - 2 else 0))
            length = min(length, left - min_stint * (n_stints - s - 1))
            split.append(length)
            left -= length
        split.append(left)
        if all(s >= min_stint for s in split) and sum(split) == total_laps:
            if split not in splits:
                splits.append(split)

    # Add a few asymmetric splits
    for first_frac in [0.3, 0.4, 0.5, 0.6]:
        split = []
        first_len = max(min_stint, int(total_laps * first_frac))
        left = total_laps - first_len
        split.append(first_len)
        for s in range(n_stints - 2):
            length = max(min_stint, left // (n_stints - 1 - s))
            split.append(length)
            left -= length
        split.append(left)
        if all(s >= min_stint for s in split) and sum(split) == total_laps:
            if split not in splits:
                splits.append(split)

    return splits[:20]  # cap at 20 splits per stop count


def _sim_strategy(
    stint_laps: list[int],
    compounds: list[str],
    deg_lookup: dict[str, float],
    pit_loss_s: float,
    fuel_effect_s: float,
    total_laps: int,
) -> dict:
    """Simulate a single strategy and return total race time + breakdown.

    For each stint, compute lap times as:
      lap_time = baseline + degradation * tyre_age + fuel_effect(lap)

    The baseline is normalized to 0 so we compare relative times.
    """
    total_time = 0.0
    stint_details = []
    global_lap = 0

    for i, (length, compound) in enumerate(zip(stint_laps, compounds)):
        deg = _get_deg(compound, deg_lookup)
        stint_time = 0.0
        for tyre_lap in range(length):
            lap_delta = deg * tyre_lap + _fuel_effect(global_lap, total_laps, fuel_effect_s)
            stint_time += lap_delta
            global_lap += 1

        # Add pit stop time (not on the last stint)
        pit = pit_loss_s if i < len(stint_laps) - 1 else 0.0
        total_time += stint_time + pit

        stint_details.append({
            "stint": i + 1,
            "compound": compound,
            "laps": length,
            "stint_delta_s": round(stint_time, 3),
            "pit_s": pit,
        })

    return {
        "total_delta_s": round(total_time, 3),
        "stops": len(stint_laps) - 1,
        "stints": stint_details,
        "stint_laps": stint_laps,
        "compounds": compounds,
    }


@router.post("/strategy/simulate")
async def strategy_simulate(body: StrategySimRequest):
    """Interactive strategy simulator.

    Runs Monte Carlo-style exploration of all N-stop strategies
    using tyre degradation curves from MongoDB. Returns ranked
    strategies with time deltas relative to the fastest option.
    """
    db = _get_db()
    deg_lookup = _load_deg_lookup(db, body.race_name)

    all_results = []

    for n_stops in range(0, body.max_stops + 1):
        splits = _generate_splits(body.total_laps, n_stops)
        if not splits:
            continue

        # Generate compound permutations for this stop count
        from itertools import product
        n_stints = n_stops + 1
        compound_combos = list(product(body.compounds, repeat=n_stints))
        # Cap to avoid explosion
        if len(compound_combos) > 50:
            compound_combos = compound_combos[:50]

        for split in splits:
            for combo in compound_combos:
                result = _sim_strategy(
                    split, list(combo), deg_lookup,
                    body.pit_loss_s, body.fuel_effect_s, body.total_laps,
                )
                all_results.append(result)

    # Sort by total time
    all_results.sort(key=lambda r: r["total_delta_s"])

    # Compute deltas relative to the best strategy
    best_time = all_results[0]["total_delta_s"] if all_results else 0
    for r in all_results:
        r["gap_to_best_s"] = round(r["total_delta_s"] - best_time, 3)

    # Return top 20 strategies
    top = all_results[:20]

    return _sanitize({
        "race_name": body.race_name,
        "total_laps": body.total_laps,
        "pit_loss_s": body.pit_loss_s,
        "fuel_effect_s": body.fuel_effect_s,
        "deg_rates": deg_lookup,
        "strategies_evaluated": len(all_results),
        "top_strategies": top,
    })
