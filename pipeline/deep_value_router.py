"""Deep Value router — Trident convergence reports + Crossover entity similarity.

Trident: On-demand 4-section reports synthesized from KeX, anomaly, and forecast data.
Crossover: Entity similarity matrix and clustering using VectorProfiles / VictoryProfiles embeddings.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/deep-value", tags=["DeepValue"])

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


def _llm_synthesize(prompt: str, system: str = "") -> str:
    """Call Groq to synthesize text."""
    groq = _get_groq()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    completion = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )
    return completion.choices[0].message.content or ""


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


def _gather_kex_data(db, scope: str, entity: str | None) -> str:
    """Gather KeX briefing data for Key Insights section."""
    parts = []

    # Grid-wide McLaren briefings
    for doc in db["kex_briefings"].find({}, {"_id": 0}).sort("year", -1).limit(2):
        text = doc.get("text") or doc.get("briefing") or doc.get("summary", "")
        if text:
            parts.append(f"[{doc.get('year', '?')} Briefing] {text[:500]}")

    # Per-driver briefings
    filt = {}
    if scope == "driver" and entity:
        filt["driver_code"] = entity.upper()
    for doc in db["kex_driver_briefings"].find(filt, {"_id": 0}).sort("generated_at", -1).limit(10):
        text = doc.get("text") or doc.get("summary", "")
        if text:
            parts.append(f"[{doc.get('driver_code', '?')}] {text[:300]}")

    return "\n\n".join(parts) if parts else "No KeX briefing data available."


def _gather_anomaly_data(db, scope: str, entity: str | None) -> str:
    """Gather anomaly data for Anomaly Patterns section."""
    parts = []

    # Live snapshot
    snapshot = db["anomaly_scores_snapshot"].find_one({}, {"_id": 0}) or {}
    drivers = snapshot.get("drivers", [])
    if scope == "driver" and entity:
        drivers = [d for d in drivers if d.get("driver_code", "").upper() == entity.upper()]
    elif scope == "team" and entity:
        drivers = [d for d in drivers if entity.lower() in (d.get("team", "") or "").lower()]

    for d in drivers[:10]:
        code = d.get("driver_code", "?")
        races = d.get("races", [])
        if races:
            latest = races[-1]
            systems = latest.get("systems", {})
            critical = [s for s, v in systems.items() if v.get("level") == "critical"]
            warning = [s for s, v in systems.items() if v.get("level") == "warning"]
            parts.append(
                f"[{code}] Race: {latest.get('race', '?')} — "
                f"Critical: {critical or 'none'}, Warning: {warning or 'none'}"
            )

    # Anomaly briefings
    filt = {}
    if scope == "driver" and entity:
        filt["driver_code"] = entity.upper()
    for doc in db["kex_anomaly_briefings"].find(filt, {"_id": 0}).limit(5):
        text = doc.get("text") or doc.get("summary", "")
        if text:
            parts.append(f"[Anomaly Brief: {doc.get('driver_code', '?')}] {text[:300]}")

    return "\n\n".join(parts) if parts else "No anomaly data available."


def _gather_forecast_data(db, scope: str, entity: str | None) -> str:
    """Gather forecast/trend data for Forecast Signals section."""
    parts = []

    # Telemetry race summaries — recent trends
    filt = {}
    if scope == "driver" and entity:
        filt["driver_code"] = entity.upper()
    elif scope == "team" and entity:
        filt["team"] = {"$regex": entity, "$options": "i"}

    for doc in db["telemetry_race_summary"].find(filt, {"_id": 0}).sort("year", -1).limit(20):
        code = doc.get("driver_code", "?")
        race = doc.get("race", "?")
        year = doc.get("year", "?")
        metrics = {k: v for k, v in doc.items()
                   if isinstance(v, (int, float)) and k not in ("year", "_id")}
        top_metrics = dict(list(metrics.items())[:6])
        parts.append(f"[{code} {race} {year}] {top_metrics}")

    return "\n\n".join(parts) if parts else "No forecast/trend data available."


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

    # Gather raw data
    kex_raw = _gather_kex_data(db, scope, entity)
    anomaly_raw = _gather_anomaly_data(db, scope, entity)
    forecast_raw = _gather_forecast_data(db, scope, entity)

    scope_label = f"the entire F1 grid" if scope == "grid" else f"{entity}"

    # Synthesize 3 content sections
    key_insights = _llm_synthesize(
        f"Based on the following KeX briefings and driver intelligence for {scope_label}, "
        f"write 3-5 key insights about current performance patterns:\n\n{kex_raw}",
        TRIDENT_SYSTEM,
    )

    anomaly_patterns = _llm_synthesize(
        f"Based on the following anomaly detection data for {scope_label}, "
        f"identify 3-5 significant anomaly patterns and health trends:\n\n{anomaly_raw}",
        TRIDENT_SYSTEM,
    )

    forecast_signals = _llm_synthesize(
        f"Based on the following telemetry trends and race data for {scope_label}, "
        f"identify 3-5 forecast signals and predicted performance trends:\n\n{forecast_raw}",
        TRIDENT_SYSTEM,
    )

    # Synthesize recommendations from the 3 sections
    recommendations = _llm_synthesize(
        f"Based on these three intelligence pillars for {scope_label}, "
        f"provide 3-5 actionable strategic recommendations:\n\n"
        f"KEY INSIGHTS:\n{key_insights}\n\n"
        f"ANOMALY PATTERNS:\n{anomaly_patterns}\n\n"
        f"FORECAST SIGNALS:\n{forecast_signals}",
        TRIDENT_SYSTEM,
    )

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


def _fetch_embeddings(db, source: str, entity_type: str, season: int | None) -> tuple[list[str], list[str], np.ndarray]:
    """Fetch entity labels, teams, and embedding matrix from a collection.

    Returns (labels, teams, embedding_matrix).
    """
    conf = COLLECTION_MAP.get(source)
    if not conf:
        raise HTTPException(400, f"Unknown source: {source}")

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

    clusters = []
    for c in range(n_clusters):
        members = [labels[i] for i in range(n) if cluster_labels[i] == c]
        centroid = kmeans.cluster_centers_[c].tolist()
        clusters.append({"id": c, "members": members, "centroid": centroid})

    return _sanitize({
        "entity_type": body.entity_type,
        "entities": entities,
        "clusters": clusters,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "source": body.source,
    })


@router.post("/crossover/insight")
async def crossover_insight(body: CrossoverInsightRequest):
    """LLM synthesis of cross-entity similarity patterns."""
    db = _get_db()
    labels, teams, matrix = _fetch_embeddings(db, "VectorProfiles", body.entity_type, body.season)

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

    prompt = (
        f"Analyze these F1 {body.entity_type} similarity patterns:\n\n"
        f"MOST SIMILAR PAIRS:\n"
        + "\n".join(f"  {p['a']} ↔ {p['b']}: {p['score']}" for p in most_similar)
        + f"\n\nMOST DISSIMILAR PAIRS:\n"
        + "\n".join(f"  {p['a']} ↔ {p['b']}: {p['score']}" for p in most_dissimilar)
        + f"\n\nAll {body.entity_type}s analyzed: {', '.join(labels)}"
        + f"\n\nExplain the groupings: why are certain {body.entity_type}s similar? "
        f"What driving style, performance, or strategic characteristics do they share? "
        f"What makes the outliers distinct?"
    )

    insight = _llm_synthesize(prompt, TRIDENT_SYSTEM)

    return _sanitize({
        "insight": insight,
        "model_used": GROQ_MODEL,
        "entity_type": body.entity_type,
        "count": n,
        "pairs": {"most_similar": most_similar, "most_dissimilar": most_dissimilar},
    })
