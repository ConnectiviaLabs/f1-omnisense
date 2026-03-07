"""Advantage router — Trident convergence reports + Crossover entity similarity.

Trident: On-demand 4-section reports synthesized from KeX, anomaly, and forecast data.
Crossover: Entity similarity matrix and clustering using VectorProfiles / VictoryProfiles embeddings.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
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


def _gather_structured_context(db, scope: str, entity: str | None) -> str:
    """Pull structured metrics from Victory collections for comparative reasoning."""
    parts = []

    if scope == "grid" or scope == "team":
        # Team-level comparison table from victory_team_kb
        filt: dict = {}
        if scope == "team" and entity:
            filt["team"] = {"$regex": f"^{entity}$", "$options": "i"}
        teams = list(db["victory_team_kb"].find(filt, {"_id": 0, "embedding": 0, "narrative": 0}).limit(13))
        if teams:
            parts.append("TEAM PERFORMANCE COMPARISON:")
            for t in teams:
                meta = t.get("metadata") or {}
                car = meta.get("car") or {}
                con = (car.get("constructor") or {}) if isinstance(car, dict) else {}
                telem = (car.get("telemetry") or {}) if isinstance(car, dict) else {}
                health = (car.get("health") or {}) if isinstance(car, dict) else {}
                strat = meta.get("strategy") or {}
                parts.append(
                    f"  {t.get('team')}: wins={con.get('total_wins', '?')}, "
                    f"podiums={con.get('total_podiums', '?')}, points={con.get('total_points', '?')}, "
                    f"avg_finish={con.get('avg_finish_position', '?')}, "
                    f"dnf_rate={con.get('dnf_rate', '?')}, "
                    f"avg_speed={telem.get('avg_speed', '?')}kph, "
                    f"overall_health={health.get('overall_health', '?')}%, "
                    f"undercut_aggression={strat.get('team_undercut_aggression', '?')}, "
                    f"one_stop_freq={strat.get('team_one_stop_freq', '?')}, "
                    f"avg_tyre_life={strat.get('team_avg_tyre_life', '?')} laps"
                )

    if scope == "driver" and entity:
        # Driver strategy profile
        sp = db["victory_strategy_profiles"].find_one(
            {"driver_code": entity.upper()}, {"_id": 0, "embedding": 0, "narrative": 0}
        )
        if sp:
            ps = sp.get("pit_strategy", {})
            pe = sp.get("pit_execution", {})
            parts.append(f"DRIVER STRATEGY ({entity.upper()}):")
            parts.append(
                f"  undercut_aggression={ps.get('undercut_aggression', '?')}, "
                f"tyre_extension_bias={ps.get('tyre_extension_bias', '?')}, "
                f"one_stop_freq={ps.get('one_stop_freq', '?'):.3f}, "
                f"avg_first_stop_lap={ps.get('avg_first_stop_lap', '?')}, "
                f"pit_stops={pe.get('total_stops', '?')}, "
                f"avg_pit_duration={pe.get('avg_duration_s', '?')}s, "
                f"best_pit={pe.get('best_duration_s', '?')}s"
            )
            comps = sp.get("compound_profiles", [])
            if comps:
                parts.append("  Compound profiles:")
                for c in comps:
                    parts.append(
                        f"    {c['compound']}: laps={c.get('total_laps', '?')}, "
                        f"avg_lap={c.get('avg_lap_time_s', 0):.2f}s, "
                        f"tyre_life={c.get('avg_tyre_life', 0):.1f} laps"
                    )

        # Compare vs grid (top 5 rivals)
        rivals = list(db["victory_strategy_profiles"].find(
            {"driver_code": {"$ne": entity.upper()}},
            {"_id": 0, "embedding": 0, "narrative": 0, "compound_profiles": 0}
        ).limit(5))
        if rivals:
            parts.append(f"RIVAL STRATEGY COMPARISON (vs {entity.upper()}):")
            for r in rivals:
                rps = r.get("pit_strategy", {})
                rpe = r.get("pit_execution", {})
                parts.append(
                    f"  {r.get('driver_code')}/{r.get('team')}: "
                    f"undercut={rps.get('undercut_aggression', '?')}, "
                    f"one_stop={rps.get('one_stop_freq', '?'):.3f}, "
                    f"avg_pit={rpe.get('avg_duration_s', '?')}s"
                )

    return "\n\n".join(parts) if parts else ""


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
    driver_limit = 5 if scope == "grid" else 10
    for doc in db["kex_driver_briefings"].find(filt, {"_id": 0}).sort("generated_at", -1).limit(driver_limit):
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
        drivers = [d for d in drivers if (d.get("team", "") or "").lower() == entity.lower()]

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
        filt["team"] = {"$regex": f"^{entity}$", "$options": "i"}

    telem_limit = 10 if scope == "grid" else 20
    for doc in db["telemetry_race_summary"].find(filt, {"_id": 0}).sort("year", -1).limit(telem_limit):
        code = doc.get("driver_code", "?")
        race = doc.get("race", "?")
        year = doc.get("year", "?")
        metrics = {k: v for k, v in doc.items()
                   if isinstance(v, (int, float)) and k not in ("year", "_id")}
        top_metrics = dict(list(metrics.items())[:6])
        parts.append(f"[{code} {race} {year}] {top_metrics}")

    # Forecast briefings
    brief_filt = {}
    if scope == "driver" and entity:
        brief_filt["driver_code"] = entity.upper()
    for doc in db["kex_forecast_briefings"].find(brief_filt, {"_id": 0}).limit(5):
        text = doc.get("text") or doc.get("summary", "")
        if text:
            parts.append(f"[Forecast Brief: {doc.get('driver_code', '?')}] {text[:300]}")

    # Car telemetry briefings
    car_filt = {}
    if scope == "driver" and entity:
        car_filt["driver_code"] = entity.upper()
    elif scope == "team" and entity:
        car_filt["team"] = {"$regex": f"^{entity}$", "$options": "i"}
    for doc in db["kex_car_telemetry_briefings"].find(car_filt, {"_id": 0}).limit(5):
        text = doc.get("text") or doc.get("summary", "")
        if text:
            parts.append(f"[Car Telemetry Brief: {doc.get('driver_code', doc.get('team', '?'))}] {text[:300]}")

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

    # Gather raw data + structured metrics
    structured = _gather_structured_context(db, scope, entity)
    kex_raw = _gather_kex_data(db, scope, entity)
    anomaly_raw = _gather_anomaly_data(db, scope, entity)
    forecast_raw = _gather_forecast_data(db, scope, entity)

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
