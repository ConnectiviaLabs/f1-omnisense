"""
Build VectorProfiles Collection
────────────────────────────────
Merges 5 driver data sources into per-driver vector profile documents.
Each document contains structured data, a text narrative, and a 768-dim
Nomic embedding for similarity search and RAG grounding.

Source collections:
  1. driver_performance_markers   (keyed by Driver)
  2. driver_overtake_profiles     (keyed by driver_code)
  3. driver_telemetry_profiles    (keyed by driver_code)
  4. anomaly_scores_snapshot      (single doc with drivers[] array)
  5. kex_driver_briefings         (keyed by driver_code)

Output collection:
  - VectorProfiles (one doc per driver, with embedding field)

Usage:
    python pipeline/build_vector_profiles.py
    python pipeline/build_vector_profiles.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from updater._db import get_db

OUT_COL = "VectorProfiles"
EMBEDDING_DIM = 768


# ── Source fetching ──────────────────────────────────────────────────────


def _collect_driver_codes(db) -> list[str]:
    """Union of driver codes across all 5 source collections (legacy, no season filter)."""
    codes = set()
    codes.update(db["driver_performance_markers"].distinct("Driver"))
    codes.update(db["driver_overtake_profiles"].distinct("driver_code"))
    codes.update(db["driver_telemetry_profiles"].distinct("driver_code"))
    codes.update(c for c in db["opponent_profiles"].distinct("driver_code") if c)

    snapshot = db["anomaly_scores_snapshot"].find_one({}, {"drivers.code": 1})
    if snapshot:
        for d in snapshot.get("drivers", []):
            if d.get("code"):
                codes.add(d["code"])

    return sorted(c for c in codes if c and len(c) == 3)


def _collect_season_driver_codes(db, season: int) -> list[str]:
    """Union of driver codes from season-filtered upstream collections."""
    codes = set()
    codes.update(db["driver_performance_markers"].distinct("Driver", {"season": season}))
    codes.update(db["driver_overtake_profiles"].distinct("driver_code", {"season": season}))
    codes.update(db["driver_telemetry_profiles"].distinct("driver_code", {"season": season}))
    # Also include drivers from anomaly snapshot (current state, always included)
    snapshot = db["anomaly_scores_snapshot"].find_one({}, {"drivers.code": 1})
    if snapshot:
        for d in snapshot.get("drivers", []):
            if d.get("code"):
                codes.add(d["code"])
    return sorted(c for c in codes if c and len(c) == 3)


def _fetch_all_sources(db) -> dict:
    """Bulk fetch all 5 collections into {driver_code: doc} dicts."""
    sources = {}

    # 1. Performance markers (keyed by "Driver")
    sources["performance"] = {
        doc["Driver"]: {k: v for k, v in doc.items() if k not in ("_id", "Driver")}
        for doc in db["driver_performance_markers"].find({}, {"_id": 0})
        if doc.get("Driver")
    }

    # 2. Overtake profiles (keyed by "driver_code")
    sources["overtaking"] = {
        doc["driver_code"]: {k: v for k, v in doc.items() if k not in ("_id", "driver_code")}
        for doc in db["driver_overtake_profiles"].find({}, {"_id": 0})
        if doc.get("driver_code")
    }

    # 3. Telemetry profiles (keyed by "driver_code")
    sources["telemetry"] = {
        doc["driver_code"]: {k: v for k, v in doc.items() if k not in ("_id", "driver_code")}
        for doc in db["driver_telemetry_profiles"].find({}, {"_id": 0})
        if doc.get("driver_code")
    }

    # 4. Anomaly snapshot (single doc → unpack drivers array)
    health = {}
    snapshot = db["anomaly_scores_snapshot"].find_one({}, {"_id": 0})
    if snapshot:
        for d in snapshot.get("drivers", []):
            code = d.get("code")
            if not code:
                continue
            systems = {}
            for race in d.get("races", []):
                for sys_name, sys_data in race.get("systems", {}).items():
                    if sys_name not in systems:
                        systems[sys_name] = sys_data.get("health", 0)
            health[code] = {
                "team": d.get("team", ""),
                "overall_health": d.get("overall_health", 0),
                "overall_level": d.get("overall_level", "unknown"),
                "last_race": d.get("last_race", ""),
                "race_count": d.get("race_count", 0),
                "systems": systems,
            }
    sources["health"] = health

    # 5. KeX driver briefings (keyed by "driver_code")
    sources["briefings"] = {}
    for doc in db["kex_driver_briefings"].find({}, {"_id": 0}):
        code = doc.get("driver_code")
        if code:
            sources["briefings"][code] = {
                "text": (doc.get("text") or "")[:500],
                "sentiment": doc.get("sentiment", {}),
                "topics": doc.get("topics", []),
            }

    # 6. Opponent profiles — 58 numeric metrics per driver (strategy, quali, endurance)
    _OPP_SKIP = {"_id", "driver_id", "driver_code", "forename", "surname",
                  "nationality", "dob", "seasons", "updated_at",
                  "career_stats_updated_at", "jolpica_enriched_at",
                  "driver_number", "age", "preferred_compound"}
    sources["opponent"] = {}
    for doc in db["opponent_profiles"].find({}, {"_id": 0}):
        code = doc.get("driver_code")
        if code:
            sources["opponent"][code] = {
                k: v for k, v in doc.items() if k not in _OPP_SKIP
            }

    # 7. Communication profiles — aggregated radio signal data per driver
    sources["communication"] = {}
    for doc in db["driver_communication_profiles"].find({}, {"_id": 0, "updated_at": 0}):
        code = doc.get("driver_code")
        if code:
            sources["communication"][code] = {
                k: v for k, v in doc.items()
                if k not in ("_id", "driver_code", "driver_name", "driver_number", "season", "updated_at")
            }

    return sources


def _fetch_season_sources(db, season: int) -> dict:
    """Fetch sources filtered to a specific season.

    First tries upstream season-tagged collections (2025+).
    Falls back to computing per-season metrics from raw jolpica/fastf1 data.
    Anomaly snapshot and KeX briefings are snapshot data — included as-is.
    """
    sources = {}

    # 1. Performance markers — try season-filtered upstream first
    sources["performance"] = {
        doc["Driver"]: {k: v for k, v in doc.items() if k not in ("_id", "Driver", "season")}
        for doc in db["driver_performance_markers"].find({"season": season}, {"_id": 0})
        if doc.get("Driver")
    }
    # Fallback: compute from raw jolpica + fastf1 data
    if not sources["performance"]:
        sources["performance"] = _compute_season_performance(db, season)

    # 2. Overtake profiles — try upstream first
    sources["overtaking"] = {
        doc["driver_code"]: {k: v for k, v in doc.items() if k not in ("_id", "driver_code", "season")}
        for doc in db["driver_overtake_profiles"].find({"season": season}, {"_id": 0})
        if doc.get("driver_code")
    }
    if not sources["overtaking"]:
        sources["overtaking"] = _compute_season_overtaking(db, season)

    # 3. Telemetry profiles — try upstream first
    sources["telemetry"] = {
        doc["driver_code"]: {k: v for k, v in doc.items() if k not in ("_id", "driver_code", "season")}
        for doc in db["driver_telemetry_profiles"].find({"season": season}, {"_id": 0})
        if doc.get("driver_code")
    }
    if not sources["telemetry"]:
        sources["telemetry"] = _compute_season_telemetry(db, season)

    # 4. Anomaly snapshot — current state, not season-specific
    health = {}
    snapshot = db["anomaly_scores_snapshot"].find_one({}, {"_id": 0})
    if snapshot:
        for d in snapshot.get("drivers", []):
            code = d.get("code")
            if not code:
                continue
            systems = {}
            for race in d.get("races", []):
                for sys_name, sys_data in race.get("systems", {}).items():
                    if sys_name not in systems:
                        systems[sys_name] = sys_data.get("health", 0)
            health[code] = {
                "team": d.get("team", ""),
                "overall_health": d.get("overall_health", 0),
                "overall_level": d.get("overall_level", "unknown"),
                "last_race": d.get("last_race", ""),
                "race_count": d.get("race_count", 0),
                "systems": systems,
            }
    sources["health"] = health

    # 5. KeX briefings — current state, not season-specific
    sources["briefings"] = {}
    for doc in db["kex_driver_briefings"].find({}, {"_id": 0}):
        code = doc.get("driver_code")
        if code:
            sources["briefings"][code] = {
                "text": (doc.get("text") or "")[:500],
                "sentiment": doc.get("sentiment", {}),
                "topics": doc.get("topics", []),
            }

    # 6. Opponent profiles — career-wide data, not season-specific
    _OPP_SKIP = {"_id", "driver_id", "driver_code", "forename", "surname",
                  "nationality", "dob", "seasons", "updated_at",
                  "career_stats_updated_at", "jolpica_enriched_at",
                  "driver_number", "age", "preferred_compound"}
    sources["opponent"] = {}
    for doc in db["opponent_profiles"].find({}, {"_id": 0}):
        code = doc.get("driver_code")
        if code:
            sources["opponent"][code] = {
                k: v for k, v in doc.items() if k not in _OPP_SKIP
            }

    # 7. Communication profiles — per-driver radio signal aggregates
    sources["communication"] = {}
    for doc in db["driver_communication_profiles"].find(
        {"season": season} if season else {}, {"_id": 0, "updated_at": 0}
    ):
        code = doc.get("driver_code")
        if code:
            sources["communication"][code] = {
                k: v for k, v in doc.items()
                if k not in ("_id", "driver_code", "driver_name", "driver_number", "season", "updated_at")
            }

    return sources


# ── Per-season metric computation from raw data ─────────────────────────


def _to_python(obj):
    """Convert numpy types to Python builtins for MongoDB storage."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


def _compute_season_performance(db, season: int) -> dict:
    """Compute performance markers from jolpica_race_results + fastf1_laps."""
    results = {}
    drivers = db["jolpica_race_results"].distinct("driver_code", {"season": season})

    for driver in drivers:
        # Get race results
        races = list(db["jolpica_race_results"].find(
            {"season": season, "driver_code": driver},
            {"_id": 0, "grid": 1, "position": 1, "points": 1, "status": 1,
             "race_name": 1, "constructor_name": 1},
        ))
        if not races:
            continue

        # Get lap data
        laps = list(db["fastf1_laps"].find(
            {"Year": season, "Driver": driver, "IsAccurate": True},
            {"_id": 0, "LapTime": 1, "Sector1Time": 1, "Sector2Time": 1,
             "Sector3Time": 1, "SpeedST": 1, "TyreLife": 1, "Compound": 1},
        ))

        perf: dict = {}
        perf["races_count"] = len(races)
        perf["avg_grid"] = round(np.mean([
            float(r["grid"]) for r in races
            if r.get("grid") and str(r["grid"]).isdigit()
        ] or [0]), 1)
        perf["avg_position"] = round(np.mean([
            float(r["position"]) for r in races
            if r.get("position") and str(r["position"]).isdigit()
        ] or [0]), 1)
        perf["total_points"] = sum(float(r.get("points", 0) or 0) for r in races)
        perf["team"] = races[0].get("constructor_name", "")

        if laps:
            lap_times = [float(l["LapTime"]) for l in laps if l.get("LapTime")]
            if lap_times:
                perf["avg_lap_time_s"] = round(np.mean(lap_times), 3)
                perf["lap_time_consistency_std"] = round(np.std(lap_times), 3)

            speeds = [float(l["SpeedST"]) for l in laps if l.get("SpeedST")]
            if speeds:
                perf["avg_top_speed_kmh"] = round(np.mean(speeds), 1)

            # Sector CVs
            for si in [1, 2, 3]:
                col = f"Sector{si}Time"
                vals = [float(l[col]) for l in laps if l.get(col)]
                if vals:
                    perf[f"sector{si}_cv"] = round(np.std(vals) / max(np.mean(vals), 0.01), 4)

            # Degradation slope (simple: lap time vs tyre life)
            tyre_data = [(float(l["TyreLife"]), float(l["LapTime"]))
                         for l in laps if l.get("TyreLife") and l.get("LapTime")]
            if len(tyre_data) > 10:
                x = np.array([t[0] for t in tyre_data])
                y = np.array([t[1] for t in tyre_data])
                if np.std(x) > 0:
                    slope = float(np.polyfit(x, y, 1)[0])
                    perf["degradation_slope_s_per_lap"] = round(slope, 4)

        results[driver] = _to_python(perf)
    return results


def _compute_season_overtaking(db, season: int) -> dict:
    """Compute overtake metrics from jolpica_race_results positions_gained."""
    results = {}
    drivers = db["jolpica_race_results"].distinct("driver_code", {"season": season})

    for driver in drivers:
        races = list(db["jolpica_race_results"].find(
            {"season": season, "driver_code": driver},
            {"_id": 0, "grid": 1, "position": 1, "positions_gained": 1},
        ))
        if not races:
            continue

        gains = [
            float(r["positions_gained"])
            for r in races if r.get("positions_gained") is not None
        ]
        if not gains:
            continue

        made = sum(max(0, g) for g in gains)
        lost = sum(abs(min(0, g)) for g in gains)
        n = len(gains)

        results[driver] = {
            "races_analysed": n,
            "total_overtakes_made": int(made),
            "total_times_overtaken": int(lost),
            "overtakes_per_race": round(made / max(n, 1), 1),
            "times_overtaken_per_race": round(lost / max(n, 1), 1),
            "overtake_net": int(made - lost),
            "overtake_ratio": round(made / max(made + lost, 1), 3),
        }
    return results


def _compute_season_telemetry(db, season: int) -> dict:
    """Compute telemetry profile from telemetry_race_summary."""
    results = {}
    for doc in db["telemetry_race_summary"].find({"Year": season}, {"_id": 0}):
        driver = doc.get("Driver")
        if not driver:
            continue
        if driver not in results:
            results[driver] = {"_races": []}
        results[driver]["_races"].append(doc)

    final = {}
    for driver, data in results.items():
        races = data["_races"]
        if not races:
            continue
        prof: dict = {}
        for key in ["avg_speed", "top_speed", "avg_throttle", "brake_pct", "drs_pct"]:
            vals = [float(r[key]) for r in races if r.get(key)]
            if vals:
                if key == "avg_speed":
                    prof["avg_race_speed_kmh"] = round(np.mean(vals), 1)
                elif key == "top_speed":
                    prof["top_speed_kmh"] = round(np.mean(vals), 1)
                elif key == "avg_throttle":
                    prof["avg_throttle_pct"] = round(np.mean(vals), 1)
                elif key == "brake_pct":
                    prof["avg_brake_pct"] = round(np.mean(vals), 1)
                elif key == "drs_pct":
                    prof["drs_usage_ratio"] = round(np.mean(vals) / 100, 3)
        final[driver] = _to_python(prof)
    return final


# ── Per-driver merge ─────────────────────────────────────────────────────


def _merge_driver(code: str, sources: dict) -> dict:
    """Merge one driver's data from all 5 sources."""
    found = []
    doc = {"driver_code": code}

    # Team from health data or performance data
    health = sources["health"].get(code)
    perf_team = (sources["performance"].get(code) or {}).get("team", "")
    doc["team"] = (health["team"] if health else "") or perf_team

    # Performance
    perf = sources["performance"].get(code)
    if perf:
        doc["performance"] = perf
        found.append("performance")
    else:
        doc["performance"] = None

    # Overtaking
    ot = sources["overtaking"].get(code)
    if ot:
        doc["overtaking"] = ot
        found.append("overtaking")
    else:
        doc["overtaking"] = None

    # Telemetry
    tel = sources["telemetry"].get(code)
    if tel:
        doc["telemetry"] = tel
        found.append("telemetry")
    else:
        doc["telemetry"] = None

    # Health
    if health:
        doc["health"] = health
        found.append("health")
    else:
        doc["health"] = None

    # Briefing
    brief = sources["briefings"].get(code)
    if brief:
        doc["briefing_summary"] = brief["text"]
        found.append("briefing")
    else:
        doc["briefing_summary"] = None

    # Opponent profile (strategy, qualifying, endurance — 58 metrics)
    opp = sources.get("opponent", {}).get(code)
    if opp:
        doc["opponent"] = opp
        found.append("opponent")
    else:
        doc["opponent"] = None

    # Communication profile (radio signals)
    comm = sources.get("communication", {}).get(code)
    if comm:
        doc["communication"] = comm
        found.append("communication")
    else:
        doc["communication"] = None

    doc["sources_found"] = found
    return doc


# ── Narrative generation ─────────────────────────────────────────────────


def _q_deg(v):
    if v is None: return "unknown"
    if v < 0.03: return "excellent"
    if v < 0.06: return "moderate"
    return "high"

def _q_ot(v):
    if v is None: return "unknown"
    if v > 0.7: return "dominant"
    if v > 0.5: return "positive"
    return "defensive"

def _q_brake(v):
    if v is None: return "unknown"
    if v > 4.5: return "very aggressive"
    if v > 3.5: return "aggressive"
    return "moderate"

def _q_health(v):
    if v is None: return "unknown"
    if v >= 80: return "healthy"
    if v >= 60: return "concerning"
    return "critical"

def _safe(val, fmt=".3f", suffix=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:{fmt}}{suffix}"


def _build_narrative(doc: dict) -> str:
    """Build natural-language narrative from merged driver data."""
    parts = []
    code = doc["driver_code"]
    team = doc.get("team") or "an F1 team"

    parts.append(f"{code} drives for {team}.")

    # Performance paragraph
    p = doc.get("performance")
    if p:
        deg = p.get("degradation_slope_s_per_lap")
        parts.append(
            f"Race pace: {_q_deg(deg)} tyre degradation at {_safe(deg)} s/lap, "
            f"late-race delta {_safe(p.get('late_race_delta_s'))} s, "
            f"lap consistency std {_safe(p.get('lap_time_consistency_std'))} s. "
            f"Sector CVs: S1={_safe(p.get('sector1_cv'))}, S2={_safe(p.get('sector2_cv'))}, S3={_safe(p.get('sector3_cv'))}. "
            f"Heat sensitivity {_safe(p.get('heat_lap_delta_s'))} s, humidity {_safe(p.get('humidity_lap_delta_s'))} s. "
            f"Top speed {_safe(p.get('avg_top_speed_kmh'), '.1f', ' km/h')}."
        )

    # Overtaking paragraph
    ot = doc.get("overtaking")
    if ot:
        ratio = ot.get("overtake_ratio")
        parts.append(
            f"Overtaking: {_q_ot(ratio)} style with {ot.get('total_overtakes_made', '?')} overtakes made, "
            f"{ot.get('total_times_overtaken', '?')} times overtaken across {ot.get('races_analysed', '?')} races. "
            f"Per race: {_safe(ot.get('overtakes_per_race'), '.1f')} made, "
            f"{_safe(ot.get('times_overtaken_per_race'), '.1f')} lost. "
            f"Net balance {ot.get('overtake_net', '?'):+d}, ratio {_safe(ratio)}."
        )

    # Telemetry paragraph
    t = doc.get("telemetry")
    if t:
        brk = t.get("avg_braking_g")
        parts.append(
            f"Telemetry: {_q_brake(brk)} braking at {_safe(brk)} G avg, {_safe(t.get('max_braking_g'))} G peak. "
            f"Brake-to-throttle transition {_safe(t.get('brake_to_throttle_avg_s'), '.3f', ' s')}. "
            f"Throttle {_safe(t.get('avg_throttle_pct'), '.1f')}% avg, "
            f"full throttle ratio {_safe(t.get('full_throttle_ratio'))}. "
            f"Race speed {_safe(t.get('avg_race_speed_kmh'), '.1f', ' km/h')}, "
            f"top speed {_safe(t.get('top_speed_kmh'), '.1f', ' km/h')}. "
            f"DRS usage {_safe(t.get('drs_usage_ratio'))}, gain {_safe(t.get('drs_speed_gain_kmh'), '.1f', ' km/h')}."
        )

    # Health paragraph
    h = doc.get("health")
    if h:
        oh = h.get("overall_health")
        sys_parts = ", ".join(
            f"{name}: {score}/100" for name, score in h.get("systems", {}).items()
        )
        parts.append(
            f"System health: {_q_health(oh)} at {oh}/100 overall ({h.get('overall_level', '?')}). "
            + (f"Systems: {sys_parts}." if sys_parts else "")
        )

    # Opponent profile (strategy, qualifying, endurance)
    opp = doc.get("opponent")
    if opp:
        parts.append(
            f"Strategy: undercut aggression {_safe(opp.get('undercut_aggression_score'))}, "
            f"tyre extension bias {_safe(opp.get('tyre_extension_bias'))}, "
            f"avg tyre life {_safe(opp.get('avg_tyre_life'), '.1f')} laps. "
            f"Pit stops: 1-stop {_safe(opp.get('one_stop_freq'))}, "
            f"2-stop {_safe(opp.get('two_stop_freq'))}, "
            f"3-stop {_safe(opp.get('three_stop_freq'))}. "
            f"First stop lap {_safe(opp.get('avg_first_stop_lap'), '.1f')}. "
            f"Qualifying: Q3 rate {_safe(opp.get('q3_appearance_rate'))}, "
            f"avg grid P{_safe(opp.get('avg_quali_position'), '.1f')}, "
            f"quali-race delta {_safe(opp.get('quali_race_pace_delta_pct'), '.1f')}%. "
            f"Late race: speed drop {_safe(opp.get('late_race_speed_drop'), '.1f')} km/h, "
            f"degradation {_safe(opp.get('late_race_degradation'))}, "
            f"position loss {_safe(opp.get('late_race_position_loss'), '.1f')}. "
            f"Endurance: long stint {_safe(opp.get('long_stint_capability'))}, "
            f"stint slope {_safe(opp.get('stint_endurance_slope'))}. "
            f"Consistency: braking G-std {_safe(opp.get('g_consistency'))}, "
            f"throttle smoothness {_safe(opp.get('throttle_smoothness'), '.1f')}, "
            f"position volatility {_safe(opp.get('position_volatility'), '.1f')}."
        )

    # Communication profile (radio intelligence)
    comm = doc.get("communication")
    if comm:
        sent = comm.get("sentiment_distribution", {})
        top_signals = ", ".join(comm.get("top_signal_types", [])[:3])
        parts.append(
            f"Radio communication: {comm.get('total_messages', 0)} messages, "
            f"{comm.get('messages_per_session_avg', 0)} per session. "
            f"Sentiment: {_safe(sent.get('negative'), '.0%')} negative, "
            f"{_safe(sent.get('positive'), '.0%')} positive. "
            f"Complaint frequency {_safe(comm.get('complaint_frequency'), '.0%')}. "
            f"Top signals: {top_signals}."
        )

    # Briefing snippet
    b = doc.get("briefing_summary")
    if b:
        parts.append(f"Intelligence briefing: {b}")

    return " ".join(parts)


# ── Embedding ────────────────────────────────────────────────────────────


def _embed_narratives(narratives: list[str]) -> list[list[float]]:
    """Batch embed using NomicEmbedder (768-dim)."""
    from embeddings import NomicEmbedder

    embedder = NomicEmbedder()
    return embedder.embed(narratives)


# ── Similarity search ────────────────────────────────────────────────────


def _season_filter(season: int | None) -> dict:
    """Build MongoDB filter for season: integer season or None for legacy."""
    if season is not None:
        return {"season": season}
    return {"season": None}


def _team_embedding(team: str, db, season: int | None = None) -> np.ndarray | None:
    """Average embedding for all drivers in a team."""
    filt = {"team": team, **_season_filter(season)}
    docs = list(db[OUT_COL].find(filt, {"embedding": 1}))
    vecs = [np.array(d["embedding"]) for d in docs if "embedding" in d]
    if not vecs:
        return None
    avg = np.mean(vecs, axis=0)
    return avg / (np.linalg.norm(avg) + 1e-10)


def find_similar_teams(team: str, k: int = 5, db=None, season: int | None = None) -> list[dict]:
    """Find k most similar teams by average driver embedding cosine similarity."""
    if db is None:
        db = get_db()

    target_vec = _team_embedding(team, db, season)
    if target_vec is None:
        raise ValueError(f"No VectorProfiles found for team '{team}'")

    s_filt = _season_filter(season)
    all_teams = db[OUT_COL].distinct("team", s_filt)
    results = []
    for t in all_teams:
        if not t or t == team:
            continue
        vec = _team_embedding(t, db, season)
        if vec is None:
            continue
        score = float(np.dot(target_vec, vec))
        drivers = [d["driver_code"] for d in db[OUT_COL].find({"team": t, **s_filt}, {"driver_code": 1})]
        results.append({"team": t, "score": round(score, 4), "drivers": drivers})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


def get_intra_team_similarity(team: str, db=None, season: int | None = None) -> list[dict]:
    """Get pairwise similarity between all drivers within a team."""
    if db is None:
        db = get_db()

    filt = {"team": team, **_season_filter(season)}
    docs = list(db[OUT_COL].find(filt, {"driver_code": 1, "embedding": 1}))
    docs = [d for d in docs if "embedding" in d]
    if len(docs) < 2:
        return []

    pairs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            v1 = np.array(docs[i]["embedding"])
            v2 = np.array(docs[j]["embedding"])
            score = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            pairs.append({
                "driver_a": docs[i]["driver_code"],
                "driver_b": docs[j]["driver_code"],
                "score": round(score, 4),
            })
    pairs.sort(key=lambda x: x["score"], reverse=True)
    return pairs


def find_similar(driver_code: str, k: int = 5, db=None, season: int | None = None) -> list[dict]:
    """Find k most similar drivers by cosine similarity of embeddings."""
    if db is None:
        db = get_db()

    coll = db[OUT_COL]
    s_filt = _season_filter(season)
    target = coll.find_one({"driver_code": driver_code.upper(), **s_filt})
    if not target or "embedding" not in target:
        raise ValueError(f"No VectorProfile found for {driver_code}")

    target_vec = np.array(target["embedding"])

    candidate_filt = {"driver_code": {"$ne": driver_code.upper()}, **s_filt}
    results = []
    for doc in coll.find(candidate_filt, {"driver_code": 1, "team": 1, "embedding": 1}):
        if "embedding" not in doc:
            continue
        vec = np.array(doc["embedding"])
        score = float(np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-10))
        results.append({
            "driver_code": doc["driver_code"],
            "team": doc.get("team", ""),
            "score": round(score, 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


# ── Main pipeline ────────────────────────────────────────────────────────


def main(rebuild: bool = False):
    t0 = time.time()

    print("[1/5] Connecting to MongoDB...")
    db = get_db()

    print("[2/5] Collecting driver codes...")
    codes = _collect_driver_codes(db)
    print(f"       Found {len(codes)} drivers: {', '.join(codes[:10])}{'...' if len(codes) > 10 else ''}")

    print("[3/5] Fetching source data...")
    sources = _fetch_all_sources(db)
    for name, data in sources.items():
        print(f"       {name}: {len(data)} drivers")

    print(f"[4/5] Merging profiles & generating narratives...")
    docs = []
    narratives = []
    for code in codes:
        merged = _merge_driver(code, sources)
        narrative = _build_narrative(merged)
        merged["narrative"] = narrative
        docs.append(merged)
        narratives.append(narrative)

    print(f"       Embedding {len(narratives)} narratives (Nomic 768-dim)...")
    embeddings = _embed_narratives(narratives)
    for doc, emb in zip(docs, embeddings):
        doc["embedding"] = emb
        doc["built_at"] = datetime.now(timezone.utc).isoformat()

    print(f"[5/6] Upserting legacy profiles to {OUT_COL}...")
    coll = db[OUT_COL]

    if rebuild:
        coll.drop()
        print("       Dropped existing collection (--rebuild)")

    # Set season: None on legacy docs
    for doc in docs:
        doc["season"] = None

    ops = [
        UpdateOne(
            {"driver_code": doc["driver_code"], "season": None},
            {"$set": doc},
            upsert=True,
        )
        for doc in docs
    ]
    result = coll.bulk_write(ops)

    # Migrate any existing docs without season field
    migrated = coll.update_many(
        {"season": {"$exists": False}},
        {"$set": {"season": None}},
    )
    if migrated.modified_count:
        print(f"       Migrated {migrated.modified_count} legacy docs (set season: null)")

    # Create compound unique index
    coll.drop_indexes()
    coll.create_index([("driver_code", 1), ("season", 1)], unique=True)

    print(f"       Legacy: {result.upserted_count} new, {result.modified_count} updated")

    # ── Per-season profiles ──────────────────────────────────────────────
    # Find available seasons from raw data (jolpica has race results per season)
    available_seasons = set()
    for s in db["jolpica_race_results"].distinct("season"):
        if s is not None:
            available_seasons.add(int(s))

    if available_seasons:
        print(f"\n[6/6] Building per-season profiles for: {sorted(available_seasons)}")
        for season in sorted(available_seasons):
            season_sources = _fetch_season_sources(db, season)

            # Collect drivers from raw race results for this season
            season_codes = sorted(
                db["jolpica_race_results"].distinct("driver_code", {"season": season})
            )
            season_codes = [c for c in season_codes if c and len(c) == 3]

            # Only include drivers that have at least 1 season-specific source
            season_codes = [
                c for c in season_codes
                if c in season_sources["performance"]
                or c in season_sources["overtaking"]
                or c in season_sources["telemetry"]
            ]

            if not season_codes:
                print(f"       Season {season}: no drivers with season-specific data")
                continue

            season_docs = []
            season_narratives = []
            for code in season_codes:
                merged = _merge_driver(code, season_sources)
                merged["season"] = season
                narrative = f"[{season} season] " + _build_narrative(merged)
                merged["narrative"] = narrative
                season_docs.append(merged)
                season_narratives.append(narrative)

            print(f"       Season {season}: embedding {len(season_narratives)} drivers...")
            season_embeddings = _embed_narratives(season_narratives)
            for doc, emb in zip(season_docs, season_embeddings):
                doc["embedding"] = emb
                doc["built_at"] = datetime.now(timezone.utc).isoformat()

            s_ops = [
                UpdateOne(
                    {"driver_code": doc["driver_code"], "season": doc["season"]},
                    {"$set": doc},
                    upsert=True,
                )
                for doc in season_docs
            ]
            s_result = coll.bulk_write(s_ops)
            print(f"       Season {season}: {s_result.upserted_count} new, {s_result.modified_count} updated ({len(season_docs)} drivers)")
    else:
        print(f"\n[6/6] No season data found in jolpica_race_results")

    # Log to pipeline_log
    db["pipeline_log"].insert_one({
        "chunk": "build_vector_profiles",
        "status": "complete",
        "drivers_processed": len(docs),
        "seasons_processed": sorted(available_seasons) if available_seasons else [],
        "embedding_dim": EMBEDDING_DIM,
        "upserted": result.upserted_count,
        "modified": result.modified_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    elapsed = time.time() - t0
    print(f"\n  Done! {len(docs)} legacy profiles → {OUT_COL} ({elapsed:.1f}s)")

    # Sample output
    sample = coll.find_one({"driver_code": codes[0], "season": None}, {"embedding": 0, "_id": 0})
    print(f"\n  Sample ({codes[0]}):")
    for k, v in (sample or {}).items():
        if isinstance(v, str) and len(v) > 80:
            v = v[:80] + "..."
        print(f"    {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VectorProfiles collection")
    parser.add_argument("--rebuild", action="store_true", help="Drop and rebuild from scratch")
    parser.add_argument("--season", type=int, help="Build only a specific season (e.g. 2024)")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
