"""
Build VictoryProfiles -- 4-layer knowledge base
------------------------------------------------
Layer 1: victory_driver_profiles     (driver_code, team, season)
Layer 2: victory_car_profiles        (team, season)
Layer 3: victory_strategy_profiles   (driver_code, team, season)
Layer 4: victory_team_kb             (team, season) — combines all layers

Usage:
    python pipeline/build_victory_profiles.py
    python pipeline/build_victory_profiles.py --rebuild
    python pipeline/build_victory_profiles.py --season 2024
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from groq import Groq
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from updater._db import get_db
from embeddings import NomicEmbedder

GROQ_MODEL = os.getenv("VICTORY_LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
EMBEDDING_DIM = 768

_groq: Groq | None = None


def _get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq


# -- Narrative generation (Groq) ----------------------------------------------

def _generate_narrative(prompt: str) -> str:
    """Generate a ~200-word narrative summary via Groq."""
    try:
        client = _get_groq()
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an F1 data analyst. Write concise ~200-word profile summaries from the data provided. Be factual and specific with numbers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"       Groq error: {e}")
        return prompt  # Fallback: use raw prompt as narrative


# -- Helpers ------------------------------------------------------------------

def _safe(val, fmt=".3f", suffix=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:{fmt}}{suffix}"


# -- Layer 1: Driver Profiles ------------------------------------------------

def _fetch_driver_sources(db, season: int | None) -> dict:
    """Fetch 4 driver source collections."""
    sources = {}
    s_filter = {"season": season} if season else {}

    # 1. Performance markers
    sources["performance"] = {
        doc["Driver"]: {k: v for k, v in doc.items() if k not in ("_id", "Driver", "season")}
        for doc in db["driver_performance_markers"].find(s_filter, {"_id": 0})
        if doc.get("Driver")
    }

    # 2. Overtake profiles
    sources["overtaking"] = {
        doc["driver_code"]: {k: v for k, v in doc.items() if k not in ("_id", "driver_code", "season")}
        for doc in db["driver_overtake_profiles"].find(s_filter, {"_id": 0})
        if doc.get("driver_code")
    }

    # 3. Telemetry profiles
    sources["telemetry"] = {
        doc["driver_code"]: {k: v for k, v in doc.items() if k not in ("_id", "driver_code", "season")}
        for doc in db["driver_telemetry_profiles"].find(s_filter, {"_id": 0})
        if doc.get("driver_code")
    }

    # 4. Anomaly snapshot (current state, not season-filtered)
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
                "systems": systems,
            }
    sources["health"] = health

    return sources


def _collect_driver_codes(sources: dict) -> list[str]:
    """Union of driver codes across all sources."""
    codes = set()
    for src in sources.values():
        if isinstance(src, dict):
            codes.update(src.keys())
    return sorted(c for c in codes if c and len(c) == 3)


def _merge_driver(code: str, sources: dict) -> dict:
    """Merge one driver's data from all 4 sources."""
    doc = {"driver_code": code}
    health = sources["health"].get(code)
    doc["team"] = health["team"] if health else ""

    for key in ("performance", "overtaking", "telemetry"):
        doc[key] = sources[key].get(code)

    doc["health"] = health
    doc["sources_found"] = [k for k in ("performance", "overtaking", "telemetry", "health") if doc.get(k)]
    return doc


def _driver_narrative_prompt(doc: dict) -> str:
    """Build prompt for Ollama to generate driver narrative."""
    code = doc["driver_code"]
    team = doc.get("team") or "unknown team"
    parts = [f"Summarize this F1 driver profile in ~200 words for {code} ({team}):"]

    p = doc.get("performance")
    if p:
        parts.append(f"Performance: top speed {_safe(p.get('avg_top_speed_kmh'), '.1f')} km/h, "
                      f"consistency {_safe(p.get('lap_time_consistency_std'))} s std, "
                      f"late race delta {_safe(p.get('late_race_delta_s'))} s, "
                      f"throttle smoothness {_safe(p.get('throttle_smoothness'))}.")

    ot = doc.get("overtaking")
    if ot:
        parts.append(f"Overtaking: {ot.get('total_overtakes_made', '?')} made, "
                      f"{ot.get('total_times_overtaken', '?')} lost, "
                      f"ratio {_safe(ot.get('overtake_ratio'))}.")

    t = doc.get("telemetry")
    if t:
        parts.append(f"Telemetry: avg braking {_safe(t.get('avg_braking_g'))} G, "
                      f"throttle {_safe(t.get('avg_throttle_pct'), '.1f')}%, "
                      f"DRS usage {_safe(t.get('drs_usage_ratio'))}.")

    h = doc.get("health")
    if h:
        sys_str = ", ".join(f"{k}: {v}/100" for k, v in h.get("systems", {}).items())
        parts.append(f"System health: {h.get('overall_health', '?')}/100 ({h.get('overall_level', '?')}). "
                      f"Systems: {sys_str}.")

    return "\n".join(parts)


def build_driver_profiles(db, season: int | None, embedder: NomicEmbedder, rebuild: bool = False) -> list[dict]:
    """Build Layer 1: victory_driver_profiles."""
    coll = db["victory_driver_profiles"]
    print(f"\n[Layer 1] Building driver profiles (season={season})...")

    sources = _fetch_driver_sources(db, season)
    codes = _collect_driver_codes(sources)
    print(f"  Found {len(codes)} drivers")

    docs = []
    for code in codes:
        merged = _merge_driver(code, sources)
        merged["season"] = season

        # Generate narrative via Ollama
        prompt = _driver_narrative_prompt(merged)
        narrative = _generate_narrative(prompt)
        merged["narrative"] = narrative
        docs.append(merged)

    # Batch embed
    if docs:
        print(f"  Embedding {len(docs)} driver narratives...")
        narratives = [d["narrative"] for d in docs]
        embeddings = embedder.embed(narratives)
        for doc, emb in zip(docs, embeddings):
            doc["embedding"] = emb
            doc["built_at"] = datetime.now(timezone.utc).isoformat()

        if rebuild:
            coll.delete_many({"season": season})

        ops = [
            UpdateOne(
                {"driver_code": d["driver_code"], "team": d["team"], "season": season},
                {"$set": d},
                upsert=True,
            )
            for d in docs
        ]
        result = coll.bulk_write(ops)
        print(f"  Upserted {result.upserted_count} new, {result.modified_count} updated")

    # Ensure index
    coll.create_index([("driver_code", 1), ("team", 1), ("season", 1)], unique=True)
    return docs


# -- Layer 2: Car Profiles ---------------------------------------------------

def _fetch_car_sources(db, season: int | None) -> dict:
    """Fetch car-level data: anomaly health per team, telemetry summaries, constructor profiles."""
    sources = {}

    # 1. Anomaly health aggregated by team (from anomaly_scores_snapshot)
    team_health = {}
    snapshot = db["anomaly_scores_snapshot"].find_one({}, {"_id": 0})
    if snapshot:
        for d in snapshot.get("drivers", []):
            team = d.get("team", "")
            if not team:
                continue
            if team not in team_health:
                team_health[team] = {"systems": {}, "drivers": [], "overall_health_sum": 0, "count": 0}

            # Aggregate system health across drivers for each team
            for race in d.get("races", []):
                for sys_name, sys_data in race.get("systems", {}).items():
                    if sys_name not in team_health[team]["systems"]:
                        team_health[team]["systems"][sys_name] = []
                    team_health[team]["systems"][sys_name].append(sys_data.get("health", 0))

            team_health[team]["drivers"].append(d.get("code", ""))
            team_health[team]["overall_health_sum"] += d.get("overall_health", 0)
            team_health[team]["count"] += 1

    # Average system health per team
    for team, data in team_health.items():
        for sys_name in data["systems"]:
            vals = data["systems"][sys_name]
            data["systems"][sys_name] = round(sum(vals) / len(vals), 1) if vals else 0
        data["overall_health"] = round(data["overall_health_sum"] / data["count"], 1) if data["count"] else 0
        del data["overall_health_sum"]

    sources["health"] = team_health

    # 2. Telemetry race summaries aggregated by team
    team_telemetry = {}
    year_filter = {"Year": season} if season else {}
    for doc in db["telemetry_race_summary"].find(year_filter, {"_id": 0}):
        driver = doc.get("Driver", "")
        # We need team mapping -- get from health data
        team = None
        for t, th in team_health.items():
            if driver in th["drivers"]:
                team = t
                break
        if not team:
            continue

        if team not in team_telemetry:
            team_telemetry[team] = {
                "avg_speed": [], "top_speed": [], "avg_rpm": [],
                "avg_throttle": [], "brake_pct": [], "drs_pct": [],
                "race_count": 0,
            }
        tt = team_telemetry[team]
        for field in ("avg_speed", "top_speed", "avg_rpm", "avg_throttle", "brake_pct", "drs_pct"):
            val = doc.get(field)
            if val is not None:
                tt[field].append(val)
        tt["race_count"] += 1

    # Average telemetry per team
    for team, data in team_telemetry.items():
        for field in ("avg_speed", "top_speed", "avg_rpm", "avg_throttle", "brake_pct", "drs_pct"):
            vals = data[field]
            data[field] = round(sum(vals) / len(vals), 2) if vals else 0
    sources["telemetry"] = team_telemetry

    # 3. Constructor profiles
    cp_filter = {"season": season} if season else {}
    sources["constructor"] = {}
    for doc in db["constructor_profiles"].find(cp_filter, {"_id": 0}):
        team_id = doc.get("constructor_id") or doc.get("constructor_name", "")
        if team_id:
            sources["constructor"][team_id] = doc
    return sources


def _car_narrative_prompt(team: str, sources: dict) -> str:
    """Build prompt for car profile narrative."""
    parts = [f"Summarize this F1 car profile in ~200 words for {team}:"]

    h = sources["health"].get(team)
    if h:
        sys_str = ", ".join(f"{k}: {v}/100" for k, v in h.get("systems", {}).items())
        parts.append(f"System health: overall {h.get('overall_health', '?')}/100. {sys_str}.")

    t = sources["telemetry"].get(team)
    if t:
        parts.append(f"Telemetry: avg speed {t.get('avg_speed', '?')} km/h, "
                      f"top speed {t.get('top_speed', '?')} km/h, "
                      f"avg throttle {t.get('avg_throttle', '?')}%, "
                      f"brake usage {t.get('brake_pct', '?')}%, "
                      f"DRS {t.get('drs_pct', '?')}%.")

    # Constructor profile -- check both team name and constructor_id
    cp = sources["constructor"].get(team)
    if not cp:
        # Try lowercase/normalized match
        for k, v in sources["constructor"].items():
            if k.lower().replace(" ", "_") == team.lower().replace(" ", "_"):
                cp = v
                break
            if v.get("constructor_name", "").lower() == team.lower():
                cp = v
                break
    if cp:
        parts.append(f"Race stats: {cp.get('total_wins', 0)} wins, {cp.get('total_podiums', 0)} podiums, "
                      f"DNF rate {_safe(cp.get('dnf_rate'), '.2f')}, "
                      f"avg finish P{_safe(cp.get('avg_finish_position'), '.1f')}, "
                      f"avg pit {_safe(cp.get('avg_pit_duration_s'), '.1f')}s.")

    return "\n".join(parts)


def build_car_profiles(db, season: int | None, embedder: NomicEmbedder, rebuild: bool = False) -> list[dict]:
    """Build Layer 2: victory_car_profiles."""
    coll = db["victory_car_profiles"]
    print(f"\n[Layer 2] Building car profiles (season={season})...")

    sources = _fetch_car_sources(db, season)
    teams = sorted(set(list(sources["health"].keys()) + list(sources["telemetry"].keys())))
    print(f"  Found {len(teams)} teams")

    docs = []
    for team in teams:
        doc = {
            "team": team,
            "season": season,
            "health": sources["health"].get(team),
            "telemetry": sources["telemetry"].get(team),
            "constructor": None,
        }

        # Match constructor profile
        for k, v in sources["constructor"].items():
            if (k.lower().replace(" ", "_") == team.lower().replace(" ", "_")
                    or v.get("constructor_name", "").lower() == team.lower()):
                doc["constructor"] = {
                    key: v.get(key) for key in (
                        "total_wins", "total_podiums", "total_points", "dnf_count",
                        "dnf_rate", "avg_finish_position", "avg_grid_position",
                        "avg_pit_duration_s", "best_pit_duration_s", "pit_consistency",
                        "fleet_avg_speed", "fleet_top_speed", "q3_rate", "pole_count",
                    )
                }
                break

        doc["sources_found"] = [k for k in ("health", "telemetry", "constructor") if doc.get(k)]

        prompt = _car_narrative_prompt(team, sources)
        narrative = _generate_narrative(prompt)
        doc["narrative"] = narrative
        docs.append(doc)

    if docs:
        print(f"  Embedding {len(docs)} car narratives...")
        narratives = [d["narrative"] for d in docs]
        embeddings = embedder.embed(narratives)
        for doc, emb in zip(docs, embeddings):
            doc["embedding"] = emb
            doc["built_at"] = datetime.now(timezone.utc).isoformat()

        if rebuild:
            coll.delete_many({"season": season})

        ops = [
            UpdateOne(
                {"team": d["team"], "season": season},
                {"$set": d},
                upsert=True,
            )
            for d in docs
        ]
        result = coll.bulk_write(ops)
        print(f"  Upserted {result.upserted_count} new, {result.modified_count} updated")

    coll.create_index([("team", 1), ("season", 1)], unique=True)
    return docs


# -- Layer 3: Strategy Profiles -----------------------------------------------

def _fetch_strategy_sources(db, season: int | None) -> dict:
    """Fetch strategy data: opponent profiles, compound profiles, pit stops."""
    sources = {}

    # 1. Opponent profiles — per-driver strategy metrics (career-wide, not season-filtered)
    _STRATEGY_FIELDS = (
        "undercut_aggression_score", "tyre_extension_bias",
        "one_stop_freq", "two_stop_freq", "three_stop_freq",
        "avg_first_stop_lap", "median_first_stop_lap",
        "early_stop_freq", "late_stop_freq",
        "avg_tyre_life", "long_stint_capability", "preferred_compound",
        "stint_endurance_slope", "avg_lap_time_degradation_per_lap",
        "avg_pit_duration_s", "min_pit_duration_s",
    )
    opp_by_code = {}
    id_to_code = {}  # driver_id -> driver_code mapping
    for doc in db["opponent_profiles"].find({}, {"_id": 0}):
        code = doc.get("driver_code")
        did = doc.get("driver_id")
        if code:
            opp_by_code[code] = {k: doc.get(k) for k in _STRATEGY_FIELDS if doc.get(k) is not None}
        if did and code:
            id_to_code[did] = code
    sources["opponent"] = opp_by_code
    sources["_id_to_code"] = id_to_code

    # 2. Compound profiles — per-driver per-compound tyre behavior
    compound_by_code: dict[str, list[dict]] = {}
    for doc in db["opponent_compound_profiles"].find({}, {"_id": 0}):
        did = doc.get("driver_id", "")
        code = id_to_code.get(did, did.upper()[:3] if did else "")
        if not code or len(code) != 3:
            continue
        compound_by_code.setdefault(code, []).append({
            "compound": doc.get("compound"),
            "total_laps": doc.get("total_laps"),
            "avg_lap_time_s": doc.get("avg_lap_time_s"),
            "std_lap_time_s": doc.get("std_lap_time_s"),
            "avg_tyre_life": doc.get("avg_tyre_life"),
            "degradation_slope": doc.get("degradation_slope"),
        })
    sources["compounds"] = compound_by_code

    # 3. Pit stops aggregated per driver — filtered by season if provided
    pit_filter = {"season": season} if season else {}
    pit_by_id: dict[str, list[dict]] = {}
    for doc in db["jolpica_pit_stops"].find(pit_filter, {"_id": 0}):
        did = doc.get("driver_id", "")
        pit_by_id.setdefault(did, []).append(doc)

    # Aggregate pit stats per driver_code
    pit_by_code = {}
    for did, stops in pit_by_id.items():
        code = id_to_code.get(did)
        if not code:
            continue
        durations = [s["duration_s"] for s in stops if s.get("duration_s") and s["duration_s"] < 60]
        if not durations:
            continue
        # Group by (season, round) to count stops per race
        races: dict[tuple, int] = {}
        for s in stops:
            key = (s.get("season"), s.get("round"))
            races[key] = max(races.get(key, 0), s.get("stop", 1))
        stops_per_race = list(races.values())

        pit_by_code[code] = {
            "total_stops": len(durations),
            "avg_duration_s": round(sum(durations) / len(durations), 2),
            "best_duration_s": round(min(durations), 2),
            "pit_consistency_std": round(float(np.std(durations)), 2) if len(durations) > 1 else 0,
            "avg_stops_per_race": round(sum(stops_per_race) / len(stops_per_race), 2) if stops_per_race else 0,
            "races_with_pits": len(stops_per_race),
        }
    sources["pits"] = pit_by_code

    return sources


def _merge_strategy(code: str, team: str, sources: dict) -> dict:
    """Merge strategy data for one driver."""
    doc = {"driver_code": code, "team": team}

    opp = sources["opponent"].get(code)
    if opp:
        doc["pit_strategy"] = {
            "undercut_aggression": opp.get("undercut_aggression_score"),
            "tyre_extension_bias": opp.get("tyre_extension_bias"),
            "one_stop_freq": opp.get("one_stop_freq"),
            "two_stop_freq": opp.get("two_stop_freq"),
            "three_stop_freq": opp.get("three_stop_freq"),
            "avg_first_stop_lap": opp.get("avg_first_stop_lap"),
            "early_stop_freq": opp.get("early_stop_freq"),
            "late_stop_freq": opp.get("late_stop_freq"),
        }
        doc["tyre_management"] = {
            "avg_tyre_life": opp.get("avg_tyre_life"),
            "long_stint_capability": opp.get("long_stint_capability"),
            "preferred_compound": opp.get("preferred_compound"),
            "stint_endurance_slope": opp.get("stint_endurance_slope"),
            "degradation_per_lap": opp.get("avg_lap_time_degradation_per_lap"),
        }

    compounds = sources["compounds"].get(code, [])
    if compounds:
        # Deduplicate by compound name, keeping highest total_laps
        best: dict[str, dict] = {}
        for c in compounds:
            name = c.get("compound")
            if not name:
                continue
            if name not in best or (c.get("total_laps") or 0) > (best[name].get("total_laps") or 0):
                best[name] = c
        deduped = sorted(best.values(), key=lambda c: c.get("total_laps") or 0, reverse=True)
        doc["compound_profiles"] = deduped[:4]

    pits = sources["pits"].get(code)
    if pits:
        doc["pit_execution"] = pits

    doc["sources_found"] = [
        k for k in ("pit_strategy", "compound_profiles", "pit_execution")
        if doc.get(k)
    ]
    return doc


def _strategy_narrative_prompt(doc: dict) -> str:
    """Build prompt for strategy profile narrative."""
    code = doc["driver_code"]
    team = doc.get("team") or "unknown team"
    parts = [f"Summarize this F1 strategy profile in ~200 words for {code} ({team}):"]

    ps = doc.get("pit_strategy")
    if ps:
        parts.append(
            f"Pit strategy: undercut aggression {_safe(ps.get('undercut_aggression'))}, "
            f"tyre extension bias {_safe(ps.get('tyre_extension_bias'))}, "
            f"1-stop freq {_safe(ps.get('one_stop_freq'))}, "
            f"2-stop freq {_safe(ps.get('two_stop_freq'))}, "
            f"avg first stop lap {_safe(ps.get('avg_first_stop_lap'), '.1f')}."
        )

    tm = doc.get("tyre_management")
    if tm:
        parts.append(
            f"Tyre management: avg life {_safe(tm.get('avg_tyre_life'), '.1f')} laps, "
            f"long stint capability {_safe(tm.get('long_stint_capability'), '.1f')} laps, "
            f"preferred compound {tm.get('preferred_compound', '?')}, "
            f"degradation {_safe(tm.get('degradation_per_lap'))} s/lap."
        )

    comps = doc.get("compound_profiles", [])
    if comps:
        comp_strs = [f"{c.get('compound', '?')}: {c.get('total_laps', 0)} laps, "
                     f"deg slope {_safe(c.get('degradation_slope'))}" for c in comps]
        parts.append(f"Compound breakdown: {'; '.join(comp_strs)}.")

    pe = doc.get("pit_execution")
    if pe:
        parts.append(
            f"Pit execution: avg {pe.get('avg_duration_s', '?')}s, "
            f"best {pe.get('best_duration_s', '?')}s, "
            f"avg {pe.get('avg_stops_per_race', '?')} stops/race."
        )

    return "\n".join(parts)


def build_strategy_profiles(db, season: int | None, embedder: NomicEmbedder,
                            driver_docs: list[dict], rebuild: bool = False) -> list[dict]:
    """Build Layer 3: victory_strategy_profiles."""
    coll = db["victory_strategy_profiles"]
    print(f"\n[Layer 3] Building strategy profiles (season={season})...")

    sources = _fetch_strategy_sources(db, season)

    # Build strategy docs for drivers that exist in driver_docs
    driver_teams = {d["driver_code"]: d.get("team", "") for d in driver_docs}
    codes = sorted(driver_teams.keys())
    print(f"  Found {len(codes)} drivers with strategy data potential")

    docs = []
    for code in codes:
        merged = _merge_strategy(code, driver_teams[code], sources)
        merged["season"] = season

        if not merged.get("sources_found"):
            continue  # Skip drivers with no strategy data at all

        prompt = _strategy_narrative_prompt(merged)
        narrative = _generate_narrative(prompt)
        merged["narrative"] = narrative
        docs.append(merged)

    if docs:
        print(f"  Embedding {len(docs)} strategy narratives...")
        narratives = [d["narrative"] for d in docs]
        embeddings = embedder.embed(narratives)
        for doc, emb in zip(docs, embeddings):
            doc["embedding"] = emb
            doc["built_at"] = datetime.now(timezone.utc).isoformat()

        if rebuild:
            coll.delete_many({"season": season})

        ops = [
            UpdateOne(
                {"driver_code": d["driver_code"], "team": d["team"], "season": season},
                {"$set": d},
                upsert=True,
            )
            for d in docs
        ]
        result = coll.bulk_write(ops)
        print(f"  Upserted {result.upserted_count} new, {result.modified_count} updated")

    coll.create_index([("driver_code", 1), ("team", 1), ("season", 1)], unique=True)
    return docs


# -- Layer 4: Team Knowledge Base --------------------------------------------

def _team_kb_prompt(team: str, driver_docs: list[dict], car_doc: dict | None,
                    strategy_docs: list[dict] | None = None) -> str:
    """Build prompt for team KB narrative."""
    parts = [f"Write a comprehensive ~200-word intelligence briefing for {team} combining driver, car, and strategy analysis:"]

    for dd in driver_docs:
        code = dd["driver_code"]
        p = dd.get("performance")
        if p:
            parts.append(f"Driver {code}: top speed {_safe(p.get('avg_top_speed_kmh'), '.1f')} km/h, "
                          f"consistency {_safe(p.get('lap_time_consistency_std'))} s, "
                          f"late race {_safe(p.get('late_race_delta_s'))} s.")

    if car_doc:
        h = car_doc.get("health")
        if h:
            parts.append(f"Car health: {h.get('overall_health', '?')}/100.")
        c = car_doc.get("constructor")
        if c:
            parts.append(f"Race record: {c.get('total_wins', 0)} wins, "
                          f"DNF rate {_safe(c.get('dnf_rate'), '.2f')}.")

    if strategy_docs:
        for sd in strategy_docs:
            code = sd["driver_code"]
            ps = sd.get("pit_strategy")
            tm = sd.get("tyre_management")
            if ps:
                parts.append(f"Strategy {code}: undercut aggression {_safe(ps.get('undercut_aggression'))}, "
                              f"1-stop freq {_safe(ps.get('one_stop_freq'))}.")
            if tm:
                parts.append(f"Tyre mgmt {code}: avg life {_safe(tm.get('avg_tyre_life'), '.1f')} laps, "
                              f"preferred {tm.get('preferred_compound', '?')}.")

    return "\n".join(parts)


def build_team_kbs(db, season: int | None, embedder: NomicEmbedder,
                   driver_docs: list[dict], car_docs: list[dict],
                   strategy_docs: list[dict] | None = None, rebuild: bool = False) -> list[dict]:
    """Build Layer 4: victory_team_kb."""
    coll = db["victory_team_kb"]
    print(f"\n[Layer 4] Building team knowledge bases (season={season})...")

    # Group drivers by team
    drivers_by_team: dict[str, list[dict]] = {}
    for dd in driver_docs:
        team = dd.get("team", "")
        if team:
            drivers_by_team.setdefault(team, []).append(dd)

    # Group strategy by team
    strategy_by_team: dict[str, list[dict]] = {}
    for sd in (strategy_docs or []):
        team = sd.get("team", "")
        if team:
            strategy_by_team.setdefault(team, []).append(sd)

    # Index car docs by team
    car_by_team = {cd["team"]: cd for cd in car_docs}

    teams = sorted(set(list(drivers_by_team.keys()) + list(car_by_team.keys())))
    print(f"  Found {len(teams)} teams")

    docs = []
    for team in teams:
        team_drivers = drivers_by_team.get(team, [])
        team_strategy = strategy_by_team.get(team, [])
        car = car_by_team.get(team)

        # Structured metadata: raw metrics for filtering/comparison
        metadata = {
            "driver_count": len(team_drivers),
            "drivers": [],
            "car": {},
            "strategy": {},
        }

        for dd in team_drivers:
            driver_meta = {"driver_code": dd["driver_code"]}
            p = dd.get("performance")
            if p:
                for key in ("avg_top_speed_kmh", "lap_time_consistency_std",
                             "late_race_delta_s", "throttle_smoothness", "degradation_slope_s_per_lap"):
                    if p.get(key) is not None:
                        driver_meta[key] = p[key]
            ot = dd.get("overtaking")
            if ot:
                for key in ("overtake_ratio", "total_overtakes_made", "total_times_overtaken"):
                    if ot.get(key) is not None:
                        driver_meta[key] = ot[key]
            h = dd.get("health")
            if h:
                driver_meta["overall_health"] = h.get("overall_health")
                driver_meta["systems"] = h.get("systems", {})
            metadata["drivers"].append(driver_meta)

        if car:
            metadata["car"]["health"] = car.get("health")
            metadata["car"]["telemetry"] = car.get("telemetry")
            metadata["car"]["constructor"] = car.get("constructor")

        # Aggregate strategy metadata per team
        if team_strategy:
            # Average pit strategy across drivers
            agg_undercut = [s["pit_strategy"]["undercut_aggression"] for s in team_strategy
                           if s.get("pit_strategy", {}).get("undercut_aggression") is not None]
            agg_tyre_life = [s["tyre_management"]["avg_tyre_life"] for s in team_strategy
                            if s.get("tyre_management", {}).get("avg_tyre_life") is not None]
            agg_one_stop = [s["pit_strategy"]["one_stop_freq"] for s in team_strategy
                           if s.get("pit_strategy", {}).get("one_stop_freq") is not None]

            metadata["strategy"] = {
                "team_undercut_aggression": round(sum(agg_undercut) / len(agg_undercut), 3) if agg_undercut else None,
                "team_avg_tyre_life": round(sum(agg_tyre_life) / len(agg_tyre_life), 1) if agg_tyre_life else None,
                "team_one_stop_freq": round(sum(agg_one_stop) / len(agg_one_stop), 3) if agg_one_stop else None,
                "drivers": [
                    {
                        "driver_code": s["driver_code"],
                        "undercut_aggression": s.get("pit_strategy", {}).get("undercut_aggression"),
                        "tyre_extension_bias": s.get("pit_strategy", {}).get("tyre_extension_bias"),
                        "avg_tyre_life": s.get("tyre_management", {}).get("avg_tyre_life"),
                        "preferred_compound": s.get("tyre_management", {}).get("preferred_compound"),
                    }
                    for s in team_strategy
                ],
            }

        # Generate narrative
        prompt = _team_kb_prompt(team, team_drivers, car, team_strategy)
        narrative = _generate_narrative(prompt)

        doc = {
            "team": team,
            "season": season,
            "narrative": narrative,
            "metadata": metadata,
            "sources_found": [],
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        if team_drivers:
            doc["sources_found"].append("drivers")
        if car:
            doc["sources_found"].append("car")
        if team_strategy:
            doc["sources_found"].append("strategy")
        docs.append(doc)

    if docs:
        print(f"  Embedding {len(docs)} team narratives...")
        narratives = [d["narrative"] for d in docs]
        embeddings = embedder.embed(narratives)
        for doc, emb in zip(docs, embeddings):
            doc["embedding"] = emb

        if rebuild:
            coll.delete_many({"season": season})

        ops = [
            UpdateOne(
                {"team": d["team"], "season": season},
                {"$set": d},
                upsert=True,
            )
            for d in docs
        ]
        result = coll.bulk_write(ops)
        print(f"  Upserted {result.upserted_count} new, {result.modified_count} updated")

    coll.create_index([("team", 1), ("season", 1)], unique=True)
    return docs


# -- Main ---------------------------------------------------------------------

def main(season: int | None = None, rebuild: bool = False):
    t0 = time.time()

    print("=== VictoryProfiles Pipeline ===")
    print(f"Season: {season or 'all (legacy)'}")

    db = get_db()
    embedder = NomicEmbedder()

    driver_docs = build_driver_profiles(db, season, embedder, rebuild)
    car_docs = build_car_profiles(db, season, embedder, rebuild)
    strategy_docs = build_strategy_profiles(db, season, embedder, driver_docs, rebuild)
    team_docs = build_team_kbs(db, season, embedder, driver_docs, car_docs, strategy_docs, rebuild)

    # Log
    db["pipeline_log"].insert_one({
        "chunk": "build_victory_profiles",
        "status": "complete",
        "season": season,
        "layers": {
            "driver_profiles": len(driver_docs),
            "car_profiles": len(car_docs),
            "strategy_profiles": len(strategy_docs),
            "team_kbs": len(team_docs),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    elapsed = time.time() - t0
    print(f"\nDone! {len(driver_docs)} drivers, {len(car_docs)} cars, "
          f"{len(strategy_docs)} strategies, {len(team_docs)} teams ({elapsed:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VictoryProfiles (4-layer knowledge base)")
    parser.add_argument("--season", type=int, default=None, help="Season year (default: all/legacy)")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild for target season")
    args = parser.parse_args()
    main(season=args.season, rebuild=args.rebuild)
