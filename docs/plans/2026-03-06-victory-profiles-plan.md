# VictoryProfiles Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a 3-layer knowledge base (driver profiles → car profiles → team KB) with hybrid embeddings for competitive intelligence and internal improvement analysis.

**Architecture:** Single pipeline script (`pipeline/build_victory_profiles.py`) builds all 3 layers sequentially — driver profiles from 4 driver sources, car profiles from anomaly snapshot + telemetry summaries + constructor profiles, team KBs combining both. Each layer generates Ollama narratives, embeds with Nomic 768-dim, and stores structured metadata alongside. Four FastAPI endpoints in `chat_server.py` expose the data.

**Tech Stack:** Python, pymongo, Nomic embeddings (768-dim via `pipeline/embeddings.py`), Ollama qwen3.5:9b for narrative generation, FastAPI, numpy for cosine similarity.

**Design doc:** `docs/plans/2026-03-06-victory-profiles-design.md`

---

## Task 1: Driver Profile Builder

Build Layer 1 — merge 4 driver data sources into `victory_driver_profiles` with narrative + embedding.

**Files:**
- Create: `pipeline/build_victory_profiles.py`

**Step 1: Create the pipeline script with driver profile builder**

This reuses patterns from `pipeline/build_vector_profiles.py` (same source collections, same merge logic) but keys by `(driver_code, team, season)` instead of just `driver_code`.

```python
"""
Build VictoryProfiles — 3-layer knowledge base
───────────────────────────────────────────────
Layer 1: victory_driver_profiles  (driver_code, team, season)
Layer 2: victory_car_profiles     (team, season)
Layer 3: victory_team_kb          (team, season)

Usage:
    python pipeline/build_victory_profiles.py
    python pipeline/build_victory_profiles.py --rebuild
    python pipeline/build_victory_profiles.py --season 2024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from updater._db import get_db
from embeddings import NomicEmbedder

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
EMBEDDING_DIM = 768


# ── Ollama narrative generation ──────────────────────────────────────────

def _generate_narrative(prompt: str) -> str:
    """Generate a ~200-word narrative summary via Ollama."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 300},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"       Ollama error: {e}")
        return prompt  # Fallback: use raw prompt as narrative


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe(val, fmt=".3f", suffix=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:{fmt}}{suffix}"


def _q_health(v):
    if v is None:
        return "unknown"
    if v >= 80:
        return "healthy"
    if v >= 60:
        return "concerning"
    return "critical"


# ── Layer 1: Driver Profiles ────────────────────────────────────────────

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


# ── Layer 2: Car Profiles ───────────────────────────────────────────────

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
        # We need team mapping — get from health data
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

    # Constructor profile — check both team name and constructor_id
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


# ── Layer 3: Team Knowledge Base ────────────────────────────────────────

def _team_kb_prompt(team: str, driver_docs: list[dict], car_doc: dict | None) -> str:
    """Build prompt for team KB narrative."""
    parts = [f"Write a comprehensive ~200-word intelligence briefing for {team} combining driver and car analysis:"]

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

    return "\n".join(parts)


def build_team_kbs(db, season: int | None, embedder: NomicEmbedder,
                   driver_docs: list[dict], car_docs: list[dict], rebuild: bool = False) -> list[dict]:
    """Build Layer 3: victory_team_kb."""
    coll = db["victory_team_kb"]
    print(f"\n[Layer 3] Building team knowledge bases (season={season})...")

    # Group drivers by team
    drivers_by_team: dict[str, list[dict]] = {}
    for dd in driver_docs:
        team = dd.get("team", "")
        if team:
            drivers_by_team.setdefault(team, []).append(dd)

    # Index car docs by team
    car_by_team = {cd["team"]: cd for cd in car_docs}

    teams = sorted(set(list(drivers_by_team.keys()) + list(car_by_team.keys())))
    print(f"  Found {len(teams)} teams")

    docs = []
    for team in teams:
        team_drivers = drivers_by_team.get(team, [])
        car = car_by_team.get(team)

        # Structured metadata: raw metrics for filtering/comparison
        metadata = {
            "driver_count": len(team_drivers),
            "drivers": [],
            "car": {},
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

        # Generate narrative
        prompt = _team_kb_prompt(team, team_drivers, car)
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


# ── Main ────────────────────────────────────────────────────────────────

def main(season: int | None = None, rebuild: bool = False):
    t0 = time.time()

    print("=== VictoryProfiles Pipeline ===")
    print(f"Season: {season or 'all (legacy)'}")

    db = get_db()
    embedder = NomicEmbedder()

    driver_docs = build_driver_profiles(db, season, embedder, rebuild)
    car_docs = build_car_profiles(db, season, embedder, rebuild)
    team_docs = build_team_kbs(db, season, embedder, driver_docs, car_docs, rebuild)

    # Log
    db["pipeline_log"].insert_one({
        "chunk": "build_victory_profiles",
        "status": "complete",
        "season": season,
        "layers": {
            "driver_profiles": len(driver_docs),
            "car_profiles": len(car_docs),
            "team_kbs": len(team_docs),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    elapsed = time.time() - t0
    print(f"\nDone! {len(driver_docs)} drivers, {len(car_docs)} cars, {len(team_docs)} teams ({elapsed:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VictoryProfiles (3-layer knowledge base)")
    parser.add_argument("--season", type=int, default=None, help="Season year (default: all/legacy)")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild for target season")
    args = parser.parse_args()
    main(season=args.season, rebuild=args.rebuild)
```

**Step 2: Run the pipeline**

```bash
cd /home/pedrad/javier_project_folder/f1
PYTHONPATH=pipeline:. python pipeline/build_victory_profiles.py --season 2024
```

Expected: 3 layers built, ~20 drivers, ~10 teams, documents upserted to MongoDB.

**Step 3: Verify in MongoDB**

```bash
cd /home/pedrad/javier_project_folder/f1
python -c "
from pipeline.updater._db import get_db
db = get_db()
for col in ('victory_driver_profiles', 'victory_car_profiles', 'victory_team_kb'):
    print(f'{col}: {db[col].count_documents({})} docs')
    sample = db[col].find_one({}, {'embedding': 0, '_id': 0, 'narrative': 0})
    if sample:
        print(f'  keys: {list(sample.keys())}')
"
```

Expected: All 3 collections populated with correct keys.

**Step 4: Commit**

```bash
git add pipeline/build_victory_profiles.py
git commit -m "feat: add VictoryProfiles pipeline — 3-layer knowledge base builder"
```

---

## Task 2: API Endpoints — Team KB & Compare

Add 4 FastAPI endpoints to `chat_server.py`.

**Files:**
- Modify: `pipeline/chat_server.py` — add endpoints after existing vector profile routes

**Step 1: Add the victory endpoints**

Find the existing `/api/local/team_intel/similar/` endpoint in `chat_server.py` and add the victory endpoints nearby. Add these 4 endpoints:

```python
# ── VictoryProfiles endpoints ────────────────────────────────────────────

@app.get("/api/local/victory/team/{team}/{season}")
async def victory_team_kb(team: str, season: int):
    """Full team KB with driver + car profiles."""
    db = get_data_db()
    kb = db["victory_team_kb"].find_one(
        {"team": {"$regex": f"^{team}$", "$options": "i"}, "season": season},
        {"_id": 0, "embedding": 0},
    )
    if not kb:
        raise HTTPException(404, f"No VictoryProfile for {team} {season}")

    # Attach driver profiles
    drivers = list(db["victory_driver_profiles"].find(
        {"team": {"$regex": f"^{team}$", "$options": "i"}, "season": season},
        {"_id": 0, "embedding": 0},
    ))
    kb["driver_profiles"] = drivers

    # Attach car profile
    car = db["victory_car_profiles"].find_one(
        {"team": {"$regex": f"^{team}$", "$options": "i"}, "season": season},
        {"_id": 0, "embedding": 0},
    )
    kb["car_profile"] = car

    return kb


@app.post("/api/local/victory/compare")
async def victory_compare(body: dict):
    """Compare 2+ teams by cosine similarity + structured diff.

    Body: {"teams": ["McLaren", "Red Bull Racing"], "season": 2024}
    """
    teams = body.get("teams", [])
    season = body.get("season")
    if len(teams) < 2:
        raise HTTPException(400, "Provide at least 2 teams to compare")

    db = get_data_db()

    # Fetch team KB embeddings and metadata
    team_data = {}
    for team in teams:
        doc = db["victory_team_kb"].find_one(
            {"team": {"$regex": f"^{team}$", "$options": "i"}, "season": season},
            {"_id": 0},
        )
        if doc:
            team_data[doc["team"]] = doc

    if len(team_data) < 2:
        found = list(team_data.keys())
        raise HTTPException(404, f"Need 2+ teams with profiles. Found: {found}")

    # Cosine similarity matrix
    team_names = list(team_data.keys())
    similarities = []
    for i in range(len(team_names)):
        for j in range(i + 1, len(team_names)):
            v1 = np.array(team_data[team_names[i]]["embedding"])
            v2 = np.array(team_data[team_names[j]]["embedding"])
            score = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            similarities.append({
                "team_a": team_names[i],
                "team_b": team_names[j],
                "similarity": round(score, 4),
            })

    # Structured diff on metadata
    diffs = []
    for i in range(len(team_names)):
        for j in range(i + 1, len(team_names)):
            meta_a = team_data[team_names[i]].get("metadata", {})
            meta_b = team_data[team_names[j]].get("metadata", {})
            diff = _compute_metadata_diff(team_names[i], meta_a, team_names[j], meta_b)
            diffs.append(diff)

    return {
        "teams": team_names,
        "season": season,
        "similarities": similarities,
        "diffs": diffs,
    }


@app.post("/api/local/victory/search")
async def victory_search(body: dict):
    """Semantic search across team KBs.

    Body: {"query": "teams with strong brakes", "season": 2024, "k": 5}
    """
    query = body.get("query", "")
    season = body.get("season")
    k = body.get("k", 5)
    if not query:
        raise HTTPException(400, "Provide a query string")

    db = get_data_db()

    from pipeline.embeddings import NomicEmbedder
    embedder = NomicEmbedder()
    query_vec = np.array(embedder.embed_query(query))

    s_filter = {"season": season} if season else {}
    results = []
    for doc in db["victory_team_kb"].find(s_filter, {"_id": 0}):
        if "embedding" not in doc:
            continue
        vec = np.array(doc["embedding"])
        score = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-10))
        results.append({
            "team": doc["team"],
            "season": doc.get("season"),
            "score": round(score, 4),
            "narrative": doc.get("narrative", "")[:300],
            "metadata": doc.get("metadata"),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"query": query, "results": results[:k]}


@app.get("/api/local/victory/regression/{team}")
async def victory_regression(team: str, season_a: int = 2023, season_b: int = 2024):
    """Season-over-season diff for internal improvement analysis."""
    db = get_data_db()

    kb_a = db["victory_team_kb"].find_one(
        {"team": {"$regex": f"^{team}$", "$options": "i"}, "season": season_a},
        {"_id": 0, "embedding": 0},
    )
    kb_b = db["victory_team_kb"].find_one(
        {"team": {"$regex": f"^{team}$", "$options": "i"}, "season": season_b},
        {"_id": 0, "embedding": 0},
    )

    if not kb_a or not kb_b:
        missing = []
        if not kb_a:
            missing.append(str(season_a))
        if not kb_b:
            missing.append(str(season_b))
        raise HTTPException(404, f"Missing {team} profile for season(s): {', '.join(missing)}")

    diff = _compute_metadata_diff(
        f"{team} {season_a}", kb_a.get("metadata", {}),
        f"{team} {season_b}", kb_b.get("metadata", {}),
    )

    return {
        "team": team,
        "season_a": season_a,
        "season_b": season_b,
        "diff": diff,
        "narrative_a": kb_a.get("narrative", ""),
        "narrative_b": kb_b.get("narrative", ""),
    }


def _compute_metadata_diff(name_a: str, meta_a: dict, name_b: str, meta_b: dict) -> dict:
    """Compute structured diff between two team metadata blobs."""
    diff = {"teams": [name_a, name_b], "car": {}, "drivers": {}}

    # Car-level diff
    car_a = meta_a.get("car", {})
    car_b = meta_b.get("car", {})

    # Health systems diff
    sys_a = (car_a.get("health") or {}).get("systems", {})
    sys_b = (car_b.get("health") or {}).get("systems", {})
    all_systems = sorted(set(list(sys_a.keys()) + list(sys_b.keys())))
    system_diffs = []
    for sys_name in all_systems:
        va = sys_a.get(sys_name)
        vb = sys_b.get(sys_name)
        if va is not None and vb is not None:
            delta = round(vb - va, 1)
            system_diffs.append({"system": sys_name, name_a: va, name_b: vb, "delta": delta})
    diff["car"]["systems"] = system_diffs

    # Constructor stats diff
    con_a = car_a.get("constructor") or {}
    con_b = car_b.get("constructor") or {}
    con_fields = ["total_wins", "total_podiums", "dnf_rate", "avg_finish_position",
                  "avg_pit_duration_s", "q3_rate"]
    con_diffs = []
    for field in con_fields:
        va = con_a.get(field)
        vb = con_b.get(field)
        if va is not None and vb is not None:
            delta = round(vb - va, 3) if isinstance(vb, float) else vb - va
            con_diffs.append({"metric": field, name_a: va, name_b: vb, "delta": delta})
    diff["car"]["constructor"] = con_diffs

    return diff
```

**Step 2: Add `numpy` import if not already present at top of chat_server.py**

Check if `numpy` is already imported. If not, add `import numpy as np` near the top imports.

**Step 3: Add middleware rewrite for victory endpoints**

In `chat_server.py`, find the `_RewriteMiddleware` class and add `/api/victory/` to the list of paths that get rewritten to `/api/local/victory/`. This allows the frontend to call `/api/victory/...` and have it routed correctly.

Find the line like:
```python
REWRITE_PREFIXES = ["/api/openf1/", "/api/driver_intel/", ...]
```

Add `"/api/victory/"` to this list.

**Step 4: Verify endpoints**

```bash
# Start server
PYTHONPATH=pipeline:. python -m uvicorn pipeline.chat_server:app --host 0.0.0.0 --port 8300 &

# Test team KB
curl -s http://localhost:8300/api/local/victory/team/McLaren/2024 | python -m json.tool | head -30

# Test compare
curl -s -X POST http://localhost:8300/api/local/victory/compare \
  -H "Content-Type: application/json" \
  -d '{"teams": ["McLaren", "Red Bull Racing"], "season": 2024}' | python -m json.tool | head -30

# Test search
curl -s -X POST http://localhost:8300/api/local/victory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "teams with strong brakes and reliability", "season": 2024}' | python -m json.tool | head -30
```

**Step 5: Commit**

```bash
git add pipeline/chat_server.py
git commit -m "feat: add VictoryProfiles API endpoints — team KB, compare, search, regression"
```

---

## Task 3: Update Data Tracker

Add the 3 new collections to `data_tracker.html`.

**Files:**
- Modify: `data_tracker.html`

**Step 1: Add victory collections to the tracker**

Find the collections array in `data_tracker.html` and add:

```javascript
{
    name: "victory_driver_profiles",
    category: "Intelligence",
    description: "Per-driver per-team per-season profiles with Nomic embeddings — Layer 1 of VictoryProfiles",
    writers: ["build_victory_profiles.py"],
    readers: ["chat_server.py"],
    frontendConsumers: ["Team Intel (future)"],
    status: "active"
},
{
    name: "victory_car_profiles",
    category: "Intelligence",
    description: "Per-team per-season car profiles (health, telemetry, reliability) with embeddings — Layer 2",
    writers: ["build_victory_profiles.py"],
    readers: ["chat_server.py"],
    frontendConsumers: ["Team Intel (future)"],
    status: "active"
},
{
    name: "victory_team_kb",
    category: "Intelligence",
    description: "Team-wide knowledge base combining driver + car profiles with structured metadata — Layer 3",
    writers: ["build_victory_profiles.py"],
    readers: ["chat_server.py"],
    frontendConsumers: ["Team Intel (future)"],
    status: "active"
}
```

**Step 2: Commit**

```bash
git add data_tracker.html
git commit -m "docs: add VictoryProfiles collections to data tracker"
```

---

## Task 4: Integration Test

End-to-end validation of the full pipeline and API.

**Step 1: Run the full pipeline for season 2024**

```bash
cd /home/pedrad/javier_project_folder/f1
PYTHONPATH=pipeline:. python pipeline/build_victory_profiles.py --season 2024
```

Expected output:
```
=== VictoryProfiles Pipeline ===
Season: 2024
[Layer 1] Building driver profiles (season=2024)...
  Found ~20 drivers
  Embedding ~20 driver narratives...
  Upserted N new, M updated
[Layer 2] Building car profiles (season=2024)...
  Found ~10 teams
  Embedding ~10 car narratives...
  Upserted N new, M updated
[Layer 3] Building team knowledge bases (season=2024)...
  Found ~10 teams
  Embedding ~10 team narratives...
  Upserted N new, M updated
Done! X drivers, Y cars, Z teams (Ns)
```

**Step 2: Verify MongoDB collections**

```bash
python -c "
from pipeline.updater._db import get_db
db = get_db()
for col in ('victory_driver_profiles', 'victory_car_profiles', 'victory_team_kb'):
    count = db[col].count_documents({})
    sample = db[col].find_one({}, {'_id': 0, 'embedding': 0})
    print(f'{col}: {count} docs')
    if sample:
        print(f'  team: {sample.get(\"team\", sample.get(\"driver_code\", \"?\"))}')
        print(f'  keys: {sorted(sample.keys())}')
        print(f'  has narrative: {bool(sample.get(\"narrative\"))}')
        print()
"
```

**Step 3: Test API endpoints with curl**

```bash
# Team KB
curl -s http://localhost:8300/api/local/victory/team/McLaren/2024 | python -m json.tool | head -50

# Compare McLaren vs Red Bull
curl -s -X POST http://localhost:8300/api/local/victory/compare \
  -H "Content-Type: application/json" \
  -d '{"teams": ["McLaren", "Red Bull Racing"], "season": 2024}' | python -m json.tool

# Semantic search
curl -s -X POST http://localhost:8300/api/local/victory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "fastest car with best reliability", "season": 2024, "k": 3}' | python -m json.tool
```

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: VictoryProfiles pipeline adjustments from integration testing"
```
