"""
Build Constructor Performance Profiles
───────────────────────────────────────
Aggregates existing MongoDB collections into a compact team-level profile
collection that the chat_server can query directly.

Collections created:
  - constructor_profiles : per-team per-season aggregated performance
                           (~80-100 docs, one per team per season)

Source data (no API calls needed):
  - jolpica_race_results       → wins, podiums, points, positions, DNFs
  - jolpica_qualifying         → qualifying pace, Q3 rate
  - jolpica_pit_stops          → pit execution speed, strategy patterns
  - jolpica_constructor_standings → final championship position
  - telemetry_race_summary     → fleet speed, throttle, brake metrics
  - jolpica_driver_standings   → driver lineup per season

Usage:
    python3 -m pipeline.enrichment.build_constructor_profiles
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from updater._db import get_db

COL_NAME = "constructor_profiles"


# ── Helpers ─────────────────────────────────────────────────────────────

def _safe_round(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), decimals)


def _parse_lap_time(t: str | None) -> float | None:
    """Convert 'M:SS.mmm' or 'MM:SS.mmm' to seconds."""
    if not t or not isinstance(t, str):
        return None
    try:
        parts = t.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    except (ValueError, IndexError):
        pass
    return None


# ── Data loaders ────────────────────────────────────────────────────────

def load_race_results(db) -> list[dict]:
    return list(db["jolpica_race_results"].find({}, {"_id": 0}))


def load_qualifying(db) -> list[dict]:
    return list(db["jolpica_qualifying"].find({}, {"_id": 0}))


def load_pit_stops(db) -> list[dict]:
    return list(db["jolpica_pit_stops"].find({}, {"_id": 0}))


def load_constructor_standings(db) -> list[dict]:
    return list(db["jolpica_constructor_standings"].find({}, {"_id": 0}))


def load_driver_standings(db) -> list[dict]:
    return list(db["jolpica_driver_standings"].find({}, {"_id": 0}))


def load_telemetry_summaries(db) -> list[dict]:
    return list(db["telemetry_race_summary"].find({}, {"_id": 0}))


# ── Builder: driver-to-constructor mapping per season ───────────────────

def build_driver_constructor_map(race_results: list[dict]) -> dict:
    """Map (season, driver_code) → constructor_id from race results."""
    mapping = {}
    for r in race_results:
        key = (r.get("season"), r.get("driver_code"))
        if key[0] and key[1] and r.get("constructor_id"):
            mapping[key] = r["constructor_id"]
    return mapping


# ── Builder: race dimension ─────────────────────────────────────────────

def build_race_stats(race_results: list[dict]) -> dict:
    """Aggregate race results per (constructor_id, season)."""
    groups = defaultdict(list)
    for r in race_results:
        cid = r.get("constructor_id")
        season = r.get("season")
        if cid and season:
            groups[(cid, season)].append(r)

    stats = {}
    for (cid, season), results in groups.items():
        positions = []
        for r in results:
            try:
                p = int(r["position"])
                positions.append(p)
            except (KeyError, TypeError, ValueError):
                pass
        points = [float(r.get("points", 0) or 0) for r in results]
        dnf_statuses = {"Retired", "Accident", "Collision", "Engine", "Gearbox",
                        "Hydraulics", "Brakes", "Suspension", "Electrical",
                        "Mechanical", "Spun off", "Withdrew", "Did not finish"}
        dnfs = sum(1 for r in results if r.get("status") in dnf_statuses)
        grid_positions = []
        for r in results:
            try:
                g = int(r["grid"])
                if g > 0:
                    grid_positions.append(g)
            except (KeyError, TypeError, ValueError):
                pass
        positions_gained = []
        for r in results:
            try:
                pg = float(r["positions_gained"])
                positions_gained.append(pg)
            except (KeyError, TypeError, ValueError):
                pass

        # Fastest lap ranks
        fl_ranks = []
        for r in results:
            try:
                flr = int(r["fastest_lap_rank"])
                if flr <= 3:
                    fl_ranks.append(flr)
            except (KeyError, TypeError, ValueError):
                pass

        # Unique races
        race_rounds = set((r.get("round"),) for r in results)

        stats[(cid, season)] = {
            "total_entries": len(results),
            "total_races": len(race_rounds),
            "total_wins": sum(1 for p in positions if p == 1),
            "total_podiums": sum(1 for p in positions if p <= 3),
            "total_points": _safe_round(sum(points), 1),
            "avg_finish_position": _safe_round(np.mean(positions), 2) if positions else None,
            "best_finish": min(positions) if positions else None,
            "dnf_count": dnfs,
            "dnf_rate": _safe_round(dnfs / len(results) * 100, 1) if results else None,
            "avg_grid_position": _safe_round(np.mean(grid_positions), 2) if grid_positions else None,
            "avg_positions_gained": _safe_round(np.mean(positions_gained), 2) if positions_gained else None,
            "fastest_lap_top3_count": len(fl_ranks),
            "points_per_race": _safe_round(sum(points) / len(race_rounds), 2) if race_rounds else None,
        }
    return stats


# ── Builder: qualifying dimension ───────────────────────────────────────

def build_qualifying_stats(qualifying: list[dict]) -> dict:
    """Aggregate qualifying per (constructor_id, season)."""
    groups = defaultdict(list)
    for q in qualifying:
        cid = q.get("constructor_id")
        season = q.get("season")
        if cid and season:
            groups[(cid, season)].append(q)

    stats = {}
    for (cid, season), quals in groups.items():
        positions = []
        for q in quals:
            try:
                positions.append(int(q["position"]))
            except (KeyError, TypeError, ValueError):
                pass
        q3_count = sum(1 for q in quals if q.get("q3"))

        # Best qualifying times
        best_times = []
        for q in quals:
            for field in ["q3", "q2", "q1"]:
                t = _parse_lap_time(q.get(field))
                if t and t > 30:  # sanity check
                    best_times.append(t)
                    break

        stats[(cid, season)] = {
            "qual_entries": len(quals),
            "avg_qual_position": _safe_round(np.mean(positions), 2) if positions else None,
            "best_qual_position": min(positions) if positions else None,
            "q3_appearances": q3_count,
            "q3_rate": _safe_round(q3_count / len(quals) * 100, 1) if quals else None,
            "front_row_count": sum(1 for p in positions if p <= 2),
            "pole_count": sum(1 for p in positions if p == 1),
        }
    return stats


# ── Builder: pit stop dimension ─────────────────────────────────────────

def build_pit_stats(
    pit_stops: list[dict],
    driver_map: dict,
) -> dict:
    """Aggregate pit stops per (constructor_id, season)."""
    groups = defaultdict(list)
    for p in pit_stops:
        season = p.get("season")
        driver_id = p.get("driver_id")
        # Map driver_id → constructor_id via race results
        # pit_stops use driver_id, need to match
        cid = None
        # Try direct driver_code match first
        if season and driver_id:
            # driver_id in pit_stops is like "norris", "hamilton"
            # We need to find constructor from race_results
            for (s, dc), c in driver_map.items():
                if s == season and dc and driver_id.lower() in dc.lower():
                    cid = c
                    break
        if not cid:
            continue
        dur = p.get("duration_s")
        if dur and isinstance(dur, (int, float)) and 1.0 < dur < 60.0:
            groups[(cid, season)].append(dur)

    stats = {}
    for (cid, season), durations in groups.items():
        stats[(cid, season)] = {
            "total_pit_stops": len(durations),
            "avg_pit_duration_s": _safe_round(np.mean(durations), 2) if durations else None,
            "best_pit_duration_s": _safe_round(min(durations), 2) if durations else None,
            "pit_consistency": _safe_round(np.std(durations), 2) if len(durations) > 1 else None,
        }
    return stats


# ── Builder: telemetry dimension ────────────────────────────────────────

def build_telemetry_stats(
    tel_summaries: list[dict],
    driver_map: dict,
) -> dict:
    """Aggregate telemetry_race_summary per (constructor_id, season)."""
    groups = defaultdict(list)
    for t in tel_summaries:
        year = t.get("Year")
        driver = t.get("Driver")
        if year and driver:
            cid = driver_map.get((year, driver)) or driver_map.get((int(year), driver))
            if cid:
                groups[(cid, int(year))].append(t)

    stats = {}
    for (cid, season), docs in groups.items():
        avg_speeds = [d["avg_speed"] for d in docs if d.get("avg_speed")]
        top_speeds = [d["top_speed"] for d in docs if d.get("top_speed")]
        throttles = [d["avg_throttle"] for d in docs if d.get("avg_throttle")]
        brakes = [d["brake_pct"] for d in docs if d.get("brake_pct")]
        drs_pcts = [d["drs_pct"] for d in docs if d.get("drs_pct")]

        stats[(cid, season)] = {
            "tel_race_count": len(set((d.get("Race"),) for d in docs)),
            "fleet_avg_speed": _safe_round(np.mean(avg_speeds), 1) if avg_speeds else None,
            "fleet_top_speed": _safe_round(np.max(top_speeds), 1) if top_speeds else None,
            "fleet_avg_throttle": _safe_round(np.mean(throttles), 1) if throttles else None,
            "fleet_avg_brake_pct": _safe_round(np.mean(brakes), 1) if brakes else None,
            "fleet_avg_drs_pct": _safe_round(np.mean(drs_pcts), 1) if drs_pcts else None,
        }
    return stats


# ── Builder: driver lineup ──────────────────────────────────────────────

def build_lineups(driver_standings: list[dict]) -> dict:
    """Get driver lineups per (constructor_id, season)."""
    groups = defaultdict(list)
    for d in driver_standings:
        cid = d.get("constructor_id")
        season = d.get("season")
        if cid and season:
            groups[(cid, season)].append({
                "driver_code": d.get("driver_code"),
                "driver_name": d.get("driver_name"),
                "position": d.get("position"),
                "points": d.get("points"),
                "wins": d.get("wins"),
            })

    lineups = {}
    for key, drivers in groups.items():
        drivers.sort(key=lambda x: x.get("position") or 99)
        lineups[key] = drivers
    return lineups


# ── Main assembly ───────────────────────────────────────────────────────

def build_profiles(db) -> list[dict]:
    """Assemble constructor profiles from all data sources."""
    print("  Loading data sources...")
    race_results = load_race_results(db)
    qualifying = load_qualifying(db)
    pit_stops = load_pit_stops(db)
    standings = load_constructor_standings(db)
    driver_stnd = load_driver_standings(db)
    tel_summaries = load_telemetry_summaries(db)

    print(f"    race_results: {len(race_results)}")
    print(f"    qualifying: {len(qualifying)}")
    print(f"    pit_stops: {len(pit_stops)}")
    print(f"    constructor_standings: {len(standings)}")
    print(f"    driver_standings: {len(driver_stnd)}")
    print(f"    telemetry_race_summary: {len(tel_summaries)}")

    # Build driver→constructor map from race results
    driver_map = build_driver_constructor_map(race_results)

    # Build per-dimension stats
    print("  Computing race stats...")
    race_stats = build_race_stats(race_results)

    print("  Computing qualifying stats...")
    qual_stats = build_qualifying_stats(qualifying)

    print("  Computing pit stop stats...")
    pit_stats = build_pit_stats(pit_stops, driver_map)

    print("  Computing telemetry stats...")
    tel_stats = build_telemetry_stats(tel_summaries, driver_map)

    print("  Building driver lineups...")
    lineups = build_lineups(driver_stnd)

    # Build championship standings lookup
    champ_lookup = {}
    for s in standings:
        cid = s.get("constructor_id")
        season = s.get("season")
        if cid and season:
            champ_lookup[(cid, season)] = {
                "championship_position": s.get("position"),
                "championship_points": s.get("points"),
                "championship_wins": s.get("wins"),
            }

    # Get constructor metadata
    constructor_meta = {}
    for r in race_results:
        cid = r.get("constructor_id")
        if cid and cid not in constructor_meta:
            constructor_meta[cid] = {
                "constructor_name": r.get("constructor_name"),
                "nationality": r.get("nationality") if "nationality" in r else None,
            }
    # Also from standings
    for s in standings:
        cid = s.get("constructor_id")
        if cid:
            if cid not in constructor_meta:
                constructor_meta[cid] = {}
            constructor_meta[cid]["constructor_name"] = s.get("constructor_name")
            if s.get("nationality"):
                constructor_meta[cid]["nationality"] = s["nationality"]

    # Collect all (constructor_id, season) keys
    all_keys = set()
    all_keys.update(race_stats.keys())
    all_keys.update(qual_stats.keys())
    all_keys.update(champ_lookup.keys())

    now = datetime.now(timezone.utc)
    profiles = []
    for (cid, season) in sorted(all_keys):
        meta = constructor_meta.get(cid, {})
        doc = {
            "constructor_id": cid,
            "season": season,
            "constructor_name": meta.get("constructor_name", cid.replace("_", " ").title()),
            "nationality": meta.get("nationality"),
        }

        # Championship
        doc.update(champ_lookup.get((cid, season), {}))

        # Race
        doc.update(race_stats.get((cid, season), {}))

        # Qualifying
        doc.update(qual_stats.get((cid, season), {}))

        # Pit stops
        doc.update(pit_stats.get((cid, season), {}))

        # Telemetry
        doc.update(tel_stats.get((cid, season), {}))

        # Driver lineup
        doc["drivers"] = lineups.get((cid, season), [])

        # Qualifying vs race delta
        avg_qual = doc.get("avg_qual_position")
        avg_finish = doc.get("avg_finish_position")
        if avg_qual and avg_finish:
            doc["qual_to_race_delta"] = _safe_round(avg_qual - avg_finish, 2)

        doc["updated_at"] = now
        profiles.append(doc)

    return profiles


def main():
    db = get_db()
    print("✅ Connected to MongoDB")

    profiles = build_profiles(db)
    print(f"\n  Built {len(profiles)} constructor-season profiles")

    if not profiles:
        print("  ⚠ No profiles to insert")
        return

    # Upsert each profile
    from pymongo import UpdateOne
    ops = [
        UpdateOne(
            {"constructor_id": p["constructor_id"], "season": p["season"]},
            {"$set": p},
            upsert=True,
        )
        for p in profiles
    ]
    result = db[COL_NAME].bulk_write(ops)
    print(f"  Upserted: {result.upserted_count}, Modified: {result.modified_count}")

    # Indexes
    db[COL_NAME].create_index([("constructor_id", 1), ("season", 1)], unique=True)
    db[COL_NAME].create_index("season")
    db[COL_NAME].create_index("championship_position")

    # Summary
    seasons = sorted(set(p["season"] for p in profiles))
    teams = sorted(set(p["constructor_id"] for p in profiles))
    print(f"\n✅ constructor_profiles complete:")
    print(f"  Seasons: {min(seasons)}–{max(seasons)}")
    print(f"  Teams: {len(teams)}")
    print(f"  Total docs: {len(profiles)}")

    # Sample top team per season
    print("\n  Championship winners by season:")
    for s in seasons:
        winner = next((p for p in profiles
                       if p["season"] == s and p.get("championship_position") == 1), None)
        if winner:
            print(f"    {s}: {winner['constructor_name']} "
                  f"({winner.get('championship_points', '?')} pts, "
                  f"{winner.get('total_wins', '?')} wins)")


if __name__ == "__main__":
    main()
