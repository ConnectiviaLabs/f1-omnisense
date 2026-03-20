"""
Build Opponent Profiles from Scratch
─────────────────────────────────────
Creates/rebuilds opponent_profiles, opponent_circuit_profiles, and
opponent_compound_profiles from all available MongoDB data sources.

Sources:
  - jolpica_race_results     → career stats, positions, DNF rate
  - jolpica_pit_stops        → pit strategy, undercut aggression
  - jolpica_qualifying       → quali positions, Q3 rate
  - jolpica_sprint_results   → sprint performance
  - jolpica_constructor_standings → constructor-adjusted finish
  - fastf1_laps              → tyre behavior, lap degradation (2018-2024)
  - openf1_drivers           → bio (name, nationality, dob, number)
  - openf1_stints            → compounds per stint (2023-2025)
  - openf1_position          → lap-by-lap positions (2023-2025)
  - openf1_pit               → pit timing (2023-2025)
  - telemetry_race_summary   → speed, throttle, braking aggregates
  - driver_telemetry_profiles → pre-computed telemetry metrics

Usage:
    python -m pipeline.enrichment.build_opponent_profiles [--force]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from updater._db import get_db


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_mean(vals: list) -> float | None:
    return round(statistics.mean(vals), 4) if vals else None


def _safe_median(vals: list) -> float | None:
    return round(statistics.median(vals), 4) if vals else None


def _safe_stdev(vals: list) -> float | None:
    return round(statistics.stdev(vals), 4) if len(vals) >= 2 else None


def _freq(vals: list, predicate) -> float:
    if not vals:
        return 0.0
    return round(sum(1 for v in vals if predicate(v)) / len(vals), 4)


# ── 1. Discover all drivers ───────────────────────────────────────────────

def discover_drivers(db) -> dict[str, dict]:
    """Build driver_id → {driver_code, forename, surname, ...} from all sources."""
    drivers = {}

    # From jolpica_race_results (primary — has driver_id, driver_code, driver_name)
    for doc in db["jolpica_race_results"].find(
        {}, {"driver_id": 1, "driver_code": 1, "driver_name": 1,
             "constructor_id": 1, "season": 1, "_id": 0}
    ):
        did = doc["driver_id"]
        if did not in drivers:
            name_parts = doc.get("driver_name", "").split(" ", 1)
            drivers[did] = {
                "driver_id": did,
                "driver_code": doc.get("driver_code", ""),
                "forename": name_parts[0] if name_parts else "",
                "surname": name_parts[1] if len(name_parts) > 1 else "",
                "seasons": set(),
            }
        drivers[did]["seasons"].add(doc["season"])

    # From openf1_drivers (fills in nationality, dob, number, and 2025 rookies)
    of1_seen = set()
    for doc in db["openf1_drivers"].find(
        {}, {"name_acronym": 1, "full_name": 1, "country_code": 1,
             "driver_number": 1, "team_name": 1, "session_key": 1, "_id": 0}
    ):
        code = doc.get("name_acronym", "")
        if not code or code in of1_seen:
            continue
        of1_seen.add(code)

        # Try to match to existing driver by code
        matched = None
        for did, d in drivers.items():
            if d["driver_code"] == code:
                matched = did
                break

        if matched:
            d = drivers[matched]
            if not d.get("nationality"):
                d["nationality"] = doc.get("country_code", "")
            if not d.get("driver_number"):
                d["driver_number"] = doc.get("driver_number")
        else:
            # New driver (2025 rookie) — create from OpenF1
            name = doc.get("full_name", code)
            name_parts = name.split(" ", 1)
            # Use code as driver_id for rookies
            did = code.lower()
            drivers[did] = {
                "driver_id": did,
                "driver_code": code,
                "forename": name_parts[0] if name_parts else "",
                "surname": name_parts[1] if len(name_parts) > 1 else "",
                "nationality": doc.get("country_code", ""),
                "driver_number": doc.get("driver_number"),
                "seasons": set(),
            }

    # Determine seasons from openf1_sessions for drivers not in Jolpica
    session_years = {}
    for s in db["openf1_sessions"].find({}, {"session_key": 1, "year": 1, "_id": 0}):
        session_years[s["session_key"]] = s["year"]

    of1_driver_sessions = defaultdict(set)
    for doc in db["openf1_drivers"].find({}, {"name_acronym": 1, "session_key": 1, "_id": 0}):
        code = doc.get("name_acronym", "")
        sk = doc.get("session_key")
        if code and sk and sk in session_years:
            of1_driver_sessions[code].add(session_years[sk])

    for did, d in drivers.items():
        code = d["driver_code"]
        if code in of1_driver_sessions:
            d["seasons"] |= of1_driver_sessions[code]

    # Convert seasons to sorted list
    for d in drivers.values():
        d["seasons"] = sorted(d["seasons"])

    return drivers


# ── 2. Career stats from jolpica_race_results ─────────────────────────────

def compute_career_stats(db, drivers: dict):
    """Compute career performance from jolpica_race_results."""
    print("  Computing career stats from jolpica_race_results...")

    driver_races = defaultdict(list)
    for doc in db["jolpica_race_results"].find(
        {}, {"driver_id": 1, "position": 1, "grid": 1, "points": 1,
             "status": 1, "laps": 1, "_id": 0}
    ):
        driver_races[doc["driver_id"]].append(doc)

    for did, d in drivers.items():
        races = driver_races.get(did, [])
        if not races:
            continue

        positions = [r["position"] for r in races if r.get("position")]
        grids = [r["grid"] for r in races if r.get("grid")]
        points = [r.get("points", 0) for r in races]
        gains = [r["grid"] - r["position"] for r in races
                 if r.get("grid") and r.get("position")]

        finished_statuses = {"Finished", "+1 Lap", "+2 Laps", "+3 Laps"}
        dnfs = sum(1 for r in races if r.get("status") not in finished_statuses)

        d["total_races"] = len(races)
        d["total_wins"] = sum(1 for p in positions if p == 1)
        d["total_podiums"] = sum(1 for p in positions if p and p <= 3)
        d["avg_finish_position"] = _safe_mean(positions)
        d["avg_grid_position"] = _safe_mean(grids)
        d["avg_points_per_race"] = _safe_mean(points)
        d["avg_positions_gained"] = _safe_mean(gains)
        d["dnf_rate"] = round(dnfs / len(races), 4) if races else 0


# ── 3. Qualifying stats ──────────────────────────────────────────────────

def compute_qualifying_stats(db, drivers: dict):
    """Compute qualifying metrics from jolpica_qualifying."""
    print("  Computing qualifying stats...")

    driver_quali = defaultdict(list)
    for doc in db["jolpica_qualifying"].find(
        {}, {"driver_id": 1, "position": 1, "q3": 1, "_id": 0}
    ):
        driver_quali[doc["driver_id"]].append(doc)

    for did, d in drivers.items():
        quals = driver_quali.get(did, [])
        if not quals:
            continue

        positions = [q["position"] for q in quals if q.get("position")]
        q3_count = sum(1 for q in quals if q.get("q3"))

        d["avg_quali_position"] = _safe_mean(positions)
        d["q3_appearance_rate"] = round(q3_count / len(quals), 4) if quals else 0


# ── 4. Pit strategy from jolpica_pit_stops ────────────────────────────────

def compute_pit_strategy(db, drivers: dict):
    """Compute pit stop strategy metrics."""
    print("  Computing pit strategy from jolpica_pit_stops...")

    # Group pit stops by (driver_id, season, round)
    driver_pits = defaultdict(lambda: defaultdict(list))
    for doc in db["jolpica_pit_stops"].find(
        {}, {"driver_id": 1, "season": 1, "round": 1,
             "stop": 1, "lap": 1, "duration_s": 1, "_id": 0}
    ):
        key = (doc["season"], doc["round"])
        driver_pits[doc["driver_id"]][key].append(doc)

    # Also get race lap counts for early/late classification
    race_laps = {}
    for doc in db["jolpica_race_results"].find(
        {}, {"season": 1, "round": 1, "laps": 1, "_id": 0}
    ):
        key = (doc["season"], doc["round"])
        if doc.get("laps") and doc["laps"] > 0:
            race_laps[key] = max(race_laps.get(key, 0), doc["laps"])

    for did, d in drivers.items():
        races = driver_pits.get(did, {})
        if not races:
            continue

        first_stop_laps = []
        all_durations = []
        stop_counts = []

        for race_key, pits in races.items():
            pits_sorted = sorted(pits, key=lambda p: p.get("stop", 0))
            stop_counts.append(len(pits_sorted))

            if pits_sorted and pits_sorted[0].get("lap"):
                first_stop_laps.append(pits_sorted[0]["lap"])

            for p in pits_sorted:
                dur = p.get("duration_s")
                if dur and 10 < dur < 60:  # filter outliers
                    all_durations.append(dur)

        total_race_count = len(races)

        d["avg_first_stop_lap"] = _safe_mean(first_stop_laps)
        d["median_first_stop_lap"] = _safe_median(first_stop_laps)
        d["std_first_stop_lap"] = _safe_stdev(first_stop_laps)
        d["avg_pit_duration_s"] = _safe_mean(all_durations)
        d["min_pit_duration_s"] = round(min(all_durations), 3) if all_durations else None

        # Stop frequency
        d["one_stop_freq"] = _freq(stop_counts, lambda x: x == 1)
        d["two_stop_freq"] = _freq(stop_counts, lambda x: x == 2)
        d["three_stop_freq"] = _freq(stop_counts, lambda x: x >= 3)

        # Early/late classification: first stop in first/last third of race
        early_count = 0
        late_count = 0
        for race_key, pits in races.items():
            total_laps = race_laps.get(race_key, 60)
            pits_sorted = sorted(pits, key=lambda p: p.get("stop", 0))
            if pits_sorted and pits_sorted[0].get("lap"):
                first_lap = pits_sorted[0]["lap"]
                if first_lap < total_laps * 0.33:
                    early_count += 1
                elif first_lap > total_laps * 0.66:
                    late_count += 1

        d["early_stop_freq"] = round(early_count / total_race_count, 4) if total_race_count else 0
        d["late_stop_freq"] = round(late_count / total_race_count, 4) if total_race_count else 0

        # Derived strategy scores
        esf = d.get("early_stop_freq", 0)
        tsf = d.get("two_stop_freq", 0)
        afl = d.get("avg_first_stop_lap") or 20
        d["undercut_aggression_score"] = round(
            0.4 * esf + 0.3 * tsf + 0.3 * (1.0 / max(afl, 1)), 4
        )
        d["tyre_extension_bias"] = round(
            0.5 * d.get("late_stop_freq", 0) + 0.5 * d.get("one_stop_freq", 0), 4
        )


# ── 5. Tyre behavior from fastf1_laps ────────────────────────────────────

def compute_tyre_behavior(db, drivers: dict):
    """Compute tyre usage patterns from fastf1_laps."""
    print("  Computing tyre behavior from fastf1_laps...")

    # Build code → driver_id map
    code_to_did = {d["driver_code"]: did for did, d in drivers.items() if d.get("driver_code")}

    driver_tyre = defaultdict(lambda: {"lives": [], "compounds": defaultdict(int), "deg_slopes": []})

    # Batch read — only fields we need
    for doc in db["fastf1_laps"].find(
        {"SessionType": "R", "Compound": {"$nin": [None, "", "UNKNOWN"]}},
        {"Driver": 1, "Compound": 1, "TyreLife": 1, "LapTime": 1, "_id": 0},
    ).batch_size(5000):
        code = doc.get("Driver")
        did = code_to_did.get(code)
        if not did:
            continue

        dt = driver_tyre[did]
        compound = doc.get("Compound")
        tyre_life = doc.get("TyreLife")

        if compound:
            dt["compounds"][compound] += 1
        if tyre_life and tyre_life > 0:
            dt["lives"].append(tyre_life)

    for did, dt in driver_tyre.items():
        if did not in drivers:
            continue
        d = drivers[did]

        d["avg_tyre_life"] = _safe_mean(dt["lives"])

        if dt["compounds"]:
            d["preferred_compound"] = max(dt["compounds"], key=dt["compounds"].get)

        # Long stint capability = P90 of tyre life
        if dt["lives"]:
            sorted_lives = sorted(dt["lives"])
            idx90 = int(len(sorted_lives) * 0.9)
            d["long_stint_capability"] = round(sorted_lives[min(idx90, len(sorted_lives) - 1)], 2)


# ── 6. Telemetry metrics ─────────────────────────────────────────────────

def compute_telemetry_metrics(db, drivers: dict):
    """Pull pre-computed telemetry metrics from driver_telemetry_profiles
    and telemetry_race_summary."""
    print("  Computing telemetry metrics...")

    code_to_did = {d["driver_code"]: did for did, d in drivers.items() if d.get("driver_code")}

    # From driver_telemetry_profiles (pre-aggregated)
    for doc in db["driver_telemetry_profiles"].find({}, {"_id": 0}):
        code = doc.get("Driver") or doc.get("driver_code")
        did = code_to_did.get(code)
        if not did or did not in drivers:
            continue

        d = drivers[did]
        for field in ["avg_braking_g", "max_braking_g", "avg_top_speed",
                      "avg_throttle_pct", "throttle_smoothness",
                      "brake_overlap_rate", "g_consistency"]:
            if doc.get(field) is not None:
                d[field] = round(doc[field], 4) if isinstance(doc[field], float) else doc[field]

    # From telemetry_race_summary — compute late-race deltas
    driver_tel = defaultdict(lambda: {"early": [], "late": []})
    for doc in db["telemetry_race_summary"].find(
        {}, {"Driver": 1, "avg_speed": 1, "avg_throttle": 1,
             "brake_pct": 1, "Race": 1, "_id": 0}
    ):
        code = doc.get("Driver")
        did = code_to_did.get(code)
        if not did:
            continue
        # We can't distinguish early/late race from summary alone,
        # but we can use per-race averages for driver-level stats
        if doc.get("avg_speed"):
            driver_tel[did]["early"].append(doc["avg_speed"])

    for did, dt in driver_tel.items():
        if did not in drivers:
            continue
        d = drivers[did]
        if dt["early"]:
            d.setdefault("avg_top_speed", _safe_mean(dt["early"]))


# ── 7. Position patterns from openf1_position ────────────────────────────

def compute_position_patterns(db, drivers: dict):
    """Compute position volatility, lap-1 gains, late-race loss from openf1_position."""
    print("  Computing position patterns from openf1_position...")

    code_to_did = {d["driver_code"]: did for did, d in drivers.items() if d.get("driver_code")}

    # Get race session keys
    race_sks = set(db["openf1_sessions"].distinct(
        "session_key", {"session_name": "Race"}
    ))

    # Build number → code map from openf1_drivers
    num_to_code = {}
    for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0}):
        num_to_code[doc["driver_number"]] = doc.get("name_acronym", "")

    # Group positions by (driver, session_key) → list of (date, position)
    # Read in batches to avoid memory issues
    driver_session_positions = defaultdict(lambda: defaultdict(list))

    cursor = db["openf1_position"].find(
        {"session_key": {"$in": list(race_sks)}},
        {"driver_number": 1, "session_key": 1, "position": 1, "date": 1, "_id": 0},
    ).batch_size(10000)

    count = 0
    for doc in cursor:
        dn = doc.get("driver_number")
        code = num_to_code.get(dn)
        did = code_to_did.get(code)
        if not did or not doc.get("position"):
            continue
        driver_session_positions[did][doc["session_key"]].append(doc["position"])
        count += 1
        if count % 100000 == 0:
            print(f"    ...processed {count:,} position records")

    print(f"    Total: {count:,} position records")

    for did, sessions in driver_session_positions.items():
        if did not in drivers:
            continue
        d = drivers[did]

        all_lap1_pos = []
        all_mid_pos = []
        all_final_pos = []
        all_volatilities = []
        all_lap1_to_5_gains = []
        all_late_loss = []

        for sk, positions in sessions.items():
            if len(positions) < 10:
                continue

            all_lap1_pos.append(positions[0])

            mid_idx = len(positions) // 2
            all_mid_pos.append(positions[mid_idx])
            all_final_pos.append(positions[-1])

            # Volatility = stdev of positions during race
            if len(positions) >= 2:
                all_volatilities.append(statistics.stdev(positions))

            # Lap 1-5 gain (first 5 position samples)
            if len(positions) >= 5:
                gain = positions[0] - positions[4]  # positive = gained
                all_lap1_to_5_gains.append(gain)

            # Late race loss (last 20% vs mid-race)
            late_start = int(len(positions) * 0.8)
            if late_start > mid_idx:
                late_avg = statistics.mean(positions[late_start:])
                mid_avg = statistics.mean(positions[mid_idx - 2:mid_idx + 3])
                all_late_loss.append(late_avg - mid_avg)  # positive = lost positions

        d["avg_position_lap1"] = _safe_mean(all_lap1_pos)
        d["avg_position_lap1_n"] = len(all_lap1_pos)
        d["avg_position_mid_race"] = _safe_mean(all_mid_pos)
        d["avg_final_position"] = _safe_mean(all_final_pos)
        d["position_volatility"] = _safe_mean(all_volatilities)
        d["avg_positions_gained_lap1_to_5"] = _safe_mean(all_lap1_to_5_gains)
        d["late_race_position_loss"] = _safe_mean(all_late_loss)


# ── 8. Weather-related performance ────────────────────────────────────────

def compute_weather_performance(db, drivers: dict):
    """Compute heat/humidity deltas from fastf1_laps + fastf1_weather."""
    print("  Computing weather performance deltas...")

    code_to_did = {d["driver_code"]: did for did, d in drivers.items() if d.get("driver_code")}

    # Get weather per (Year, Race) — average air temp and humidity
    race_weather = {}
    for doc in db["fastf1_weather"].find(
        {}, {"Year": 1, "Race": 1, "AirTemp": 1, "Humidity": 1, "_id": 0}
    ):
        key = (doc.get("Year"), doc.get("Race"))
        if key not in race_weather:
            race_weather[key] = {"temps": [], "humids": []}
        if doc.get("AirTemp") is not None:
            race_weather[key]["temps"].append(doc["AirTemp"])
        if doc.get("Humidity") is not None:
            race_weather[key]["humids"].append(doc["Humidity"])

    # Classify races as hot (>30°C avg) or humid (>60%)
    hot_races = set()
    humid_races = set()
    for key, w in race_weather.items():
        if w["temps"] and statistics.mean(w["temps"]) > 30:
            hot_races.add(key)
        if w["humids"] and statistics.mean(w["humids"]) > 60:
            humid_races.add(key)

    # Get driver lap times per race from fastf1_laps
    driver_race_times = defaultdict(lambda: {"hot": [], "normal": [], "humid": [], "dry": []})

    for doc in db["fastf1_laps"].find(
        {"SessionType": "R", "LapTime": {"$ne": None}},
        {"Driver": 1, "Year": 1, "Race": 1, "LapTime": 1, "_id": 0},
    ).batch_size(5000):
        code = doc.get("Driver")
        did = code_to_did.get(code)
        if not did:
            continue

        lt = doc.get("LapTime")
        if not lt or not isinstance(lt, (int, float)) or lt <= 0:
            continue

        key = (doc.get("Year"), doc.get("Race"))
        dt = driver_race_times[did]

        if key in hot_races:
            dt["hot"].append(lt)
        else:
            dt["normal"].append(lt)

        if key in humid_races:
            dt["humid"].append(lt)
        else:
            dt["dry"].append(lt)

    for did, dt in driver_race_times.items():
        if did not in drivers:
            continue
        d = drivers[did]

        # Heat delta: avg lap time in hot - avg in normal (negative = faster in heat)
        if dt["hot"] and dt["normal"]:
            d["lap_time_delta_high_heat"] = round(
                statistics.mean(dt["hot"]) - statistics.mean(dt["normal"]), 3
            )

        # Humidity delta
        if dt["humid"] and dt["dry"]:
            d["lap_time_delta_humidity"] = round(
                statistics.mean(dt["humid"]) - statistics.mean(dt["dry"]), 3
            )


# ── 9. Late-race degradation ─────────────────────────────────────────────

def compute_late_race_metrics(db, drivers: dict):
    """Compute late-race performance drop from telemetry_race_summary."""
    print("  Computing late-race metrics from telemetry...")

    code_to_did = {d["driver_code"]: did for did, d in drivers.items() if d.get("driver_code")}

    # Get per-driver per-race telemetry for late-race approximations
    driver_summaries = defaultdict(list)
    for doc in db["telemetry_race_summary"].find(
        {}, {"Driver": 1, "avg_speed": 1, "avg_throttle": 1, "brake_pct": 1, "_id": 0}
    ):
        code = doc.get("Driver")
        did = code_to_did.get(code)
        if did:
            driver_summaries[did].append(doc)

    for did, summaries in driver_summaries.items():
        if did not in drivers or len(summaries) < 2:
            continue
        d = drivers[did]

        speeds = [s["avg_speed"] for s in summaries if s.get("avg_speed")]
        throttles = [s["avg_throttle"] for s in summaries if s.get("avg_throttle")]

        # Use variance across races as a proxy for consistency
        if speeds and not d.get("avg_top_speed"):
            d["avg_top_speed"] = _safe_mean(speeds)
        if throttles and not d.get("avg_throttle_pct"):
            d["avg_throttle_pct"] = _safe_mean(throttles)


# ── 10. Constructor-adjusted finish ───────────────────────────────────────

def compute_constructor_adjusted(db, drivers: dict):
    """Compute constructor-adjusted finish from race results + standings."""
    print("  Computing constructor-adjusted finish...")

    constructor_rank = {}
    for doc in db["jolpica_constructor_standings"].find(
        {}, {"season": 1, "constructor_id": 1, "position": 1, "_id": 0}
    ):
        constructor_rank[(doc["season"], doc["constructor_id"])] = doc["position"]

    driver_adj = defaultdict(list)
    for doc in db["jolpica_race_results"].find(
        {}, {"driver_id": 1, "season": 1, "position": 1, "constructor_id": 1, "_id": 0}
    ):
        pos = doc.get("position")
        cid = doc.get("constructor_id")
        if pos and cid:
            car_rank = constructor_rank.get((doc["season"], cid))
            if car_rank:
                driver_adj[doc["driver_id"]].append(pos - car_rank)

    for did, adjustments in driver_adj.items():
        if did in drivers:
            drivers[did]["constructor_adjusted_finish"] = _safe_mean(adjustments)


# ── 11. Sprint stats ─────────────────────────────────────────────────────

def compute_sprint_stats(db, drivers: dict):
    """Compute sprint race performance."""
    print("  Computing sprint stats...")

    driver_sprint = defaultdict(lambda: {"races": 0, "positions": [], "gained": [], "points": 0})
    for doc in db["jolpica_sprint_results"].find(
        {}, {"driver_id": 1, "position": 1, "grid": 1,
             "positions_gained": 1, "points": 1, "_id": 0}
    ):
        did = doc["driver_id"]
        ds = driver_sprint[did]
        ds["races"] += 1
        if doc.get("position"):
            ds["positions"].append(doc["position"])
        if doc.get("positions_gained") is not None:
            ds["gained"].append(doc["positions_gained"])
        ds["points"] += doc.get("points", 0)

    for did, ds in driver_sprint.items():
        if did not in drivers or ds["races"] == 0:
            continue
        d = drivers[did]
        d["sprint_races"] = ds["races"]
        d["sprint_avg_finish"] = _safe_mean(ds["positions"])
        d["sprint_avg_gained"] = _safe_mean(ds["gained"])
        d["sprint_points"] = ds["points"]


# ── 12. Quali-race pace delta ─────────────────────────────────────────────

def compute_quali_race_delta(db, drivers: dict):
    """Compute qualifying vs race pace delta."""
    print("  Computing quali-race pace deltas...")

    def _parse_laptime(t):
        if not t:
            return None
        try:
            parts = t.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except (ValueError, IndexError):
            return None

    # Index qualifying best times
    driver_quali_times = defaultdict(dict)
    for doc in db["jolpica_qualifying"].find(
        {}, {"driver_id": 1, "season": 1, "round": 1, "q1": 1, "q2": 1, "q3": 1, "_id": 0}
    ):
        best = None
        for q in ["q3", "q2", "q1"]:
            t = _parse_laptime(doc.get(q))
            if t:
                best = t
                break
        if best:
            driver_quali_times[doc["driver_id"]][(doc["season"], doc["round"])] = best

    # Get race fastest laps
    driver_deltas = defaultdict(list)
    for doc in db["jolpica_race_results"].find(
        {}, {"driver_id": 1, "season": 1, "round": 1, "fastest_lap_time": 1, "_id": 0}
    ):
        did = doc["driver_id"]
        fl = _parse_laptime(doc.get("fastest_lap_time"))
        key = (doc["season"], doc["round"])
        best_q = driver_quali_times.get(did, {}).get(key)
        if fl and best_q and fl > 0 and best_q > 0:
            delta_pct = ((fl - best_q) / best_q) * 100
            driver_deltas[did].append(delta_pct)

    for did, deltas in driver_deltas.items():
        if did in drivers and deltas:
            drivers[did]["quali_race_pace_delta_pct"] = round(
                statistics.mean(deltas), 3
            )


# ── 13. Build circuit profiles ────────────────────────────────────────────

def build_circuit_profiles(db, drivers: dict) -> int:
    """Build opponent_circuit_profiles from jolpica + openf1 data."""
    print("\n  Building circuit profiles...")

    code_to_did = {d["driver_code"]: did for did, d in drivers.items() if d.get("driver_code")}

    # From jolpica_race_results
    circuit_data = defaultdict(lambda: defaultdict(lambda: {
        "finishes": [], "grids": [], "gains": [], "pit_laps": [],
        "pit_durations": [], "stop_counts": [], "races": 0,
    }))

    for doc in db["jolpica_race_results"].find(
        {}, {"driver_id": 1, "circuit_id": 1, "position": 1,
             "grid": 1, "_id": 0}
    ):
        did = doc["driver_id"]
        cid = doc["circuit_id"]
        cd = circuit_data[did][cid]
        cd["races"] += 1
        if doc.get("position"):
            cd["finishes"].append(doc["position"])
        if doc.get("grid"):
            cd["grids"].append(doc["grid"])
        if doc.get("position") and doc.get("grid"):
            cd["gains"].append(doc["grid"] - doc["position"])

    # Add pit data per circuit
    for doc in db["jolpica_pit_stops"].find(
        {}, {"driver_id": 1, "circuit_id": 1, "season": 1, "round": 1,
             "lap": 1, "duration_s": 1, "stop": 1, "_id": 0}
    ):
        did = doc["driver_id"]
        cid = doc["circuit_id"]
        cd = circuit_data[did][cid]
        if doc.get("lap"):
            cd["pit_laps"].append(doc["lap"])
        if doc.get("duration_s") and 10 < doc["duration_s"] < 60:
            cd["pit_durations"].append(doc["duration_s"])

    # Count stops per race
    race_stops = defaultdict(lambda: defaultdict(int))
    for doc in db["jolpica_pit_stops"].find(
        {}, {"driver_id": 1, "circuit_id": 1, "season": 1, "round": 1, "_id": 0}
    ):
        key = (doc["driver_id"], doc["circuit_id"], doc["season"], doc["round"])
        race_stops[key] = race_stops.get(key, 0) + 1

    # Aggregate stop counts per driver-circuit
    driver_circuit_stops = defaultdict(lambda: defaultdict(list))
    for (did, cid, _, _), count in race_stops.items():
        driver_circuit_stops[did][cid].append(count)

    # Add telemetry from telemetry_race_summary
    # Map Race name → circuit_id using jolpica data
    race_to_circuit = {}
    for doc in db["jolpica_race_results"].find(
        {}, {"race_name": 1, "circuit_id": 1, "_id": 0}
    ):
        if doc.get("race_name") and doc.get("circuit_id"):
            race_to_circuit[doc["race_name"]] = doc["circuit_id"]

    driver_circuit_tel = defaultdict(lambda: defaultdict(lambda: {"speeds": [], "braking": []}))
    for doc in db["telemetry_race_summary"].find(
        {}, {"Driver": 1, "Race": 1, "avg_speed": 1, "brake_pct": 1, "_id": 0}
    ):
        code = doc.get("Driver")
        did = code_to_did.get(code)
        race = doc.get("Race")
        cid = race_to_circuit.get(race)
        if did and cid:
            if doc.get("avg_speed"):
                driver_circuit_tel[did][cid]["speeds"].append(doc["avg_speed"])
            if doc.get("brake_pct"):
                driver_circuit_tel[did][cid]["braking"].append(doc["brake_pct"])

    # Build docs
    ops = []
    now = datetime.now(timezone.utc)

    for did, circuits in circuit_data.items():
        for cid, cd in circuits.items():
            if cd["races"] < 1:
                continue

            doc = {
                "driver_id": did,
                "circuit": cid,
                "races": cd["races"],
                "avg_finish_position": _safe_mean(cd["finishes"]),
                "avg_grid_position": _safe_mean(cd["grids"]),
                "avg_positions_gained": _safe_mean(cd["gains"]),
                "avg_first_stop_lap": _safe_mean(cd["pit_laps"]),
                "median_first_stop_lap": _safe_median(cd["pit_laps"]),
                "avg_pit_duration_s": _safe_mean(cd["pit_durations"]),
                "avg_stops_per_race": _safe_mean(driver_circuit_stops.get(did, {}).get(cid, [])),
                "updated_at": now,
            }

            # Add telemetry
            tel = driver_circuit_tel.get(did, {}).get(cid, {})
            if tel.get("speeds"):
                doc["avg_top_speed"] = _safe_mean(tel["speeds"])
            if tel.get("braking"):
                doc["avg_braking_g"] = _safe_mean(tel["braking"])

            ops.append(UpdateOne(
                {"driver_id": did, "circuit": cid},
                {"$set": doc},
                upsert=True,
            ))

    if ops:
        for i in range(0, len(ops), 500):
            db["opponent_circuit_profiles"].bulk_write(ops[i:i+500], ordered=False)

    print(f"    Wrote {len(ops)} circuit profile docs")
    return len(ops)


# ── 14. Build compound profiles ───────────────────────────────────────────

def build_compound_profiles(db, drivers: dict) -> int:
    """Build opponent_compound_profiles from fastf1_laps."""
    print("\n  Building compound profiles...")

    code_to_did = {d["driver_code"]: did for did, d in drivers.items() if d.get("driver_code")}

    compound_data = defaultdict(lambda: defaultdict(lambda: {"laps": 0, "times": [], "lives": []}))

    for doc in db["fastf1_laps"].find(
        {"SessionType": "R", "Compound": {"$nin": [None, "", "UNKNOWN"]}},
        {"Driver": 1, "Compound": 1, "TyreLife": 1, "LapTime": 1, "_id": 0},
    ).batch_size(5000):
        code = doc.get("Driver")
        did = code_to_did.get(code)
        if not did:
            continue

        compound = doc["Compound"]
        cd = compound_data[did][compound]
        cd["laps"] += 1

        lt = doc.get("LapTime")
        if lt and isinstance(lt, (int, float)) and lt > 0:
            cd["times"].append(lt)

        tl = doc.get("TyreLife")
        if tl and tl > 0:
            cd["lives"].append(tl)

    ops = []
    now = datetime.now(timezone.utc)

    for did, compounds in compound_data.items():
        for compound, cd in compounds.items():
            if cd["laps"] < 5:
                continue

            # Degradation slope: simple linear approx (time vs tyre life)
            deg_slope = None
            if len(cd["times"]) >= 10 and len(cd["lives"]) >= 10:
                # Pair up and compute slope
                pairs = list(zip(cd["lives"][:len(cd["times"])], cd["times"][:len(cd["lives"])]))
                if pairs:
                    n = len(pairs)
                    sum_x = sum(p[0] for p in pairs)
                    sum_y = sum(p[1] for p in pairs)
                    sum_xy = sum(p[0] * p[1] for p in pairs)
                    sum_x2 = sum(p[0] ** 2 for p in pairs)
                    denom = n * sum_x2 - sum_x ** 2
                    if denom != 0:
                        deg_slope = round((n * sum_xy - sum_x * sum_y) / denom, 6)

            doc = {
                "driver_id": did,
                "compound": compound,
                "total_laps": cd["laps"],
                "avg_lap_time_s": _safe_mean(cd["times"]),
                "std_lap_time_s": _safe_stdev(cd["times"]),
                "avg_tyre_life": _safe_mean(cd["lives"]),
                "degradation_slope": deg_slope,
                "updated_at": now,
            }

            ops.append(UpdateOne(
                {"driver_id": did, "compound": compound},
                {"$set": doc},
                upsert=True,
            ))

    if ops:
        for i in range(0, len(ops), 500):
            db["opponent_compound_profiles"].bulk_write(ops[i:i+500], ordered=False)

    print(f"    Wrote {len(ops)} compound profile docs")
    return len(ops)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Drop and rebuild all profiles from scratch")
    args = parser.parse_args()

    db = get_db()
    print("Connected to MongoDB\n")

    # Discover all drivers
    print("Discovering drivers from all sources...")
    drivers = discover_drivers(db)
    print(f"  Found {len(drivers)} unique drivers\n")

    # Compute all metrics
    compute_career_stats(db, drivers)
    compute_qualifying_stats(db, drivers)
    compute_pit_strategy(db, drivers)
    compute_tyre_behavior(db, drivers)
    compute_telemetry_metrics(db, drivers)
    compute_position_patterns(db, drivers)
    compute_weather_performance(db, drivers)
    compute_late_race_metrics(db, drivers)
    compute_constructor_adjusted(db, drivers)
    compute_sprint_stats(db, drivers)
    compute_quali_race_delta(db, drivers)

    # Calculate age from dob if available
    now_date = datetime.now().date()
    for d in drivers.values():
        dob = d.get("dob")
        if dob and isinstance(dob, str):
            try:
                born = datetime.strptime(dob, "%Y-%m-%d").date()
                d["age"] = (now_date - born).days // 365
            except ValueError:
                pass

    # Upsert into opponent_profiles
    print("\n  Upserting opponent_profiles...")
    now = datetime.now(timezone.utc)
    ops = []

    for did, d in drivers.items():
        d["updated_at"] = now
        # Remove internal sets
        if isinstance(d.get("seasons"), set):
            d["seasons"] = sorted(d["seasons"])

        ops.append(UpdateOne(
            {"driver_id": did},
            {"$set": d},
            upsert=True,
        ))

    if ops:
        for i in range(0, len(ops), 100):
            db["opponent_profiles"].bulk_write(ops[i:i+100], ordered=False)

    db["opponent_profiles"].create_index("driver_id", unique=True)
    db["opponent_profiles"].create_index("driver_code")

    print(f"    Wrote {len(ops)} opponent profiles")

    # Build circuit and compound profiles
    circuit_count = build_circuit_profiles(db, drivers)
    compound_count = build_compound_profiles(db, drivers)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Opponent Profile Build Complete")
    print(f"{'='*60}")
    print(f"  opponent_profiles:          {db['opponent_profiles'].count_documents({}):>6,} docs")
    print(f"  opponent_circuit_profiles:  {db['opponent_circuit_profiles'].count_documents({}):>6,} docs")
    print(f"  opponent_compound_profiles: {db['opponent_compound_profiles'].count_documents({}):>6,} docs")

    # Show sample
    sample = db["opponent_profiles"].find_one(
        {"driver_id": "norris"},
        {"_id": 0, "driver_code": 1, "total_races": 1, "seasons": 1,
         "undercut_aggression_score": 1, "tyre_extension_bias": 1,
         "avg_finish_position": 1, "position_volatility": 1},
    )
    if sample:
        print(f"\n  Sample (norris): {sample}")


if __name__ == "__main__":
    main()
