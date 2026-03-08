"""
Backtest Model Evaluators — ELT, Strategy, Tyre Cliff prediction.

Each evaluator takes frozen (pre-race) data and produces predictions
that can be compared against actual outcomes.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from updater._db import get_db

logger = logging.getLogger(__name__)


# ── Circuit ID mapping ──────────────────────────────────────────────

# jolpica circuit_id → pit_loss circuit slug
# tyre curves + ELT use race_name (e.g. "Bahrain Grand Prix")
# pit_loss uses slug (e.g. "bahrain")

def _build_circuit_map() -> dict[str, str]:
    """Build race_name → pit_loss_slug mapping from jolpica."""
    db = get_db()
    docs = db["jolpica_race_results"].aggregate([
        {"$group": {"_id": "$race_name", "circuit_id": {"$first": "$circuit_id"}}},
    ])
    return {d["_id"]: d["circuit_id"] for d in docs}


_CIRCUIT_MAP: dict[str, str] | None = None


def get_pit_loss_slug(race_name: str) -> str:
    global _CIRCUIT_MAP
    if _CIRCUIT_MAP is None:
        _CIRCUIT_MAP = _build_circuit_map()
    return _CIRCUIT_MAP.get(race_name, race_name.lower().replace(" grand prix", "").replace(" ", "_"))


# ══════════════════════════════════════════════════════════════════
# 1. ELT (Expected Lap Time) Evaluator
# ══════════════════════════════════════════════════════════════════

def _load_elt_components(race_name: str, year: int, driver_code: str) -> dict | None:
    """Load ELT components from MongoDB: baseline, driver delta, tyre curves."""
    db = get_db()

    # Circuit baseline
    bl = db["elt_parameters"].find_one(
        {"type": "circuit_baseline", "circuit": race_name, "year": year},
        {"_id": 0},
    )
    if not bl:
        # Fallback: any year for this circuit
        bl = db["elt_parameters"].find_one(
            {"type": "circuit_baseline", "circuit": race_name},
            {"_id": 0},
            sort=[("year", -1)],
        )
    if not bl:
        return None

    # Driver delta (year-specific first, then all-time)
    dd = db["elt_parameters"].find_one(
        {"type": "driver_year_delta", "driver": driver_code, "year": year},
        {"_id": 0},
    )
    if not dd:
        dd = db["elt_parameters"].find_one(
            {"type": "driver_delta", "driver": driver_code},
            {"_id": 0},
        )
    driver_delta = dd["avg_delta"] if dd else 0.0

    return {
        "baseline": bl["baseline_lap_time"],
        "baseline_std": bl.get("baseline_std", 1.0),
        "driver_delta": driver_delta,
    }


def _load_tyre_curve(race_name: str, compound: str, temp_band: str = "all") -> dict | None:
    """Load tyre degradation curve from MongoDB."""
    db = get_db()
    for tb in [temp_band, "all"]:
        doc = db["tyre_degradation_curves"].find_one(
            {"circuit": race_name, "compound": compound, "temp_band": tb},
            {"_id": 0},
        )
        if doc:
            return doc
    # Global fallback
    doc = db["tyre_degradation_curves"].find_one(
        {"circuit": "_global", "compound": compound},
        {"_id": 0},
    )
    return doc


def compute_elt(
    race_name: str, year: int, driver_code: str,
    compound: str, tyre_life: int, lap_number: int,
) -> dict | None:
    """Compute Expected Lap Time for a specific point in a race.

    ELT = baseline + tyre_degradation(compound, tyre_life) + fuel_effect(lap) + driver_delta
    """
    components = _load_elt_components(race_name, year, driver_code)
    if not components:
        return None

    curve = _load_tyre_curve(race_name, compound)
    if not curve:
        return None

    base_time = components["baseline"]

    # Tyre degradation: polynomial evaluation
    coeffs = curve["coefficients"]
    intercept = curve["intercept"]
    # coefficients are stored as [c1, c2] for degree 2: y = c1*x + c2*x^2 + intercept
    poly_coeffs = list(reversed(coeffs)) + [intercept]
    tyre_delta = float(np.polyval(poly_coeffs, tyre_life))

    # Fuel effect (linear: lighter car = faster)
    fuel_effect = curve.get("fuel_effect_per_lap", -0.05)
    baseline_midpoint = 15  # approximate
    fuel_delta = fuel_effect * (lap_number - baseline_midpoint)

    driver_delta = components["driver_delta"]

    elt = base_time + tyre_delta + fuel_delta + driver_delta
    confidence_std = curve.get("residual_std", 1.0)

    return {
        "elt": round(float(elt), 3),
        "base_time": round(float(base_time), 3),
        "tyre_delta": round(float(tyre_delta), 3),
        "fuel_delta": round(float(fuel_delta), 3),
        "driver_delta": round(float(driver_delta), 3),
        "confidence_std": round(float(confidence_std), 3),
        "deg_r2": curve.get("r2", 0),
    }


def evaluate_elt(
    race_name: str, year: int, driver_code: str,
    total_laps: int = 57,
) -> dict | None:
    """Evaluate ELT prediction for an entire race.

    Computes expected race pace at mid-race conditions and returns
    predicted performance metrics.
    """
    # Compute ELT at representative points (fresh medium tyre, mid-race)
    for compound in ["MEDIUM", "HARD", "SOFT"]:
        result = compute_elt(race_name, year, driver_code, compound, 10, total_laps // 2)
        if result:
            break
    else:
        return None

    # Also compute expected stint performance
    stints = {}
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        curve = _load_tyre_curve(race_name, compound)
        if not curve:
            continue
        cliff_lap = curve.get("cliff_lap")
        max_life = curve.get("max_tyre_life", 30)
        stints[compound] = {
            "cliff_lap": int(cliff_lap) if cliff_lap else None,
            "max_tyre_life": int(max_life),
            "deg_r2": round(curve.get("r2", 0), 4),
        }

    return {
        "predicted_pace": result["elt"],
        "baseline": result["base_time"],
        "driver_advantage": result["driver_delta"],
        "confidence_std": result["confidence_std"],
        "deg_r2": result["deg_r2"],
        "compound_stints": stints,
    }


# ══════════════════════════════════════════════════════════════════
# 2. Strategy Evaluator
# ══════════════════════════════════════════════════════════════════

def _get_pit_loss(race_name: str) -> float:
    """Get pit loss time for a circuit."""
    db = get_db()
    slug = get_pit_loss_slug(race_name)
    doc = db["circuit_pit_loss_times"].find_one({"circuit": slug}, {"_id": 0})
    if doc:
        return float(doc.get("est_pit_lane_loss_s", doc.get("median_pit_loss", 22.0)))
    return 22.0


def _simulate_strategy_time(
    race_name: str, total_laps: int, baseline_pace: float,
    strategy: list[tuple[str, int]],
) -> float | None:
    """Simulate total race time for a given strategy.

    strategy: list of (compound, stint_length) tuples
    Returns total time in seconds or None if curves missing.
    """
    total_time = 0.0
    pit_loss_s = _get_pit_loss(race_name)
    current_lap = 1

    for i, (compound, stint_laps) in enumerate(strategy):
        curve = _load_tyre_curve(race_name, compound)
        if not curve:
            return None

        coeffs = curve["coefficients"]
        intercept = curve["intercept"]
        poly_coeffs = list(reversed(coeffs)) + [intercept]
        fuel_effect = curve.get("fuel_effect_per_lap", -0.05)

        for lap_in_stint in range(stint_laps):
            if current_lap > total_laps:
                break

            tyre_life = lap_in_stint + 1
            deg = float(np.polyval(poly_coeffs, tyre_life))
            fuel_save = fuel_effect * (current_lap - total_laps // 2)
            lap_time = baseline_pace + deg - fuel_save

            if i > 0 and lap_in_stint == 0:
                lap_time += pit_loss_s

            total_time += lap_time
            current_lap += 1

    return round(total_time, 3)


def _generate_strategies(total_laps: int, min_stint: int = 8) -> list[list[tuple[str, int]]]:
    """Generate valid 1-stop and 2-stop strategies (must use 2+ compounds)."""
    compounds = ["SOFT", "MEDIUM", "HARD"]
    strategies = []

    # 1-stop strategies
    for c1 in compounds:
        for c2 in compounds:
            if c1 == c2:
                continue
            for split_pct in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
                s1 = max(min_stint, int(total_laps * split_pct))
                s2 = total_laps - s1
                if s2 >= min_stint:
                    strategies.append([(c1, s1), (c2, s2)])

    # 2-stop strategies (smaller set for speed)
    for c1 in compounds:
        for c2 in compounds:
            for c3 in compounds:
                if len({c1, c2, c3}) < 2:
                    continue
                third = total_laps // 3
                for pattern in [(third, third, total_laps - 2 * third),
                                (third - 5, third + 5, total_laps - 2 * third)]:
                    if all(s >= min_stint for s in pattern):
                        strategies.append([(c1, pattern[0]), (c2, pattern[1]), (c3, pattern[2])])

    return strategies


def evaluate_strategy(
    race_name: str, year: int, driver_code: str,
    total_laps: int = 57,
) -> dict | None:
    """Predict optimal strategy and compare to actual stints.

    Returns predicted optimal strategy, actual strategy used,
    and the time delta between them.
    """
    db = get_db()

    # Get driver baseline pace from fastf1_laps (clean laps)
    laps = list(db["fastf1_laps"].find(
        {"Race": race_name, "Year": year, "Driver": driver_code,
         "IsAccurate": True, "SessionType": "R"},
        {"_id": 0, "LapTime": 1},
    ))
    if not laps:
        return None

    lap_times = pd.to_numeric(
        pd.Series([l.get("LapTime") for l in laps]), errors="coerce"
    ).dropna()

    if len(lap_times) < 5:
        return None

    # Baseline: 10th-25th percentile (clean-air representative pace)
    q10 = lap_times.quantile(0.10)
    q75 = lap_times.quantile(0.75)
    clean = lap_times[(lap_times >= q10) & (lap_times <= q75)]
    baseline = float(clean.quantile(0.25)) if len(clean) >= 3 else float(lap_times.median())

    # Simulate all strategies
    strategies = _generate_strategies(total_laps)
    best_time = float("inf")
    best_strategy = None

    for strat in strategies:
        t = _simulate_strategy_time(race_name, total_laps, baseline, strat)
        if t is not None and t < best_time:
            best_time = t
            best_strategy = strat

    if not best_strategy:
        return None

    # Get actual stints from openf1
    session = db["openf1_sessions"].find_one(
        {"year": year, "session_type": "Race"},
        {"_id": 0, "session_key": 1, "circuit_short_name": 1},
    )
    actual_stints = []
    if session:
        # Find driver number
        drv = db["openf1_drivers"].find_one(
            {"session_key": session["session_key"], "name_acronym": driver_code},
            {"_id": 0, "driver_number": 1},
        )
        if drv:
            stints = list(db["openf1_stints"].find(
                {"session_key": session["session_key"], "driver_number": drv["driver_number"]},
                {"_id": 0, "compound": 1, "lap_start": 1, "lap_end": 1, "stint_number": 1},
            ).sort("stint_number", 1))
            for s in stints:
                if s.get("compound") and s.get("lap_start") is not None and s.get("lap_end") is not None:
                    length = int(s["lap_end"]) - int(s["lap_start"]) + 1
                    actual_stints.append((s["compound"], length))

    # Simulate actual strategy time — use same lap count for fair comparison
    actual_time = None
    actual_laps = sum(n for _, n in actual_stints) if actual_stints else total_laps
    compare_laps = min(actual_laps, total_laps)  # fair comparison over same distance

    if actual_stints:
        actual_time = _simulate_strategy_time(race_name, compare_laps, baseline, actual_stints)

    # Re-simulate predicted strategy over same lap count for apples-to-apples
    predicted_time_fair = _simulate_strategy_time(race_name, compare_laps, baseline, best_strategy)

    predicted_str = " > ".join(f"{c}({n})" for c, n in best_strategy)
    actual_str = " > ".join(f"{c}({n})" for c, n in actual_stints) if actual_stints else "unknown"
    n_stops_predicted = len(best_strategy) - 1
    n_stops_actual = len(actual_stints) - 1 if actual_stints else -1

    strategy_match = (
        n_stops_predicted == n_stops_actual
        and set(c for c, _ in best_strategy) == set(c for c, _ in actual_stints)
    ) if actual_stints else None

    time_delta = None
    if actual_time and predicted_time_fair:
        time_delta = round(actual_time - predicted_time_fair, 1)

    return {
        "predicted_strategy": predicted_str,
        "predicted_stops": n_stops_predicted,
        "predicted_time": round(best_time, 1),
        "actual_strategy": actual_str,
        "actual_stops": n_stops_actual,
        "actual_time": round(actual_time, 1) if actual_time else None,
        "time_delta_s": time_delta,
        "strategy_match": strategy_match,
        "baseline_pace": round(baseline, 3),
        "pit_loss": _get_pit_loss(race_name),
    }


# ══════════════════════════════════════════════════════════════════
# 3. Tyre Cliff Evaluator
# ══════════════════════════════════════════════════════════════════

def evaluate_tyre_cliff(
    race_name: str, year: int, driver_code: str,
) -> dict | None:
    """Evaluate tyre cliff predictions against actual stint behaviour.

    For each compound used, compare predicted cliff lap to actual
    stint length. If driver pushed past predicted cliff, flag it.
    """
    db = get_db()

    # Get actual stints
    session = db["openf1_sessions"].find_one(
        {"year": year, "session_type": "Race"},
        {"_id": 0, "session_key": 1},
    )
    if not session:
        return None

    drv = db["openf1_drivers"].find_one(
        {"session_key": session["session_key"], "name_acronym": driver_code},
        {"_id": 0, "driver_number": 1},
    )
    if not drv:
        return None

    stints = list(db["openf1_stints"].find(
        {"session_key": session["session_key"], "driver_number": drv["driver_number"]},
        {"_id": 0, "compound": 1, "lap_start": 1, "lap_end": 1, "tyre_age_at_start": 1},
    ).sort("stint_number", 1))

    if not stints:
        return None

    stint_evals = []
    cliff_warnings = 0

    for s in stints:
        compound = s.get("compound")
        if not compound or s.get("lap_start") is None or s.get("lap_end") is None:
            continue

        stint_length = int(s["lap_end"]) - int(s["lap_start"]) + 1
        tyre_age_start = int(s.get("tyre_age_at_start", 0))
        total_tyre_life = tyre_age_start + stint_length

        curve = _load_tyre_curve(race_name, compound)
        if not curve:
            continue

        cliff_lap = curve.get("cliff_lap")
        max_life = curve.get("max_tyre_life", 30)
        deg_r2 = curve.get("r2", 0)

        # Predicted degradation at end of stint
        coeffs = curve["coefficients"]
        intercept = curve["intercept"]
        poly_coeffs = list(reversed(coeffs)) + [intercept]
        deg_at_end = float(np.polyval(poly_coeffs, total_tyre_life))
        deg_at_start = float(np.polyval(poly_coeffs, tyre_age_start + 1))

        pushed_past_cliff = cliff_lap is not None and total_tyre_life > cliff_lap
        if pushed_past_cliff:
            cliff_warnings += 1

        stint_evals.append({
            "compound": compound,
            "stint_length": stint_length,
            "total_tyre_life": total_tyre_life,
            "predicted_cliff": int(cliff_lap) if cliff_lap else None,
            "pushed_past_cliff": pushed_past_cliff,
            "deg_at_end_s": round(deg_at_end, 3),
            "deg_rate_s_per_lap": round((deg_at_end - deg_at_start) / max(stint_length - 1, 1), 4),
            "deg_r2": round(deg_r2, 4),
        })

    if not stint_evals:
        return None

    return {
        "stints": stint_evals,
        "cliff_warnings": cliff_warnings,
        "total_stints": len(stint_evals),
        "avg_deg_r2": round(np.mean([s["deg_r2"] for s in stint_evals]), 4),
    }


# ══════════════════════════════════════════════════════════════════
# 4. Composite Risk Score
# ══════════════════════════════════════════════════════════════════

def compute_composite_risk(
    anomaly_health: int,
    elt_result: dict | None,
    strategy_result: dict | None,
    cliff_result: dict | None,
) -> dict:
    """Combine all model signals into a single weighted risk score.

    Risk score: 0 (safe) to 100 (critical)

    Weights:
      - Anomaly health (inverted):  40%  — car systems degradation
      - ELT confidence gap:         20%  — pace prediction uncertainty
      - Strategy suboptimality:     25%  — running wrong strategy
      - Tyre cliff proximity:       15%  — pushing past predicted limits
    """
    signals = {}
    # Strategy weight reduced — current sim doesn't capture race events (SC, traffic),
    # so it inflates risk uniformly. Anomaly is the most discriminating signal.
    weights = {"anomaly": 0.50, "elt": 0.20, "strategy": 0.10, "cliff": 0.20}

    # Anomaly: invert health (100 = safe → 0 risk, 50 = medium → 50 risk)
    anomaly_risk = max(0, min(100, 100 - anomaly_health))
    signals["anomaly"] = anomaly_risk

    # ELT: low confidence = higher risk
    if elt_result:
        conf_std = elt_result.get("confidence_std", 1.0)
        deg_r2 = elt_result.get("deg_r2", 0.5)
        # High std = uncertain = risky; low R² = bad model fit = risky
        elt_risk = min(100, conf_std * 30 + (1 - deg_r2) * 40)
        signals["elt"] = round(elt_risk, 1)
    else:
        signals["elt"] = 50  # unknown = moderate risk
        weights["elt"] = 0.10  # reduce weight if no data

    # Strategy: time delta to optimal
    if strategy_result and strategy_result.get("time_delta_s") is not None:
        delta = abs(strategy_result["time_delta_s"])
        # 0s = 0 risk, ~30s = 50 risk, ~60s+ = 100 risk (tanh curve)
        strat_risk = min(100, float(100 * np.tanh(delta / 60)))
        signals["strategy"] = round(strat_risk, 1)
    else:
        signals["strategy"] = 30  # unknown = slight risk
        weights["strategy"] = 0.10

    # Tyre cliff: pushed past cliff = high risk
    if cliff_result:
        warnings = cliff_result.get("cliff_warnings", 0)
        total = cliff_result.get("total_stints", 1)
        cliff_risk = min(100, (warnings / max(total, 1)) * 100)
        signals["cliff"] = round(cliff_risk, 1)
    else:
        signals["cliff"] = 20
        weights["cliff"] = 0.05

    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Weighted composite
    composite = sum(signals[k] * normalized_weights[k] for k in signals)

    # Risk level — calibrated for multi-signal composite
    # With 4 signals, 40+ indicates significant risk across dimensions
    if composite >= 60:
        level = "critical"
    elif composite >= 40:
        level = "high"
    elif composite >= 25:
        level = "medium"
    elif composite >= 12:
        level = "low"
    else:
        level = "normal"

    return {
        "composite_risk": round(composite, 1),
        "risk_level": level,
        "signals": signals,
        "weights": {k: round(v, 3) for k, v in normalized_weights.items()},
    }


# ══════════════════════════════════════════════════════════════════
# 5. Rich Telemetry Anomaly (McLaren-specific)
# ══════════════════════════════════════════════════════════════════

# Uniform feature set computable from fastf1_laps across ALL years (2018-2024).
# No zeroed-out features — every feature has real data for every sample.
_UNIFORM_FEATURES = [
    "speed_trap_mean",          # mean SpeedST
    "speed_trap_std",           # std SpeedST — consistency through speed trap
    "speed_trap_max",           # max SpeedST
    "speed_i1_mean",            # mean SpeedI1 (intermediate 1)
    "speed_i2_mean",            # mean SpeedI2 (intermediate 2)
    "speed_fl_mean",            # mean SpeedFL (finish line)
    "lap_time_mean",            # mean lap time (seconds)
    "lap_time_std",             # std of lap times — consistency
    "lap_time_cv",              # coefficient of variation — normalized consistency
    "sector1_mean",             # mean sector 1 time
    "sector2_mean",             # mean sector 2 time
    "sector3_mean",             # mean sector 3 time
    "sector1_std",              # sector 1 consistency
    "sector2_std",              # sector 2 consistency
    "sector3_std",              # sector 3 consistency
    "late_race_pace_drop",      # Q4 vs Q1 lap times (positive = got slower)
    "speed_degradation",        # worn tyre pace vs fresh tyre pace
    "tyre_speed_correlation",   # correlation tyre_life ↔ lap_time
    "lap_to_lap_instability",   # mean absolute lap-to-lap time delta
    "position_range",           # max position - min position during race
]


def _classify_outcome_label(result: dict) -> int:
    """Convert a jolpica race result into 0 (good) or 1 (bad) label."""
    status = str(result.get("status", "Finished"))
    pos = int(result.get("position", 0) or 0)
    grid = int(result.get("grid", 0) or 0)
    gained = grid - pos

    if status in ("Retired", "Engine", "Gearbox", "Hydraulics", "Brakes",
                   "Power Unit", "Suspension", "Electrical", "Accident", "Collision"):
        return 1
    if "Lapped" in status:
        return 1
    if gained <= -5:
        return 1
    if gained <= -3 and pos > 10:
        return 1
    return 0


# Cache the trained model to avoid retraining per driver per race
_rich_model_cache: dict | None = None


def _parse_lap_time(lt) -> float | None:
    """Parse a LapTime value (str or numeric) to seconds."""
    if lt is None:
        return None
    if isinstance(lt, str):
        try:
            parts = lt.split()[-1].split(":")
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except Exception:
            return None
    if isinstance(lt, (int, float)):
        v = float(lt)
        return v if 60 < v < 200 else None
    return None


def _compute_fastf1_features(laps: list[dict]) -> dict:
    """Compute race features from fastf1_laps documents for a single driver-race.

    Uses ONLY fields available in fastf1_laps across all years (2018-2024):
    LapTime, SpeedST, SpeedI1, SpeedI2, SpeedFL, Sector1/2/3Time,
    TyreLife, Position.  No zeroed-out features.
    """
    import statistics

    # Parse lap times
    lap_secs = []
    for lap in laps:
        lt = _parse_lap_time(lap.get("LapTime"))
        if lt and 60 < lt < 200:
            lap_secs.append(lt)

    n_laps = len(lap_secs)
    if n_laps < 5:
        return {}

    feats: dict = {}

    # ── Speed trap features ──
    def _extract(field):
        return [float(l[field]) for l in laps
                if l.get(field) is not None and float(l.get(field, 0)) > 0]

    st = _extract("SpeedST")
    feats["speed_trap_mean"] = statistics.mean(st) if st else 0.0
    feats["speed_trap_std"] = statistics.stdev(st) if len(st) > 1 else 0.0
    feats["speed_trap_max"] = max(st) if st else 0.0

    i1 = _extract("SpeedI1")
    feats["speed_i1_mean"] = statistics.mean(i1) if i1 else 0.0

    i2 = _extract("SpeedI2")
    feats["speed_i2_mean"] = statistics.mean(i2) if i2 else 0.0

    fl = _extract("SpeedFL")
    feats["speed_fl_mean"] = statistics.mean(fl) if fl else 0.0

    # ── Lap time features ──
    feats["lap_time_mean"] = statistics.mean(lap_secs)
    feats["lap_time_std"] = statistics.stdev(lap_secs) if n_laps > 1 else 0.0
    feats["lap_time_cv"] = feats["lap_time_std"] / feats["lap_time_mean"] if feats["lap_time_mean"] > 0 else 0.0

    # ── Sector features ──
    for si, sfield in enumerate(["Sector1Time", "Sector2Time", "Sector3Time"], 1):
        svals = []
        for lap in laps:
            sv = lap.get(sfield)
            if sv is not None:
                try:
                    sv = float(sv)
                    if 5 < sv < 100:
                        svals.append(sv)
                except (ValueError, TypeError):
                    pass
        feats[f"sector{si}_mean"] = statistics.mean(svals) if svals else 0.0
        feats[f"sector{si}_std"] = statistics.stdev(svals) if len(svals) > 1 else 0.0

    # ── Late race pace drop (Q4 vs Q1 lap times) ──
    q_size = max(n_laps // 4, 1)
    first_q = lap_secs[:q_size]
    last_q = lap_secs[-q_size:]
    feats["late_race_pace_drop"] = statistics.mean(last_q) - statistics.mean(first_q)

    # ── Tyre degradation ──
    tyre_laps = []
    for lap, lt in zip(laps, lap_secs):
        tl = lap.get("TyreLife")
        if tl is not None:
            try:
                tl = float(tl)
                if tl > 0:
                    tyre_laps.append((tl, lt))
            except (ValueError, TypeError):
                pass

    if len(tyre_laps) > 10:
        fresh = [lt for tl, lt in tyre_laps if tl <= 10]
        worn = [lt for tl, lt in tyre_laps if tl > 20]
        if fresh and worn:
            feats["speed_degradation"] = statistics.mean(worn) - statistics.mean(fresh)
        else:
            feats["speed_degradation"] = 0.0
        tl_vals = [tl for tl, _ in tyre_laps]
        lt_vals = [lt for _, lt in tyre_laps]
        if len(tl_vals) > 3:
            mean_tl = statistics.mean(tl_vals)
            mean_lt = statistics.mean(lt_vals)
            n = len(tl_vals)
            cov = sum((tl - mean_tl) * (lt - mean_lt) for tl, lt in zip(tl_vals, lt_vals)) / n
            std_tl = statistics.stdev(tl_vals)
            std_lt = statistics.stdev(lt_vals)
            feats["tyre_speed_correlation"] = cov / (std_tl * std_lt) if std_tl > 0 and std_lt > 0 else 0.0
        else:
            feats["tyre_speed_correlation"] = 0.0
    else:
        feats["speed_degradation"] = 0.0
        feats["tyre_speed_correlation"] = 0.0

    # ── Lap-to-lap instability ──
    if n_laps > 2:
        diffs = [abs(lap_secs[i] - lap_secs[i - 1]) for i in range(1, n_laps)]
        feats["lap_to_lap_instability"] = statistics.mean(diffs)
    else:
        feats["lap_to_lap_instability"] = 0.0

    # ── Position range (how much the car moved around) ──
    positions = [float(l["Position"]) for l in laps
                 if l.get("Position") is not None and float(l.get("Position", 0)) > 0]
    feats["position_range"] = (max(positions) - min(positions)) if len(positions) > 1 else 0.0

    return feats


def _fetch_training_rows(season: int) -> list[dict]:
    """Gather (features, label, season) rows from fastf1_laps + jolpica outcomes."""
    db = get_db()
    feat_names = _UNIFORM_FEATURES
    train_seasons = list(range(2018, season))
    rows = []

    for yr in train_seasons:
        laps_cursor = db["fastf1_laps"].find(
            {"Year": yr, "Team": "McLaren", "SessionType": "R"},
            {"Driver": 1, "Race": 1, "LapTime": 1, "LapNumber": 1,
             "SpeedST": 1, "SpeedI1": 1, "SpeedI2": 1, "SpeedFL": 1,
             "Sector1Time": 1, "Sector2Time": 1, "Sector3Time": 1,
             "TyreLife": 1, "Position": 1, "_id": 0},
        )
        race_driver_laps: dict[tuple, list] = {}
        for lap in laps_cursor:
            key = (lap.get("Race", ""), lap.get("Driver", ""))
            race_driver_laps.setdefault(key, []).append(lap)

        for (race, driver), laps_list in race_driver_laps.items():
            if len(laps_list) < 10:
                continue
            feats = _compute_fastf1_features(laps_list)
            if not feats:
                continue
            result = db["jolpica_race_results"].find_one({
                "season": yr,
                "driver_code": driver,
                "race_name": {"$regex": race.split()[0] if race else "x", "$options": "i"},
            })
            if not result:
                continue
            label = _classify_outcome_label(result)
            feat_vec = [feats.get(f, 0.0) for f in feat_names]
            rows.append({"features": feat_vec, "label": label, "season": yr})

    return rows


def _get_or_train_rich_model(season: int) -> dict | None:
    """Train an XGBoost classifier on ALL available pre-season McLaren data.

    Uses fastf1_laps uniformly for all years (2018 onwards) with a consistent
    feature set — no mixed sources, no zeroed features.
    Validated with stratified k-fold CV. SHAP explainability built in.
    Returns {model, scaler, explainer, feature_names, train_stats} or None.
    """
    global _rich_model_cache
    if _rich_model_cache is not None and _rich_model_cache.get("trained_for") == season:
        return _rich_model_cache

    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler as SKScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import shap

    feat_names = _UNIFORM_FEATURES
    rows = _fetch_training_rows(season)

    if len(rows) < 10:
        return None

    X = np.array([r["features"] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_bad = int(y.sum())
    n_good = len(y) - n_bad

    if n_bad < 2 or n_good < 2:
        return None

    scaler = SKScaler()
    X_scaled = scaler.fit_transform(X)

    # XGBoost with tuned hyperparameters for small-medium datasets
    scale_pos = n_good / n_bad if n_bad > 0 else 1.0
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_scaled, y)

    # Stratified k-fold cross-validation (proper validation, not train accuracy)
    n_folds = min(5, n_bad, n_good)
    if n_folds >= 2:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
        cv_accuracy = round(float(cv_scores.mean()) * 100, 1)
        cv_std = round(float(cv_scores.std()) * 100, 1)
    else:
        cv_accuracy = round(float((model.predict(X_scaled) == y).mean()) * 100, 1)
        cv_std = 0.0

    # SHAP explainer for per-prediction feature importance
    explainer = shap.TreeExplainer(model)

    # XGBoost native feature importance
    importance = dict(zip(feat_names, model.feature_importances_.tolist()))

    train_seasons_actual = sorted(set(r["season"] for r in rows))

    _rich_model_cache = {
        "model": model,
        "scaler": scaler,
        "explainer": explainer,
        "feature_names": feat_names,
        "trained_for": season,
        "n_train": len(rows),
        "n_bad": n_bad,
        "n_good": n_good,
        "train_seasons": train_seasons_actual,
        "train_accuracy": cv_accuracy,
        "cv_std": cv_std,
        "feature_importance": importance,
        "model_type": "xgboost",
    }
    logger.info(
        f"XGBoost model trained: {len(rows)} samples, "
        f"{n_bad} bad, {n_good} good, seasons={train_seasons_actual}, "
        f"CV acc={cv_accuracy}% ±{cv_std}%"
    )
    return _rich_model_cache


# System mapping — groups features into interpretable car system categories
_SYSTEM_MAPPING = {
    "Straight-Line Speed": ["speed_trap_mean", "speed_trap_std", "speed_trap_max",
                            "speed_i1_mean", "speed_i2_mean", "speed_fl_mean"],
    "Race Pace": ["lap_time_mean", "lap_time_std", "lap_time_cv",
                  "late_race_pace_drop", "lap_to_lap_instability"],
    "Cornering & Sectors": ["sector1_mean", "sector2_mean", "sector3_mean",
                            "sector1_std", "sector2_std", "sector3_std"],
    "Tyre Management": ["speed_degradation", "tyre_speed_correlation"],
    "Race Craft": ["position_range"],
}


def compute_rich_anomaly_health(
    driver_code: str,
    season: int,
    before_round: int,
    race_name: str,
) -> dict | None:
    """Compute anomaly health using XGBoost + SHAP on fastf1_laps features.

    Trains on all prior-season data, predicts risk for the target race.
    SHAP values provide per-feature contribution to the risk prediction,
    replacing the old logistic regression coefficient approach.
    """
    db = get_db()

    # Get the target race's laps from fastf1_laps
    target_laps = list(db["fastf1_laps"].find(
        {"Year": season, "Team": "McLaren", "SessionType": "R",
         "Driver": driver_code,
         "Race": {"$regex": race_name.split()[0], "$options": "i"}},
        {"LapTime": 1, "SpeedST": 1, "SpeedI1": 1, "SpeedI2": 1, "SpeedFL": 1,
         "Sector1Time": 1, "Sector2Time": 1, "Sector3Time": 1,
         "TyreLife": 1, "Position": 1, "_id": 0},
    ))
    if len(target_laps) < 10:
        return None

    target_feats = _compute_fastf1_features(target_laps)
    if not target_feats:
        return None

    trained = _get_or_train_rich_model(season)
    if not trained:
        return None

    model = trained["model"]
    scaler = trained["scaler"]
    explainer = trained["explainer"]
    feat_names = trained["feature_names"]

    feat_vec = np.array([[target_feats.get(f, 0.0) for f in feat_names]], dtype=float)
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
    feat_scaled = scaler.transform(feat_vec)

    prob_bad = float(model.predict_proba(feat_scaled)[0][1])
    overall_risk = prob_bad * 100
    overall_health = max(10, min(100, int(100 - overall_risk)))

    # SHAP values for this specific prediction — explains WHY the model scored this race
    shap_values = explainer.shap_values(feat_scaled)
    # For binary classification, shap_values shape is (1, n_features)
    sv = shap_values[0] if len(shap_values.shape) == 2 else shap_values[0]

    # Per-feature risk using SHAP contribution (replaces logistic coefficients)
    feature_risks = {}
    for i, fname in enumerate(feat_names):
        shap_val = float(sv[i])
        val = target_feats.get(fname, 0.0)
        # Convert SHAP value to risk percentage: positive SHAP = pushes toward "bad"
        feat_risk = 100.0 / (1 + np.exp(-shap_val))
        feature_risks[fname] = {
            "risk": round(feat_risk, 1),
            "shap_value": round(shap_val, 4),
            "value": round(val, 4) if isinstance(val, float) else val,
            "importance": round(trained["feature_importance"].get(fname, 0.0), 4),
        }

    # Group into system categories
    systems = {}
    for sys_name, sys_feats in _SYSTEM_MAPPING.items():
        sys_risks = [feature_risks[f]["risk"] for f in sys_feats if f in feature_risks]
        sys_shap = [feature_risks[f]["shap_value"] for f in sys_feats if f in feature_risks]
        if sys_risks:
            sys_risk = float(np.mean(sys_risks))
            sys_health = max(10, min(100, int(100 - sys_risk)))
            level = (
                "critical" if sys_health < 30 else
                "high" if sys_health < 50 else
                "medium" if sys_health < 70 else
                "low" if sys_health < 85 else
                "normal"
            )
            systems[sys_name] = {
                "health": sys_health,
                "level": level,
                "score_mean": round(sys_risk / 100, 4),
                "shap_sum": round(float(np.sum(sys_shap)), 4),
                "vote_count": sum(1 for f in sys_feats if f in feature_risks and feature_risks[f]["risk"] > 60),
                "total_models": len([f for f in sys_feats if f in feature_risks]),
            }

    # Trends: compare this race vs prior same-season races for this driver
    trends = {}
    prior_races = db["fastf1_laps"].aggregate([
        {"$match": {"Year": season, "Team": "McLaren", "SessionType": "R",
                     "Driver": driver_code,
                     "Race": {"$not": {"$regex": race_name.split()[0], "$options": "i"}}}},
        {"$group": {"_id": "$Race", "laps": {"$push": "$$ROOT"}}},
    ])
    prior_features = []
    for race_group in prior_races:
        pf = _compute_fastf1_features(race_group["laps"])
        if pf:
            prior_features.append(pf)

    if len(prior_features) >= 3:
        for sys_name, sys_feats in _SYSTEM_MAPPING.items():
            current_vals = [target_feats.get(f, 0) for f in sys_feats]
            prior_vals = [np.mean([pf.get(f, 0) for pf in prior_features]) for f in sys_feats]
            diffs = [abs(c - p) for c, p in zip(current_vals, prior_vals) if p != 0]
            avg_diff = np.mean(diffs) if diffs else 0
            trends[sys_name] = "degrading" if avg_diff > 0.1 else "stable"

    return {
        "systems": systems,
        "trends": trends,
        "overall_health": overall_health,
        "races_in_training": trained["n_train"],
        "train_seasons": trained["train_seasons"],
        "train_accuracy": trained["train_accuracy"],
        "cv_std": trained["cv_std"],
        "source": "fastf1_laps",
        "model_type": "xgboost",
        "feature_risks": feature_risks,
        "feature_importance": trained["feature_importance"],
        "model_prob_bad": round(prob_bad, 4),
    }


# ══════════════════════════════════════════════════════════════════
# 6. XGBoost Lap Prediction Evaluator (McLaren only)
# ══════════════════════════════════════════════════════════════════

_xgb_model_cache = None
_xgb_meta_cache = None


def _load_xgb_for_backtest():
    """Load XGBoost model and metadata for backtest (no HTTPException)."""
    global _xgb_model_cache, _xgb_meta_cache
    if _xgb_model_cache is not None:
        return _xgb_model_cache, _xgb_meta_cache

    import pickle as _pickle
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    pkl_path = root / "colabModels" / "xgboost_predictor" / "output" / "xgboost_lap_predictor.pkl"
    meta_path = root / "colabModels" / "xgboost_predictor" / "output" / "xgboost_metadata.json"

    if not pkl_path.exists():
        logger.warning("XGBoost model not found at %s", pkl_path)
        return None, None

    import json as _json
    with open(pkl_path, "rb") as f:
        _xgb_model_cache = _pickle.load(f)
    with open(meta_path) as f:
        _xgb_meta_cache = _json.load(f)

    logger.info("XGBoost lap predictor loaded for backtest")
    return _xgb_model_cache, _xgb_meta_cache


def _get_race_context(race_name: str, year: int, driver_code: str) -> dict | None:
    """Fetch shared context data for lap prediction (weather, speeds, etc.)."""
    db = get_db()

    # Encodings from XGBoost metadata (shared with BiLSTM)
    _, meta = _load_xgb_for_backtest()
    if not meta:
        return None
    encodings = meta.get("encodings", {})
    compound_map = encodings.get("compound_map", {"SOFT": 0, "MEDIUM": 1, "HARD": 2})
    circuit_rank = encodings.get("circuit_rank", {})
    driver_rank = encodings.get("driver_rank", {})
    team_rank = encodings.get("team_rank", {})

    circuit_code = circuit_rank.get(race_name)
    driver_code_enc = driver_rank.get(driver_code.upper())
    if circuit_code is None or driver_code_enc is None:
        return None

    team_code = team_rank.get("McLaren", 3)

    # Total laps
    race_doc = db["jolpica_race_results"].find_one(
        {"race_name": race_name, "season": year, "position": 1},
        {"_id": 0, "laps": 1},
    )
    total_laps = int(race_doc["laps"]) if race_doc and race_doc.get("laps") else 57

    # Weather
    weather_doc = db["fastf1_weather"].find_one(
        {"Race": race_name, "Year": year, "SessionType": "R"},
        {"_id": 0, "TrackTemp": 1, "AirTemp": 1, "Humidity": 1, "Rainfall": 1},
    )
    if not weather_doc:
        weather_doc = db["fastf1_weather"].find_one(
            {"Race": race_name, "SessionType": "R"},
            {"_id": 0, "TrackTemp": 1, "AirTemp": 1, "Humidity": 1, "Rainfall": 1},
            sort=[("Year", -1)],
        )
    air_doc = db["race_air_density"].find_one({"race": race_name}, {"_id": 0}, sort=[("year", -1)])

    air_temp = (weather_doc or {}).get("AirTemp") or (air_doc or {}).get("avg_temp_c") or 28.0
    track_temp = (weather_doc or {}).get("TrackTemp") or air_temp + 15
    humidity = (weather_doc or {}).get("Humidity") or 40.0
    rainfall = (weather_doc or {}).get("Rainfall") or 0.0
    air_density = (air_doc or {}).get("air_density_kg_m3") or 1.18

    # Sector speeds
    ss_agg = list(db["fastf1_laps"].aggregate([
        {"$match": {"Driver": driver_code.upper(), "Race": race_name,
                     "SpeedI1": {"$gt": 0}, "SessionType": "R"}},
        {"$group": {"_id": None, "SpeedI1": {"$avg": "$SpeedI1"}, "SpeedI2": {"$avg": "$SpeedI2"},
                    "SpeedFL": {"$avg": "$SpeedFL"}, "SpeedST": {"$avg": "$SpeedST"},
                    "s1": {"$avg": "$Sector1Time"}, "s2": {"$avg": "$Sector2Time"},
                    "s3": {"$avg": "$Sector3Time"}}},
    ]))
    telem_doc = db["telemetry_race_summary"].find_one(
        {"Driver": driver_code.upper(), "Race": race_name},
        {"_id": 0, "avg_speed": 1, "top_speed": 1}, sort=[("Year", -1)],
    )
    avg_speed = (telem_doc or {}).get("avg_speed", 200.0)
    top_speed = (telem_doc or {}).get("top_speed", 310.0)

    ss = ss_agg[0] if ss_agg else {}
    speed_i1 = ss.get("SpeedI1") or avg_speed * 0.95
    speed_i2 = ss.get("SpeedI2") or avg_speed
    speed_fl = ss.get("SpeedFL") or avg_speed * 1.05
    speed_st = ss.get("SpeedST") or top_speed * 0.95

    # Baseline pace
    bl_agg = list(db["fastf1_laps"].aggregate([
        {"$match": {"Driver": driver_code.upper(), "Race": race_name,
                    "LapTime": {"$gt": 50, "$lt": 200}, "SessionType": "R"}},
        {"$group": {"_id": None, "avg": {"$avg": "$LapTime"}}},
    ]))
    baseline = bl_agg[0]["avg"] if bl_agg and bl_agg[0].get("avg") else 90.0

    s1_avg = ss.get("s1") or baseline / 3
    s2_avg = ss.get("s2") or baseline / 3
    s3_avg = ss.get("s3") or baseline / 3

    return {
        "compound_map": compound_map,
        "circuit_code": circuit_code,
        "driver_code_enc": driver_code_enc,
        "team_code": team_code,
        "total_laps": total_laps,
        "air_temp": float(air_temp),
        "track_temp": float(track_temp),
        "humidity": float(humidity),
        "rainfall": float(rainfall),
        "air_density": float(air_density),
        "speed_i1": float(speed_i1),
        "speed_i2": float(speed_i2),
        "speed_fl": float(speed_fl),
        "speed_st": float(speed_st),
        "avg_speed": float(avg_speed),
        "top_speed": float(top_speed),
        "baseline": float(baseline),
        "s1_avg": float(s1_avg),
        "s2_avg": float(s2_avg),
        "s3_avg": float(s3_avg),
    }


def _get_deg_delta(race_name: str, compound: str, tyre_life: int) -> float:
    """Calculate expected degradation delta from tyre curves."""
    db = get_db()
    deg_doc = db["tyre_degradation_curves"].find_one(
        {"circuit": race_name, "compound": compound.upper(), "temp_band": "all"},
        {"_id": 0, "coefficients": 1, "intercept": 1},
    )
    if not deg_doc:
        deg_doc = db["tyre_degradation_curves"].find_one(
            {"circuit": race_name, "compound": compound.upper()},
            {"_id": 0, "coefficients": 1, "intercept": 1},
        )
    if not deg_doc or not deg_doc.get("coefficients"):
        return 0.0
    val = deg_doc.get("intercept", 0)
    for i, c in enumerate(deg_doc["coefficients"]):
        val += c * (tyre_life ** (i + 1))
    return float(val)


def _get_actual_laps_with_stints(
    race_name: str, year: int, driver_code: str,
) -> list[dict]:
    """Get actual race laps with stint/compound info for comparison."""
    db = get_db()

    laps = list(db["fastf1_laps"].find(
        {"Race": race_name, "Year": year, "Driver": driver_code.upper(),
         "IsAccurate": True, "SessionType": "R"},
        {"_id": 0, "LapNumber": 1, "LapTime": 1, "TyreLife": 1,
         "Stint": 1, "Compound": 1, "Position": 1,
         "Sector1Time": 1, "Sector2Time": 1, "Sector3Time": 1},
    ).sort("LapNumber", 1))

    result = []
    for lap in laps:
        lt = lap.get("LapTime")
        if lt is None:
            continue
        lt = float(lt) if isinstance(lt, (int, float)) else None
        if lt is None or lt < 60 or lt > 200:
            continue
        result.append({
            "lap": int(lap.get("LapNumber", 0)),
            "actual_s": lt,
            "tyre_life": int(lap.get("TyreLife", 1)),
            "stint": int(lap.get("Stint", 1)),
            "compound": lap.get("Compound", "MEDIUM"),
            "position": int(lap.get("Position", 10)),
            "s1": float(lap["Sector1Time"]) if lap.get("Sector1Time") else None,
            "s2": float(lap["Sector2Time"]) if lap.get("Sector2Time") else None,
            "s3": float(lap["Sector3Time"]) if lap.get("Sector3Time") else None,
        })

    return result


def evaluate_xgboost_laps(
    race_name: str, year: int, driver_code: str,
) -> dict | None:
    """Evaluate XGBoost lap time predictions against actual race laps.

    Loads actual laps from fastf1_laps, predicts each lap using the
    XGBoost model with the same context, compares predicted vs actual.

    Returns MAE, RMSE, R², per-stint breakdown, and per-lap predictions.
    """
    model, meta = _load_xgb_for_backtest()
    if model is None:
        return None

    ctx = _get_race_context(race_name, year, driver_code)
    if not ctx:
        return None

    actual_laps = _get_actual_laps_with_stints(race_name, year, driver_code)
    if len(actual_laps) < 5:
        return None

    features_order = meta["features"]
    compound_map = ctx["compound_map"]

    predictions = []
    prev_laps = [ctx["baseline"]] * 3
    prev_sectors = [ctx["s1_avg"], ctx["s2_avg"], ctx["s3_avg"]]

    for lap_data in actual_laps:
        compound_code = compound_map.get(lap_data["compound"].upper(), 1)
        tyre_life = lap_data["tyre_life"]
        lap_num = lap_data["lap"]
        race_progress = lap_num / ctx["total_laps"]
        deg_delta = _get_deg_delta(race_name, lap_data["compound"], tyre_life)

        feature_dict = {
            "TyreLife": tyre_life,
            "CompoundCode": compound_code,
            "LapNumber": lap_num,
            "Position": lap_data["position"],
            "Stint": lap_data["stint"],
            "FreshTyre": 1 if tyre_life == 1 else 0,
            "RaceProgress": race_progress,
            "FuelLoad": 1.0 - race_progress,
            "TotalLaps": ctx["total_laps"],
            "SpeedI1": ctx["speed_i1"],
            "SpeedI2": ctx["speed_i2"],
            "SpeedFL": ctx["speed_fl"],
            "SpeedST": ctx["speed_st"],
            "LapTime_lag1": prev_laps[0],
            "LapTime_lag2": prev_laps[1],
            "LapTime_lag3": prev_laps[2],
            "LapTime_roll3": sum(prev_laps) / 3,
            "Sector1Time_lag1": prev_sectors[0],
            "Sector2Time_lag1": prev_sectors[1],
            "Sector3Time_lag1": prev_sectors[2],
            "ExpectedDegDelta": deg_delta,
            "TrackTemp": ctx["track_temp"],
            "AirTemp": ctx["air_temp"],
            "Humidity": ctx["humidity"],
            "Rainfall": ctx["rainfall"],
            "AirDensity": ctx["air_density"],
            "avg_speed": ctx["avg_speed"],
            "top_speed": ctx["top_speed"],
            "CircuitCode": ctx["circuit_code"],
            "DriverCode": ctx["driver_code_enc"],
            "TeamCode": ctx["team_code"],
            "TyreAgeAtStart": 0 if lap_data["stint"] == 1 else tyre_life,
            "StintNumber_of1": lap_data["stint"],
            "IsUsedTyre": 0 if lap_data["stint"] == 1 else 1,
        }

        row = [feature_dict.get(f, 0) for f in features_order]
        pred = float(model.predict(np.array([row]))[0])

        error = pred - lap_data["actual_s"]
        predictions.append({
            "lap": lap_num,
            "predicted_s": round(pred, 3),
            "actual_s": round(lap_data["actual_s"], 3),
            "error_s": round(error, 3),
            "compound": lap_data["compound"],
            "stint": lap_data["stint"],
            "tyre_life": tyre_life,
        })

        # Use ACTUAL lap time for lag features (teacher forcing for fair eval)
        actual = lap_data["actual_s"]
        prev_laps = [actual, prev_laps[0], prev_laps[1]]
        s1 = lap_data["s1"] or ctx["s1_avg"]
        s2 = lap_data["s2"] or ctx["s2_avg"]
        s3 = lap_data["s3"] or ctx["s3_avg"]
        prev_sectors = [s1, s2, s3]

    # Compute metrics
    errors = np.array([p["error_s"] for p in predictions])
    actuals = np.array([p["actual_s"] for p in predictions])
    preds = np.array([p["predicted_s"] for p in predictions])

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Per-stint breakdown
    stint_groups = {}
    for p in predictions:
        key = (p["stint"], p["compound"])
        stint_groups.setdefault(key, []).append(p["error_s"])

    per_stint = []
    for (stint_num, compound), errs in sorted(stint_groups.items()):
        per_stint.append({
            "stint": stint_num,
            "compound": compound,
            "mae": round(float(np.mean(np.abs(errs))), 3),
            "n_laps": len(errs),
        })

    return {
        "model": "xgboost",
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "n_laps": len(predictions),
        "per_stint": per_stint,
        "predictions": predictions,
    }


# ══════════════════════════════════════════════════════════════════
# 7. BiLSTM Lap Prediction Evaluator (McLaren only)
# ══════════════════════════════════════════════════════════════════

_bilstm_model_cache = None
_bilstm_scalers_cache = None
_bilstm_meta_cache = None


def _load_bilstm_for_backtest():
    """Load BiLSTM model, scalers, and metadata for backtest."""
    global _bilstm_model_cache, _bilstm_scalers_cache, _bilstm_meta_cache
    if _bilstm_model_cache is not None:
        return _bilstm_model_cache, _bilstm_scalers_cache, _bilstm_meta_cache

    import pickle as _pickle
    import json as _json
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    pt_path = root / "colabModels" / "bilstm_temporal" / "output" / "bilstm_best.pt"
    scaler_path = root / "colabModels" / "bilstm_temporal" / "output" / "bilstm_scalers.pkl"
    meta_path = root / "colabModels" / "bilstm_temporal" / "output" / "bilstm_metadata.json"

    if not pt_path.exists():
        logger.warning("BiLSTM model not found at %s", pt_path)
        return None, None, None

    import torch
    import torch.nn as nn

    with open(meta_path) as f:
        _bilstm_meta_cache = _json.load(f)
    with open(scaler_path, "rb") as f:
        _bilstm_scalers_cache = _pickle.load(f)

    arch = _bilstm_meta_cache["architecture"]

    class BiLSTMLapPredictor(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                bidirectional=True, dropout=dropout if num_layers > 1 else 0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, 64), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(64, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.head(lstm_out[:, -1, :]).squeeze(-1)

    _bilstm_model_cache = BiLSTMLapPredictor(
        input_size=arch["input_size"],
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_layers"],
        dropout=arch["dropout"],
    )
    _bilstm_model_cache.load_state_dict(
        torch.load(pt_path, map_location="cpu", weights_only=True)
    )
    _bilstm_model_cache.eval()
    logger.info("BiLSTM model loaded for backtest")
    return _bilstm_model_cache, _bilstm_scalers_cache, _bilstm_meta_cache


def evaluate_bilstm_laps(
    race_name: str, year: int, driver_code: str,
) -> dict | None:
    """Evaluate BiLSTM lap time predictions against actual race laps.

    Uses a 10-lap rolling window. First window_size laps use baseline-
    initialized context, then actual laps feed into the window.
    """
    import torch

    model, scalers, meta = _load_bilstm_for_backtest()
    if model is None:
        return None

    ctx = _get_race_context(race_name, year, driver_code)
    if not ctx:
        return None

    actual_laps = _get_actual_laps_with_stints(race_name, year, driver_code)
    if len(actual_laps) < 5:
        return None

    features = meta["features"]  # 30 features
    encodings = meta.get("encodings", {})
    compound_map = encodings.get("compound_map", {"SOFT": 0, "MEDIUM": 1, "HARD": 2})
    circuit_rank = encodings.get("circuit_rank", {})
    driver_rank = encodings.get("driver_rank", {})
    team_rank = encodings.get("team_rank", {})
    window_size = meta["architecture"]["window_size"]

    circuit_code = circuit_rank.get(race_name)
    driver_code_enc = driver_rank.get(driver_code.upper())
    if circuit_code is None or driver_code_enc is None:
        return None
    team_code = team_rank.get("McLaren", 3)

    if isinstance(scalers, dict):
        feature_scaler = scalers.get("feature_scaler") or scalers.get("scaler_X")
        target_scaler = scalers.get("target_scaler") or scalers.get("scaler_y")
    else:
        feature_scaler, target_scaler = scalers[0], scalers[1]

    def build_row(lap_data, lap_time_val):
        compound_code = compound_map.get(lap_data["compound"].upper(), 1)
        tyre_life = lap_data["tyre_life"]
        lap_num = lap_data["lap"]
        race_progress = lap_num / ctx["total_laps"]
        deg_delta = _get_deg_delta(race_name, lap_data["compound"], tyre_life)

        row_dict = {
            "LapTime": lap_time_val,
            "TyreLife": tyre_life,
            "CompoundCode": compound_code,
            "LapNumber": lap_num,
            "Position": lap_data["position"],
            "Stint": lap_data["stint"],
            "FreshTyre": 1 if tyre_life == 1 else 0,
            "RaceProgress": race_progress,
            "FuelLoad": 1.0 - race_progress,
            "SpeedI1": ctx["speed_i1"],
            "SpeedI2": ctx["speed_i2"],
            "SpeedFL": ctx["speed_fl"],
            "SpeedST": ctx["speed_st"],
            "Sector1Time": lap_data["s1"] or ctx["s1_avg"],
            "Sector2Time": lap_data["s2"] or ctx["s2_avg"],
            "Sector3Time": lap_data["s3"] or ctx["s3_avg"],
            "ExpectedDegDelta": deg_delta,
            "TrackTemp": ctx["track_temp"],
            "AirTemp": ctx["air_temp"],
            "Humidity": ctx["humidity"],
            "Rainfall": ctx["rainfall"],
            "AirDensity": ctx["air_density"],
            "avg_speed": ctx["avg_speed"],
            "top_speed": ctx["top_speed"],
            "CircuitCode": circuit_code,
            "DriverCode": driver_code_enc,
            "TeamCode": team_code,
            "TyreAgeAtStart": 0 if lap_data["stint"] == 1 else tyre_life,
            "StintNumber_of1": lap_data["stint"],
            "IsUsedTyre": 0 if lap_data["stint"] == 1 else 1,
        }
        return [row_dict.get(f, 0) for f in features]

    # Initialize window with baseline rows
    baseline_row_data = {
        "lap": 0, "actual_s": ctx["baseline"], "tyre_life": 1,
        "stint": 1, "compound": actual_laps[0]["compound"] if actual_laps else "MEDIUM",
        "position": 10, "s1": ctx["s1_avg"], "s2": ctx["s2_avg"], "s3": ctx["s3_avg"],
    }
    baseline_row = build_row(baseline_row_data, ctx["baseline"])
    window = [baseline_row[:] for _ in range(window_size)]

    predictions = []

    for lap_data in actual_laps:
        # Scale window and predict
        window_arr = np.array(window, dtype=np.float32)
        if feature_scaler is not None:
            window_scaled = feature_scaler.transform(window_arr)
        else:
            window_scaled = window_arr

        with torch.no_grad():
            inp = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(inp).item()

        # Inverse scale prediction
        if target_scaler is not None:
            pred = float(target_scaler.inverse_transform([[pred_scaled]])[0][0])
        else:
            pred = pred_scaled

        error = pred - lap_data["actual_s"]
        predictions.append({
            "lap": lap_data["lap"],
            "predicted_s": round(pred, 3),
            "actual_s": round(lap_data["actual_s"], 3),
            "error_s": round(error, 3),
            "compound": lap_data["compound"],
            "stint": lap_data["stint"],
            "tyre_life": lap_data["tyre_life"],
        })

        # Slide window: add actual lap as new row (teacher forcing)
        new_row = build_row(lap_data, lap_data["actual_s"])
        window = window[1:] + [new_row]

    # Compute metrics
    errors = np.array([p["error_s"] for p in predictions])
    actuals = np.array([p["actual_s"] for p in predictions])
    preds = np.array([p["predicted_s"] for p in predictions])

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Per-stint breakdown
    stint_groups = {}
    for p in predictions:
        key = (p["stint"], p["compound"])
        stint_groups.setdefault(key, []).append(p["error_s"])

    per_stint = []
    for (stint_num, compound), errs in sorted(stint_groups.items()):
        per_stint.append({
            "stint": stint_num,
            "compound": compound,
            "mae": round(float(np.mean(np.abs(errs))), 3),
            "n_laps": len(errs),
        })

    return {
        "model": "bilstm",
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "n_laps": len(predictions),
        "per_stint": per_stint,
        "predictions": predictions,
    }
