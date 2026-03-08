"""
Backtest Replay — replay 2024 races to prove MARIP's predictive value.

For each 2024 race (in chronological order):
  1. Freeze telemetry data to only races BEFORE the target race
  2. Run anomaly ensemble on frozen data → system health scores
  3. Store pre-race predictions
  4. Compare against actual outcomes from jolpica_race_results

Usage:
    PYTHONPATH=pipeline:. python -m pipeline.backtest.replay
    PYTHONPATH=pipeline:. python -m pipeline.backtest.replay --team McLaren
    PYTHONPATH=pipeline:. python -m pipeline.backtest.replay --driver NOR
    PYTHONPATH=pipeline:. python -m pipeline.backtest.replay --race 11  # single round
"""

import argparse
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from updater._db import get_db
from pipeline.anomaly.ensemble import AnomalyDetectionEnsemble, AnomalyStatistics
from pipeline.anomaly.classifier import ClassifierPipeline
from pipeline.anomaly.run_f1_anomaly import SYSTEM_FEATURES, run_ensemble_per_system
from pipeline.anomaly.mongo_loader import load_race_summary_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BACKTEST_SEASON = 2024
TRAINING_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

# Outcomes that indicate a problem — outperformance is NOT a bad outcome
BAD_OUTCOMES = {
    "dnf_mechanical", "dnf_other", "lapped",
    "major_underperformance", "underperformance",
}


def get_race_calendar(season: int) -> list[dict]:
    """Get ordered list of races for a season from jolpica_race_results."""
    db = get_db()
    pipeline = [
        {"$match": {"season": season}},
        {"$group": {
            "_id": "$round",
            "race_name": {"$first": "$race_name"},
            "circuit_id": {"$first": "$circuit_id"},
            "date": {"$first": "$date"},
        }},
        {"$sort": {"_id": 1}},
    ]
    return [
        {"round": r["_id"], "race_name": r["race_name"],
         "circuit_id": r.get("circuit_id", ""), "date": r.get("date", "")}
        for r in db["jolpica_race_results"].aggregate(pipeline)
    ]


def get_race_results(season: int, round_num: int) -> list[dict]:
    """Get actual race results for a specific round."""
    db = get_db()
    return list(db["jolpica_race_results"].find(
        {"season": season, "round": round_num},
        {"_id": 0, "driver_code": 1, "constructor_id": 1, "constructor_name": 1,
         "grid": 1, "position": 1, "status": 1, "points": 1, "laps": 1,
         "positions_gained": 1, "fastest_lap_rank": 1},
    ))


def load_frozen_telemetry(driver_code: str, season: int, before_round: int) -> pd.DataFrame:
    """Load telemetry data frozen to before a specific round.

    Includes all training years + races from the target season that
    happened BEFORE the target round.
    """
    db = get_db()

    # Get race names for rounds before the target
    prior_races_in_season = db["jolpica_race_results"].distinct(
        "race_name",
        {"season": season, "round": {"$lt": before_round}},
    )

    # Query fastf1_laps: all training years + prior races in target season
    query = {
        "Driver": driver_code.upper(),
        "IsAccurate": True,
        "SessionType": "R",
        "$or": [
            {"Year": {"$in": TRAINING_YEARS}},
            {"Year": season, "Race": {"$in": prior_races_in_season}},
        ],
    }

    projection = {
        "_id": 0, "Year": 1, "Race": 1,
        "LapTime": 1, "Sector1Time": 1, "Sector2Time": 1, "Sector3Time": 1,
        "SpeedI1": 1, "SpeedI2": 1, "SpeedFL": 1, "SpeedST": 1,
        "TyreLife": 1, "Stint": 1, "Compound": 1,
    }

    cursor = db["fastf1_laps"].find(query, projection)
    df = pd.DataFrame(list(cursor))

    if df.empty:
        return pd.DataFrame()

    # Convert to numeric
    metric_cols = [
        "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "TyreLife",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Stint" in df.columns:
        df["Stint"] = pd.to_numeric(df["Stint"], errors="coerce")

    available = [c for c in metric_cols if c in df.columns and df[c].notna().any()]
    if not available:
        return pd.DataFrame()

    # Aggregate per race (same as mongo_loader.load_driver_race_telemetry)
    agg_funcs = {}
    for col in available:
        agg_funcs[f"{col}_mean"] = (col, "mean")
        agg_funcs[f"{col}_max"] = (col, "max")
        agg_funcs[f"{col}_std"] = (col, "std")

    if "Stint" in df.columns:
        agg_funcs["stint_count"] = ("Stint", "nunique")
    if "Compound" in df.columns:
        agg_funcs["compound_variety"] = ("Compound", "nunique")
    agg_funcs["samples"] = (available[0], "count")

    result = df.groupby(["Year", "Race"]).agg(**agg_funcs).reset_index()
    result["driver"] = driver_code.upper()
    result["race"] = result["Year"].astype(str) + " " + result["Race"]

    std_cols = [c for c in result.columns if c.endswith("_std")]
    result[std_cols] = result[std_cols].fillna(0)

    return result


def load_frozen_summary(driver_code: str, season: int, before_round: int) -> pd.DataFrame:
    """Load telemetry_race_summary frozen to before a specific round."""
    db = get_db()

    prior_races = db["jolpica_race_results"].distinct(
        "race_name",
        {"season": season, "round": {"$lt": before_round}},
    )

    query = {
        "Driver": driver_code.upper(),
        "$or": [
            {"Year": {"$in": TRAINING_YEARS}},
            {"Year": season, "Race": {"$in": prior_races}},
        ],
    }

    projection = {
        "_id": 0, "Year": 1, "Race": 1,
        "avg_rpm": 1, "max_rpm": 1, "avg_throttle": 1,
        "brake_pct": 1, "drs_pct": 1, "avg_speed": 1, "top_speed": 1,
    }

    docs = list(db["telemetry_race_summary"].find(query, projection))
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    for col in ["avg_rpm", "max_rpm", "avg_throttle", "brake_pct", "drs_pct", "avg_speed", "top_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_frozen_anomaly(driver_code: str, season: int, before_round: int) -> dict:
    """Run anomaly detection on data frozen before a specific round.

    Returns per-system health scores representing the pre-race prediction.
    """
    laps_df = load_frozen_telemetry(driver_code, season, before_round)
    if laps_df.empty:
        return {}

    summary_df = load_frozen_summary(driver_code, season, before_round)

    if not summary_df.empty:
        merged = laps_df.merge(summary_df, on=["Year", "Race"], how="left")
    else:
        merged = laps_df

    if len(merged) < 3:
        return {}

    system_results, system_col_map = run_ensemble_per_system(merged)
    if not system_results:
        return {}

    # Run classifier
    pipeline = ClassifierPipeline()
    for system, result_df in system_results.items():
        feature_cols = system_col_map.get(system, [])
        try:
            system_results[system] = pipeline.train_and_predict_system(
                merged, system, feature_cols, result_df, driver_code,
            )
        except Exception:
            pass

    # Extract latest race health (the most recent data point = trend indicator)
    latest_health = {}
    for system, result_df in system_results.items():
        if result_df.empty:
            continue
        last = result_df.iloc[-1]
        score_mean = float(last.get("Anomaly_Score_Mean", 0))
        health = max(10, min(100, int(100 - score_mean * 80)))
        level = last.get("Anomaly_Level", "normal")
        vote_count = sum(1 for col in result_df.columns
                        if col.endswith("_Anomaly") and last.get(col, 0) == 1)
        total_models = sum(1 for col in result_df.columns if col.endswith("_Anomaly"))

        latest_health[system] = {
            "health": health,
            "level": str(level),
            "score_mean": round(score_mean, 4),
            "vote_count": int(vote_count),
            "total_models": int(total_models),
        }

    # Compute trend: compare last 3 races' avg health to earlier average
    trend = {}
    for system, result_df in system_results.items():
        if len(result_df) < 5:
            continue
        scores = result_df["Anomaly_Score_Mean"].values
        recent = float(np.mean(scores[-3:]))
        earlier = float(np.mean(scores[:-3]))
        delta = recent - earlier
        if delta > 0.1:
            trend[system] = "degrading"
        elif delta < -0.1:
            trend[system] = "improving"
        else:
            trend[system] = "stable"

    overall_healths = [s["health"] for s in latest_health.values()]
    overall_health = int(np.mean(overall_healths)) if overall_healths else 0

    return {
        "systems": latest_health,
        "trends": trend,
        "overall_health": overall_health,
        "races_in_training": len(merged),
    }


def classify_result(result: dict) -> dict:
    """Classify actual race result into outcome categories."""
    status = result.get("status", "Finished")
    grid = result.get("grid", 0)
    position = result.get("position", 0)

    if isinstance(grid, str):
        grid = int(grid) if grid.isdigit() else 0
    if isinstance(position, str):
        position = int(position) if position.isdigit() else 0

    pos_delta = grid - position  # positive = gained positions

    is_dnf = status not in ("Finished", "+1 Lap", "+2 Laps", "+3 Laps", "Lapped")
    is_lapped = status == "Lapped"
    is_mechanical = status in ("Engine", "Gearbox", "Hydraulics", "Brakes",
                                "Power Unit", "Suspension", "Electrical",
                                "Overheating", "Oil leak", "Water leak",
                                "Transmission", "Retired")

    outcome = "normal"
    if is_dnf:
        outcome = "dnf_mechanical" if is_mechanical else "dnf_other"
    elif is_lapped:
        outcome = "lapped"
    elif pos_delta <= -5:
        outcome = "major_underperformance"
    elif pos_delta <= -3:
        outcome = "underperformance"
    elif pos_delta >= 5:
        outcome = "major_outperformance"
    elif pos_delta >= 3:
        outcome = "outperformance"

    return {
        "grid": grid,
        "position": position,
        "positions_gained": pos_delta,
        "status": status,
        "points": result.get("points", 0),
        "outcome": outcome,
        "is_dnf": is_dnf,
        "is_mechanical": is_mechanical,
    }


def run_backtest(
    season: int = BACKTEST_SEASON,
    driver_filter: Optional[str] = None,
    team_filter: Optional[str] = None,
    round_filter: Optional[int] = None,
) -> dict:
    """Run full backtest for a season.

    Returns structured results comparing pre-race predictions to actual outcomes.
    """
    calendar = get_race_calendar(season)
    if not calendar:
        logger.error(f"No race calendar for {season}")
        return {}

    if round_filter:
        calendar = [r for r in calendar if r["round"] == round_filter]

    logger.info(f"Backtest: {season} season, {len(calendar)} races")

    # Get grid for the season
    db = get_db()
    driver_codes = db["jolpica_race_results"].distinct("driver_code", {"season": season})

    if driver_filter:
        driver_codes = [d for d in driver_codes if d == driver_filter.upper()]
    elif team_filter:
        team_drivers = db["jolpica_race_results"].distinct(
            "driver_code",
            {"season": season, "constructor_name": {"$regex": team_filter, "$options": "i"}},
        )
        driver_codes = [d for d in driver_codes if d in team_drivers]

    logger.info(f"Drivers: {len(driver_codes)} — {sorted(driver_codes)}")

    results = []

    for race in calendar:
        round_num = race["round"]
        race_name = race["race_name"]

        if round_num < 2:
            logger.info(f"R{round_num:2d} {race_name}: Skipping (no prior data)")
            continue

        logger.info(f"\nR{round_num:2d} {race_name}")
        logger.info("=" * 60)

        actual_results = get_race_results(season, round_num)

        for driver_code in driver_codes:
            # Get actual result for this driver
            driver_actual = next(
                (r for r in actual_results if r.get("driver_code") == driver_code),
                None,
            )
            if not driver_actual:
                continue

            actual = classify_result(driver_actual)

            # ── Model 1: Anomaly ensemble (frozen) ──
            # Try rich McLaren telemetry first (much better discrimination)
            rich_anomaly = None
            try:
                from pipeline.backtest.models import compute_rich_anomaly_health
                rich_anomaly = compute_rich_anomaly_health(
                    driver_code, season, round_num, race_name,
                )
            except Exception as e:
                logger.debug(f"  {driver_code} Rich anomaly failed: {e}")

            if rich_anomaly:
                anomaly = rich_anomaly
                logger.info(f"  {driver_code}: Using rich McLaren telemetry (health={rich_anomaly['overall_health']})")
            else:
                anomaly = run_frozen_anomaly(driver_code, season, round_num)

            if not anomaly:
                logger.info(f"  {driver_code}: No anomaly data (insufficient history)")
                continue

            # ── Model 2: ELT pace prediction ──
            elt_result = None
            try:
                from pipeline.backtest.models import evaluate_elt
                total_laps = int(driver_actual.get("laps", 57) or 57)
                elt_result = evaluate_elt(race_name, season, driver_code, total_laps)
            except Exception as e:
                logger.debug(f"  {driver_code} ELT failed: {e}")

            # ── Model 3: Strategy evaluation ──
            strategy_result = None
            try:
                from pipeline.backtest.models import evaluate_strategy
                total_laps = int(driver_actual.get("laps", 57) or 57)
                strategy_result = evaluate_strategy(race_name, season, driver_code, total_laps)
            except Exception as e:
                logger.debug(f"  {driver_code} Strategy failed: {e}")

            # ── Model 4: Tyre cliff evaluation ──
            cliff_result = None
            try:
                from pipeline.backtest.models import evaluate_tyre_cliff
                cliff_result = evaluate_tyre_cliff(race_name, season, driver_code)
            except Exception as e:
                logger.debug(f"  {driver_code} Cliff failed: {e}")

            # ── Composite risk score ──
            from pipeline.backtest.models import compute_composite_risk
            composite = compute_composite_risk(
                anomaly["overall_health"], elt_result, strategy_result, cliff_result,
            )

            # Determine flagged systems from anomaly
            flagged_systems = []
            for sys_name, sys_data in anomaly.get("systems", {}).items():
                if sys_data["level"] in ("high", "critical"):
                    flagged_systems.append(sys_name)
                elif sys_data["health"] < 60:
                    flagged_systems.append(sys_name)

            degrading_systems = [
                s for s, t in anomaly.get("trends", {}).items() if t == "degrading"
            ]

            # Risk from composite (replaces simple anomaly-only check)
            predicted_risk = composite["risk_level"] in ("high", "critical")

            entry = {
                "season": season,
                "round": round_num,
                "race_name": race_name,
                "circuit_id": race.get("circuit_id", ""),
                "driver_code": driver_code,
                "constructor_id": driver_actual.get("constructor_id", ""),
                "constructor_name": driver_actual.get("constructor_name", ""),

                # Anomaly prediction
                "predicted_overall_health": anomaly["overall_health"],
                "predicted_systems": anomaly.get("systems", {}),
                "predicted_trends": anomaly.get("trends", {}),
                "flagged_systems": flagged_systems,
                "degrading_systems": degrading_systems,
                "races_in_training": anomaly.get("races_in_training", 0),
                "train_seasons": anomaly.get("train_seasons", []),
                "train_accuracy": anomaly.get("train_accuracy"),
                "cv_std": anomaly.get("cv_std"),
                "model_type": anomaly.get("model_type", "xgboost"),

                # ELT prediction
                "elt_predicted_pace": elt_result.get("predicted_pace") if elt_result else None,
                "elt_baseline": elt_result.get("baseline") if elt_result else None,
                "elt_driver_advantage": elt_result.get("driver_advantage") if elt_result else None,
                "elt_confidence_std": elt_result.get("confidence_std") if elt_result else None,
                "elt_deg_r2": elt_result.get("deg_r2") if elt_result else None,

                # Strategy prediction
                "strategy_predicted": strategy_result.get("predicted_strategy") if strategy_result else None,
                "strategy_predicted_stops": strategy_result.get("predicted_stops") if strategy_result else None,
                "strategy_actual": strategy_result.get("actual_strategy") if strategy_result else None,
                "strategy_actual_stops": strategy_result.get("actual_stops") if strategy_result else None,
                "strategy_time_delta_s": strategy_result.get("time_delta_s") if strategy_result else None,
                "strategy_match": strategy_result.get("strategy_match") if strategy_result else None,

                # Tyre cliff prediction
                "cliff_warnings": cliff_result.get("cliff_warnings") if cliff_result else None,
                "cliff_avg_deg_r2": cliff_result.get("avg_deg_r2") if cliff_result else None,

                # Composite risk
                "composite_risk": composite["composite_risk"],
                "composite_risk_level": composite["risk_level"],
                "composite_signals": composite["signals"],
                "composite_weights": composite["weights"],
                "predicted_risk": predicted_risk,

                # Actual
                "actual_grid": actual["grid"],
                "actual_position": actual["position"],
                "actual_positions_gained": actual["positions_gained"],
                "actual_status": actual["status"],
                "actual_points": actual["points"],
                "actual_outcome": actual["outcome"],
                "actual_is_dnf": actual["is_dnf"],

                # Evaluation — outperformance is a GOOD outcome, not bad
                "prediction_correct": (
                    (predicted_risk and actual["outcome"] in BAD_OUTCOMES)
                    or (not predicted_risk and actual["outcome"] not in BAD_OUTCOMES)
                ),
            }

            # Build log line
            tag = ""
            if predicted_risk and actual["outcome"] in BAD_OUTCOMES:
                tag = "HIT"
            elif not predicted_risk and actual["outcome"] in BAD_OUTCOMES:
                tag = "MISS"
            elif predicted_risk and actual["outcome"] not in BAD_OUTCOMES:
                tag = "FALSE+"

            strat_info = ""
            if strategy_result and strategy_result.get("time_delta_s") is not None:
                strat_info = f" Strat={strategy_result['time_delta_s']:+.1f}s"

            logger.info(
                f"  {driver_code}: Health={anomaly['overall_health']}% "
                f"Composite={composite['composite_risk']:.0f} ({composite['risk_level']}){strat_info} | "
                f"Grid={actual['grid']} Pos={actual['position']} "
                f"[{actual['outcome']}] {tag}"
            )

            results.append(entry)

    return {
        "season": season,
        "races_evaluated": len(set(r["round"] for r in results)),
        "total_predictions": len(results),
        "results": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def push_backtest_results(backtest_data: dict):
    """Store backtest results in MongoDB."""
    db = get_db()
    doc = {
        **backtest_data,
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }
    db["backtest_results"].insert_one(doc)
    logger.info(f"Stored backtest results in backtest_results collection")


def main():
    parser = argparse.ArgumentParser(description="MARIP Backtest — Replay 2024 Races")
    parser.add_argument("--driver", help="Single driver code (e.g. NOR)")
    parser.add_argument("--team", help="Filter by team (e.g. McLaren)")
    parser.add_argument("--race", type=int, help="Single round number (e.g. 11)")
    parser.add_argument("--no-store", action="store_true", help="Don't push to MongoDB")
    args = parser.parse_args()

    backtest_data = run_backtest(
        driver_filter=args.driver,
        team_filter=args.team,
        round_filter=args.race,
    )

    if not backtest_data or not backtest_data.get("results"):
        logger.error("No backtest results generated")
        return

    # Print summary
    from pipeline.backtest.evaluate import print_summary
    print_summary(backtest_data)

    if not args.no_store:
        # Enrich with computed metrics + case studies before storing
        from pipeline.backtest.evaluate import compute_metrics, find_case_studies
        backtest_data["metrics"] = compute_metrics(backtest_data["results"])
        backtest_data["case_studies"] = find_case_studies(backtest_data["results"])
        push_backtest_results(backtest_data)


if __name__ == "__main__":
    main()
