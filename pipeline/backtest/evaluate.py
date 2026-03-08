"""
Backtest Evaluation — analyze prediction accuracy and find case studies.

Computes:
  - Overall accuracy (correct predictions / total)
  - Precision & recall for risk flagging
  - Per-system prediction value
  - Case study candidates where MARIP foreshadowed outcomes
"""

import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# Outcomes that indicate a problem — outperformance is NOT a bad outcome
BAD_OUTCOMES = {
    "dnf_mechanical", "dnf_other", "lapped",
    "major_underperformance", "underperformance",
}


def _is_bad(outcome: str) -> bool:
    return outcome in BAD_OUTCOMES


def compute_metrics(results: list[dict]) -> dict:
    """Compute prediction accuracy metrics from backtest results."""
    if not results:
        return {}

    total = len(results)
    correct = sum(1 for r in results if r.get("prediction_correct"))

    # Risk prediction confusion matrix
    true_pos = sum(1 for r in results
                   if r.get("predicted_risk") and _is_bad(r.get("actual_outcome", "")))
    false_pos = sum(1 for r in results
                    if r.get("predicted_risk") and not _is_bad(r.get("actual_outcome", "")))
    false_neg = sum(1 for r in results
                    if not r.get("predicted_risk") and _is_bad(r.get("actual_outcome", "")))
    true_neg = sum(1 for r in results
                   if not r.get("predicted_risk") and not _is_bad(r.get("actual_outcome", "")))

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Outcome distribution
    outcomes = Counter(r.get("actual_outcome", "unknown") for r in results)

    # Per-team accuracy
    team_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        team = r.get("constructor_name", "Unknown")
        team_accuracy[team]["total"] += 1
        if r.get("prediction_correct"):
            team_accuracy[team]["correct"] += 1

    team_acc = {
        team: round(d["correct"] / d["total"] * 100, 1) if d["total"] else 0
        for team, d in team_accuracy.items()
    }

    # Multi-model metrics
    strategy_matches = [r for r in results if r.get("strategy_match") is not None]
    strategy_match_rate = (
        round(sum(1 for r in strategy_matches if r["strategy_match"]) / len(strategy_matches) * 100, 1)
        if strategy_matches else None
    )

    strategy_deltas = [r["strategy_time_delta_s"] for r in results if r.get("strategy_time_delta_s") is not None]
    avg_strategy_delta = round(sum(strategy_deltas) / len(strategy_deltas), 1) if strategy_deltas else None

    composite_scores = [r["composite_risk"] for r in results if r.get("composite_risk") is not None]
    composite_levels = Counter(r.get("composite_risk_level", "unknown") for r in results if r.get("composite_risk_level"))

    cliff_warnings_total = sum(r.get("cliff_warnings", 0) for r in results)

    elt_available = sum(1 for r in results if r.get("elt_predicted_pace") is not None)

    return {
        "total_predictions": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1) if total else 0,
        "confusion_matrix": {
            "true_positive": true_pos,
            "false_positive": false_pos,
            "false_negative": false_neg,
            "true_negative": true_neg,
        },
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1_score": round(f1 * 100, 1),
        "outcome_distribution": dict(outcomes),
        "team_accuracy": dict(team_acc),
        # Multi-model metrics
        "strategy_match_rate": strategy_match_rate,
        "avg_strategy_delta_s": avg_strategy_delta,
        "composite_risk_distribution": dict(composite_levels),
        "avg_composite_risk": round(sum(composite_scores) / len(composite_scores), 1) if composite_scores else None,
        "cliff_warnings_total": cliff_warnings_total,
        "elt_coverage": round(elt_available / total * 100, 1) if total else 0,
        "models_active": {
            "anomaly": True,
            "elt": elt_available > 0,
            "strategy": len(strategy_deltas) > 0,
            "cliff": cliff_warnings_total >= 0,
            "xgboost_laps": any(r.get("xgb_laps_mae") is not None for r in results),
            "bilstm_laps": any(r.get("bilstm_laps_mae") is not None for r in results),
        },
        # Lap prediction model metrics
        "xgb_laps_avg_mae": round(
            sum(r["xgb_laps_mae"] for r in results if r.get("xgb_laps_mae") is not None)
            / max(1, sum(1 for r in results if r.get("xgb_laps_mae") is not None)), 3
        ) if any(r.get("xgb_laps_mae") is not None for r in results) else None,
        "xgb_laps_avg_r2": round(
            sum(r["xgb_laps_r2"] for r in results if r.get("xgb_laps_r2") is not None)
            / max(1, sum(1 for r in results if r.get("xgb_laps_r2") is not None)), 4
        ) if any(r.get("xgb_laps_r2") is not None for r in results) else None,
        "bilstm_laps_avg_mae": round(
            sum(r["bilstm_laps_mae"] for r in results if r.get("bilstm_laps_mae") is not None)
            / max(1, sum(1 for r in results if r.get("bilstm_laps_mae") is not None)), 3
        ) if any(r.get("bilstm_laps_mae") is not None for r in results) else None,
        "bilstm_laps_avg_r2": round(
            sum(r["bilstm_laps_r2"] for r in results if r.get("bilstm_laps_r2") is not None)
            / max(1, sum(1 for r in results if r.get("bilstm_laps_r2") is not None)), 4
        ) if any(r.get("bilstm_laps_r2") is not None for r in results) else None,
    }


def find_case_studies(results: list[dict], team_filter: str = "mclaren") -> list[dict]:
    """Find the best case study candidates from backtest results.

    Looks for races where:
    1. MARIP flagged risk AND something bad happened (true positive)
    2. MARIP saw degrading trends AND performance dropped
    3. Notable events (DNFs, major position losses) regardless of prediction
    """
    cases = []

    for r in results:
        score = 0
        reasons = []

        # True positive — predicted risk, bad outcome
        if r.get("predicted_risk") and _is_bad(r.get("actual_outcome", "")):
            score += 3
            reasons.append(f"Correctly flagged risk: {r['actual_outcome']}")

        # Flagged degrading systems before a bad result
        degrading = r.get("degrading_systems", [])
        if degrading and _is_bad(r.get("actual_outcome", "")):
            score += 2
            reasons.append(f"Degrading systems ({', '.join(degrading)}) before {r['actual_outcome']}")

        # Specific system flagged that matches outcome
        flagged = r.get("flagged_systems", [])
        if "Power Unit" in flagged and r.get("actual_status") in ("Engine", "Power Unit"):
            score += 5
            reasons.append("Power Unit anomaly predicted engine failure")
        if "Brakes" in flagged and r.get("actual_status") == "Brakes":
            score += 5
            reasons.append("Brake anomaly predicted brake failure")
        if "Thermal" in flagged and r.get("actual_status") == "Overheating":
            score += 5
            reasons.append("Thermal anomaly predicted overheating")

        # Major underperformance
        if r.get("actual_outcome") in ("major_underperformance", "lapped"):
            score += 2
            reasons.append(f"Major underperformance: Grid {r['actual_grid']} → P{r['actual_position']}")

        # DNF
        if r.get("actual_is_dnf"):
            score += 2
            reasons.append(f"DNF: {r['actual_status']}")

        # Low health score with bad outcome
        health = r.get("predicted_overall_health", 100)
        if health < 60 and _is_bad(r.get("actual_outcome", "")):
            score += 2
            reasons.append(f"Low predicted health ({health}%) confirmed by result")

        # False negative (missed prediction) — still interesting for analysis
        if not r.get("predicted_risk") and _is_bad(r.get("actual_outcome", "")):
            score += 1
            reasons.append(f"Missed prediction: {r['actual_outcome']} not flagged")

        if team_filter:
            is_target_team = team_filter.lower() in r.get("constructor_id", "").lower()
        else:
            is_target_team = True

        if score > 0 and is_target_team:
            cases.append({
                "round": r["round"],
                "race_name": r["race_name"],
                "driver_code": r["driver_code"],
                "constructor_name": r.get("constructor_name", ""),
                "score": score,
                "reasons": reasons,
                "predicted_health": health,
                "predicted_risk": r.get("predicted_risk"),
                "flagged_systems": flagged,
                "degrading_systems": degrading,
                "actual_grid": r.get("actual_grid"),
                "actual_position": r.get("actual_position"),
                "actual_positions_gained": r.get("actual_positions_gained"),
                "actual_status": r.get("actual_status"),
                "actual_outcome": r.get("actual_outcome"),
                "actual_points": r.get("actual_points", 0),
                "actual_is_dnf": r.get("actual_is_dnf", False),
                # Multi-model signals
                "composite_risk": r.get("composite_risk"),
                "composite_risk_level": r.get("composite_risk_level"),
                "composite_signals": r.get("composite_signals", {}),
                # ELT pace
                "elt_predicted_pace": r.get("elt_predicted_pace"),
                "elt_driver_advantage": r.get("elt_driver_advantage"),
                # Strategy
                "strategy_predicted": r.get("strategy_predicted"),
                "strategy_match": r.get("strategy_match"),
                "strategy_time_delta_s": r.get("strategy_time_delta_s"),
                # Cliff
                "cliff_warnings": r.get("cliff_warnings", 0),
                # Lap prediction accuracy
                "xgb_laps_mae": r.get("xgb_laps_mae"),
                "xgb_laps_r2": r.get("xgb_laps_r2"),
                "bilstm_laps_mae": r.get("bilstm_laps_mae"),
                "bilstm_laps_r2": r.get("bilstm_laps_r2"),
                # Systems detail
                "predicted_systems": r.get("predicted_systems", {}),
            })

    # Sort by case study score (highest first)
    cases.sort(key=lambda c: c["score"], reverse=True)
    return cases


def find_system_correlations(results: list[dict]) -> dict:
    """Find correlations between system anomalies and race outcomes.

    For each system, compute how often flagging it correctly predicted
    a negative outcome.
    """
    system_stats = defaultdict(lambda: {
        "flagged_count": 0, "flagged_bad_outcome": 0,
        "degrading_count": 0, "degrading_bad_outcome": 0,
    })

    for r in results:
        bad = _is_bad(r.get("actual_outcome", ""))
        for sys in r.get("flagged_systems", []):
            system_stats[sys]["flagged_count"] += 1
            if bad:
                system_stats[sys]["flagged_bad_outcome"] += 1
        for sys in r.get("degrading_systems", []):
            system_stats[sys]["degrading_count"] += 1
            if bad:
                system_stats[sys]["degrading_bad_outcome"] += 1

    correlations = {}
    for sys, stats in system_stats.items():
        fc = stats["flagged_count"]
        dc = stats["degrading_count"]
        correlations[sys] = {
            "flag_precision": round(stats["flagged_bad_outcome"] / fc * 100, 1) if fc else 0,
            "flagged_count": fc,
            "degrading_precision": round(stats["degrading_bad_outcome"] / dc * 100, 1) if dc else 0,
            "degrading_count": dc,
        }

    return dict(correlations)


def print_summary(backtest_data: dict):
    """Print a human-readable backtest summary."""
    results = backtest_data.get("results", [])
    if not results:
        print("No results to summarize.")
        return

    metrics = compute_metrics(results)
    cases = find_case_studies(results)
    correlations = find_system_correlations(results)

    print("\n" + "=" * 70)
    print(f"  MARIP BACKTEST RESULTS — {backtest_data.get('season', '?')} Season")
    print("=" * 70)

    print(f"\n  Races evaluated:  {backtest_data.get('races_evaluated', '?')}")
    print(f"  Total predictions: {metrics['total_predictions']}")
    print(f"  Accuracy:          {metrics['accuracy']}%")
    print(f"  Precision:         {metrics['precision']}%")
    print(f"  Recall:            {metrics['recall']}%")
    print(f"  F1 Score:          {metrics['f1_score']}%")

    cm = metrics.get("confusion_matrix", {})
    print(f"\n  Confusion Matrix:")
    print(f"    True Positive:  {cm.get('true_positive', 0)} (flagged risk, bad outcome)")
    print(f"    True Negative:  {cm.get('true_negative', 0)} (no flag, normal outcome)")
    print(f"    False Positive: {cm.get('false_positive', 0)} (flagged risk, normal outcome)")
    print(f"    False Negative: {cm.get('false_negative', 0)} (missed, bad outcome)")

    print(f"\n  Outcome Distribution:")
    for outcome, count in sorted(metrics.get("outcome_distribution", {}).items()):
        print(f"    {outcome:30s} {count}")

    # Multi-model metrics
    if metrics.get("strategy_match_rate") is not None:
        print(f"\n  Strategy Model:")
        print(f"    Strategy match rate:  {metrics['strategy_match_rate']}%")
        print(f"    Avg time delta:      {metrics['avg_strategy_delta_s']:+.1f}s")
    if metrics.get("avg_composite_risk") is not None:
        print(f"\n  Composite Risk:")
        print(f"    Average risk score:  {metrics['avg_composite_risk']}")
        for level, count in sorted(metrics.get("composite_risk_distribution", {}).items()):
            print(f"    {level:20s} {count}")
    if metrics.get("elt_coverage"):
        print(f"\n  ELT Coverage:          {metrics['elt_coverage']}%")
    if metrics.get("cliff_warnings_total", 0) > 0:
        print(f"  Tyre cliff warnings:   {metrics['cliff_warnings_total']}")

    # Lap prediction models
    if metrics.get("xgb_laps_avg_mae") is not None or metrics.get("bilstm_laps_avg_mae") is not None:
        print(f"\n  Lap Prediction Models:")
        if metrics.get("xgb_laps_avg_mae") is not None:
            print(f"    XGBoost  — Avg MAE: {metrics['xgb_laps_avg_mae']:.3f}s  Avg R²: {metrics.get('xgb_laps_avg_r2', 0):.4f}")
        if metrics.get("bilstm_laps_avg_mae") is not None:
            print(f"    BiLSTM   — Avg MAE: {metrics['bilstm_laps_avg_mae']:.3f}s  Avg R²: {metrics.get('bilstm_laps_avg_r2', 0):.4f}")

    if correlations:
        print(f"\n  System Prediction Value:")
        for sys, data in sorted(correlations.items(), key=lambda x: x[1]["flag_precision"], reverse=True):
            print(f"    {sys:20s} Flag precision: {data['flag_precision']:5.1f}% ({data['flagged_count']} flags)")

    if metrics.get("team_accuracy"):
        print(f"\n  Per-Team Accuracy:")
        for team, acc in sorted(metrics["team_accuracy"].items(), key=lambda x: x[1], reverse=True):
            print(f"    {team:25s} {acc}%")

    if cases:
        print(f"\n  Top Case Study Candidates:")
        for c in cases[:10]:
            print(f"\n    R{c['round']:2d} {c['race_name']}")
            print(f"    {c['driver_code']} ({c['constructor_name']}) — Score: {c['score']}")
            print(f"    Predicted: Health={c['predicted_health']}% Risk={'YES' if c['predicted_risk'] else 'no'}")
            if c['flagged_systems']:
                print(f"    Flagged: {', '.join(c['flagged_systems'])}")
            if c['degrading_systems']:
                print(f"    Degrading: {', '.join(c['degrading_systems'])}")
            print(f"    Actual: Grid={c['actual_grid']} Pos={c['actual_position']} Status={c['actual_status']}")
            for reason in c['reasons']:
                print(f"      → {reason}")

    print("\n" + "=" * 70)
