"""
Backtest Evaluation — analyze prediction accuracy and find case studies.

Computes:
  - Overall accuracy (correct predictions / total)
  - Precision & recall for risk flagging
  - Per-system prediction value
  - Case study candidates where MARIP foreshadowed outcomes
  - Permutation-based statistical significance testing (Masters p.10)
"""

import logging
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# Outcomes that indicate a problem — outperformance is NOT a bad outcome
BAD_OUTCOMES = {
    "dnf_mechanical", "dnf_other", "lapped",
    "major_underperformance", "underperformance",
}


def _is_bad(outcome: str) -> bool:
    return outcome in BAD_OUTCOMES


# ── Permutation significance testing (Masters Ch. 1, p.10) ──────────────────


def permutation_test(
    results: list[dict],
    n_permutations: int = 1000,
    seed: int = 42,
    metrics: tuple[str, ...] = ("accuracy", "f1", "precision", "recall"),
) -> dict:
    """Test whether model predictions are statistically better than chance.

    Implements Masters' permutation methodology:
      1. Compute real metric on actual outcome labels
      2. Shuffle outcome labels N times, recompute metric each time
      3. p-value = fraction of shuffled scores >= real score

    If p < 0.05, the model's performance is statistically significant —
    it's doing something real, not just getting lucky with base rates.

    Does NOT re-run models. Only shuffles the mapping between predictions
    and outcomes, which is the correct null hypothesis: "predictions are
    independent of outcomes."
    """
    if not results:
        return {}

    rng = np.random.default_rng(seed)

    # Extract aligned arrays: predicted_risk (bool) and actual bad outcome (bool)
    predicted_risk = np.array([bool(r.get("predicted_risk")) for r in results])
    actual_outcomes = [r.get("actual_outcome", "") for r in results]
    actual_bad = np.array([_is_bad(o) for o in actual_outcomes])

    # Also extract prediction_correct for accuracy permutation
    prediction_correct = np.array([bool(r.get("prediction_correct")) for r in results])

    n = len(results)

    def _compute_from_arrays(pred_risk, act_bad, pred_correct):
        """Compute metrics from boolean arrays."""
        tp = int(np.sum(pred_risk & act_bad))
        fp = int(np.sum(pred_risk & ~act_bad))
        fn = int(np.sum(~pred_risk & act_bad))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = float(np.mean(pred_correct))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # 1. Real scores
    real_scores = _compute_from_arrays(predicted_risk, actual_bad, prediction_correct)

    # 2. Permutation loop — shuffle outcome labels
    null_distributions = {m: [] for m in metrics}

    for _ in range(n_permutations):
        # Shuffle which outcomes map to which driver-race entries
        perm_idx = rng.permutation(n)
        shuffled_bad = actual_bad[perm_idx]
        shuffled_outcomes = [actual_outcomes[i] for i in perm_idx]

        # Recompute prediction_correct under shuffled labels
        shuffled_correct = np.array([
            (bool(results[j].get("predicted_risk")) == _is_bad(shuffled_outcomes[j]))
            for j in range(n)
        ])

        perm_scores = _compute_from_arrays(predicted_risk, shuffled_bad, shuffled_correct)

        for m in metrics:
            null_distributions[m].append(perm_scores[m])

    # 3. Compute p-values
    significance = {}
    for m in metrics:
        null = np.array(null_distributions[m])
        real_val = real_scores[m]
        # p-value: fraction of permuted scores >= real score (one-tailed)
        p_value = float(np.mean(null >= real_val))
        significance[m] = {
            "real_score": round(real_val * 100, 2),
            "null_mean": round(float(np.mean(null)) * 100, 2),
            "null_std": round(float(np.std(null)) * 100, 2),
            "null_p95": round(float(np.percentile(null, 95)) * 100, 2),
            "null_p99": round(float(np.percentile(null, 99)) * 100, 2),
            "p_value": round(p_value, 4),
            "significant_05": p_value < 0.05,
            "significant_01": p_value < 0.01,
        }

    return {
        "n_permutations": n_permutations,
        "n_samples": n,
        "base_rate_bad": round(float(np.mean(actual_bad)) * 100, 2),
        "metrics": significance,
        "interpretation": _interpret_significance(significance),
    }


def _interpret_significance(significance: dict) -> str:
    """Human-readable interpretation of permutation test results."""
    sig_metrics = [m for m, s in significance.items() if s["significant_05"]]
    nonsig_metrics = [m for m, s in significance.items() if not s["significant_05"]]

    if not sig_metrics:
        return (
            "No metrics achieved statistical significance (p < 0.05). "
            "Model predictions are NOT demonstrably better than random chance. "
            "The model needs improvement before it can be trusted for real decisions."
        )
    elif len(sig_metrics) == len(significance):
        strongest = min(significance.items(), key=lambda x: x[1]["p_value"])
        return (
            f"All metrics are statistically significant (p < 0.05). "
            f"Strongest signal: {strongest[0]} (p={strongest[1]['p_value']:.4f}). "
            f"The model's predictions are reliably better than chance."
        )
    else:
        return (
            f"Mixed results: {', '.join(sig_metrics)} significant (p < 0.05), "
            f"but {', '.join(nonsig_metrics)} are NOT significant. "
            f"The model shows some predictive power but isn't reliable across all metrics."
        )


# ── Threshold calibration from backtest outcomes (Improvement #2) ────────────


def calibrate_thresholds(
    results: list[dict],
    score_key: str = "composite_risk",
    n_steps: int = 200,
) -> dict:
    """Find optimal severity thresholds by sweeping scores against actual outcomes.

    For each candidate threshold, computes F1 score (risk if score >= threshold).
    Returns optimal binary threshold + multi-level severity thresholds that
    maximize predictive value.

    This replaces hand-tuned thresholds with data-driven ones calibrated
    against real 2024 race outcomes.
    """
    # Extract score-outcome pairs
    pairs = []
    for r in results:
        score = r.get(score_key)
        outcome = r.get("actual_outcome", "")
        if score is not None:
            pairs.append((float(score), _is_bad(outcome)))

    if len(pairs) < 10:
        return {"status": "insufficient_data", "n_samples": len(pairs)}

    scores = np.array([p[0] for p in pairs])
    labels = np.array([p[1] for p in pairs])

    score_min, score_max = float(scores.min()), float(scores.max())
    candidates = np.linspace(score_min, score_max, n_steps)

    # ── 1. Binary threshold sweep (maximize F1) ──
    best_f1 = -1
    best_threshold = score_min
    sweep_results = []

    for t in candidates:
        predicted = scores >= t
        tp = int(np.sum(predicted & labels))
        fp = int(np.sum(predicted & ~labels))
        fn = int(np.sum(~predicted & labels))
        tn = int(np.sum(~predicted & ~labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        sweep_results.append({
            "threshold": round(float(t), 2),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    # ── 2. Multi-level severity thresholds ──
    # Find thresholds where bad-outcome rate changes significantly
    # Sort by score, compute rolling bad-outcome rate
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]

    # Compute bad-outcome rate above each threshold
    n_total = len(scores)
    bad_rates = []
    for i, t in enumerate(candidates):
        above = sorted_labels[sorted_scores >= t]
        rate = float(np.mean(above)) if len(above) > 0 else 0
        bad_rates.append(rate)

    bad_rates = np.array(bad_rates)

    # Find severity levels using bad-outcome probability breakpoints:
    # CRITICAL: >60% of entries above this score had bad outcomes
    # HIGH: >40% bad outcome rate
    # MEDIUM: >20% bad outcome rate
    # LOW: >10% bad outcome rate (everything else = NORMAL)
    severity_thresholds = {}
    for level, target_rate in [("low", 0.10), ("medium", 0.20), ("high", 0.40), ("critical", 0.60)]:
        matching = candidates[bad_rates >= target_rate]
        if len(matching) > 0:
            severity_thresholds[level] = round(float(matching[0]), 2)
        else:
            severity_thresholds[level] = None

    # ── 3. Per-system calibration ──
    system_calibration = _calibrate_per_system(results)

    # Best operating point
    best_entry = next(
        (s for s in sweep_results if abs(s["threshold"] - best_threshold) < 0.01),
        sweep_results[0] if sweep_results else {},
    )

    return {
        "status": "ok",
        "score_key": score_key,
        "n_samples": len(pairs),
        "base_rate_bad": round(float(np.mean(labels)) * 100, 2),
        # Binary threshold
        "optimal_threshold": round(best_threshold, 2),
        "optimal_f1": round(best_f1, 4),
        "optimal_precision": best_entry.get("precision", 0),
        "optimal_recall": best_entry.get("recall", 0),
        "confusion_at_optimal": {
            "tp": best_entry.get("tp", 0),
            "fp": best_entry.get("fp", 0),
            "fn": best_entry.get("fn", 0),
            "tn": best_entry.get("tn", 0),
        },
        # Multi-level severity
        "severity_thresholds": severity_thresholds,
        # Per-system
        "system_calibration": system_calibration,
        # Full sweep for plotting
        "sweep_top10": sorted(sweep_results, key=lambda x: x["f1"], reverse=True)[:10],
    }


def _calibrate_per_system(results: list[dict]) -> dict:
    """Calibrate per-system anomaly thresholds from backtest outcomes.

    Uses the `predicted_systems` dict in each result (system → {health, level, score_mean})
    to find which system scores best discriminate bad outcomes.
    """
    system_data = defaultdict(lambda: {"scores": [], "labels": []})

    for r in results:
        bad = _is_bad(r.get("actual_outcome", ""))
        systems = r.get("predicted_systems", {})
        for sys_name, sys_info in systems.items():
            score = sys_info.get("score_mean")
            if score is not None:
                system_data[sys_name]["scores"].append(float(score))
                system_data[sys_name]["labels"].append(bad)

    calibration = {}
    for sys_name, data in system_data.items():
        if len(data["scores"]) < 5:
            continue

        scores = np.array(data["scores"])
        labels = np.array(data["labels"])

        # Sweep for optimal F1
        candidates = np.linspace(float(scores.min()), float(scores.max()), 100)
        best_f1 = -1
        best_t = float(scores.min())

        for t in candidates:
            predicted = scores >= t
            tp = int(np.sum(predicted & labels))
            fp = int(np.sum(predicted & ~labels))
            fn = int(np.sum(~predicted & labels))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        # Bad-outcome rate when system is flagged (score > mean + 1 std)
        flag_threshold = float(np.mean(scores) + np.std(scores))
        flagged = labels[scores >= flag_threshold]
        flag_precision = float(np.mean(flagged)) if len(flagged) > 0 else 0

        calibration[sys_name] = {
            "n_samples": len(scores),
            "optimal_threshold": round(best_t, 4),
            "optimal_f1": round(best_f1, 4),
            "mean_score_bad": round(float(np.mean(scores[labels])), 4) if labels.any() else None,
            "mean_score_good": round(float(np.mean(scores[~labels])), 4) if (~labels).any() else None,
            "flag_precision": round(flag_precision, 4),
        }

    return dict(calibration)


def print_calibration(cal: dict):
    """Print calibration results in human-readable format."""
    if cal.get("status") != "ok":
        print(f"  Calibration failed: {cal.get('status', 'unknown')}")
        return

    print(f"\n{'='*70}")
    print(f"  THRESHOLD CALIBRATION — {cal['score_key']}")
    print(f"{'='*70}")
    print(f"  Samples: {cal['n_samples']}  Base rate (bad outcomes): {cal['base_rate_bad']}%")

    print(f"\n  Optimal Binary Threshold: {cal['optimal_threshold']}")
    print(f"    F1: {cal['optimal_f1']:.4f}  Precision: {cal['optimal_precision']:.4f}  Recall: {cal['optimal_recall']:.4f}")
    cm = cal.get("confusion_at_optimal", {})
    print(f"    TP: {cm.get('tp',0)}  FP: {cm.get('fp',0)}  FN: {cm.get('fn',0)}  TN: {cm.get('tn',0)}")

    sev = cal.get("severity_thresholds", {})
    print(f"\n  Calibrated Severity Thresholds (bad-outcome probability):")
    for level in ["low", "medium", "high", "critical"]:
        t = sev.get(level)
        print(f"    {level:10s} >= {t}" if t is not None else f"    {level:10s} (no threshold found)")

    sys_cal = cal.get("system_calibration", {})
    if sys_cal:
        print(f"\n  Per-System Calibration:")
        for sys_name, sc in sorted(sys_cal.items(), key=lambda x: x[1].get("optimal_f1", 0), reverse=True):
            bad_mean = f"{sc['mean_score_bad']:.4f}" if sc.get("mean_score_bad") is not None else "N/A"
            good_mean = f"{sc['mean_score_good']:.4f}" if sc.get("mean_score_good") is not None else "N/A"
            print(
                f"    {sys_name:20s}  Threshold: {sc['optimal_threshold']:.4f}  "
                f"F1: {sc['optimal_f1']:.4f}  "
                f"Bad mean: {bad_mean}  Good mean: {good_mean}"
            )

    print(f"\n  Top 5 Sweep Points:")
    for s in cal.get("sweep_top10", [])[:5]:
        print(f"    T={s['threshold']:6.2f}  F1={s['f1']:.4f}  P={s['precision']:.4f}  R={s['recall']:.4f}  TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']}")

    print(f"{'='*70}")


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
        # Statistical significance (Masters permutation test)
        "significance": permutation_test(results),
        # Threshold calibration
        "calibration": calibrate_thresholds(results),
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

    # Statistical significance
    sig = metrics.get("significance", {})
    if sig:
        print(f"\n  Statistical Significance (Permutation Test, n={sig.get('n_permutations', '?')}):")
        print(f"    Base rate (bad outcomes): {sig.get('base_rate_bad', '?')}%")
        for m, s in sig.get("metrics", {}).items():
            star = "***" if s["significant_01"] else ("*" if s["significant_05"] else "")
            print(f"    {m:12s}  Real: {s['real_score']:6.2f}%  Null: {s['null_mean']:6.2f}% ± {s['null_std']:.2f}%  p={s['p_value']:.4f} {star}")
        print(f"    Interpretation: {sig.get('interpretation', '')}")

    # Threshold calibration
    cal = metrics.get("calibration", {})
    if cal.get("status") == "ok":
        print_calibration(cal)

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
