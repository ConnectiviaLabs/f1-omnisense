"""Shared data-gathering functions for Trident convergence reports.

Extracted from advantage_router.py so they can be reused by both the
manual Trident endpoint and the TridentInsightAgent (omniagent #7).
"""

from __future__ import annotations

import numpy as np


def gather_structured_context(db, scope: str, entity: str | None) -> str:
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
                f"one_stop_freq={ps.get('one_stop_freq', '?')}, "
                f"avg_first_stop_lap={ps.get('avg_first_stop_lap', '?')}, "
                f"pit_stops={pe.get('total_stops', '?')}, "
                f"avg_pit_duration={pe.get('avg_duration_s', '?')}s, "
                f"best_pit={pe.get('best_duration_s', '?')}s"
            )
            comps = sp.get("compound_profiles") or []
            if comps:
                parts.append("  Compound profiles:")
                for c in comps:
                    parts.append(
                        f"    {c.get('compound', '?')}: laps={c.get('total_laps', '?')}, "
                        f"avg_lap={c.get('avg_lap_time_s', '?')}s, "
                        f"tyre_life={c.get('avg_tyre_life', '?')} laps"
                    )

        # Compare vs grid (top 5 rivals)
        rivals = list(db["victory_strategy_profiles"].find(
            {"driver_code": {"$ne": entity.upper()}},
            {"_id": 0, "embedding": 0, "narrative": 0, "compound_profiles": 0}
        ).limit(5))
        if rivals:
            parts.append(f"RIVAL STRATEGY COMPARISON (vs {entity.upper()}):")
            for r in rivals:
                rps = r.get("pit_strategy") or {}
                rpe = r.get("pit_execution") or {}
                parts.append(
                    f"  {r.get('driver_code', '?')}/{r.get('team', '?')}: "
                    f"undercut={rps.get('undercut_aggression', '?')}, "
                    f"one_stop={rps.get('one_stop_freq', '?')}, "
                    f"avg_pit={rpe.get('avg_duration_s', '?')}s"
                )

    return "\n\n".join(parts) if parts else ""


def gather_kex_data(db, scope: str, entity: str | None) -> str:
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


def gather_anomaly_data(db, scope: str, entity: str | None) -> str:
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


def gather_forecast_data(db, scope: str, entity: str | None) -> str:
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
