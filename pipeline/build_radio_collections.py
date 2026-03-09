"""
Build Derived Radio Intelligence Collections
─────────────────────────────────────────────
Aggregates tagged team_radio_transcripts into 5 derived collections:

  1. radio_signal_index        — every signal, searchable by session/driver/type
  2. driver_communication_profiles — per-driver, per-season aggregates
  3. car_balance_timeline      — per-circuit balance feedback across seasons
  4. reliability_precursors    — radio messages preceding mechanical issues
  5. strategy_comms_log        — pit/compound discussions per race

Usage:
    python pipeline/build_radio_collections.py
    python pipeline/build_radio_collections.py --year 2024
    python pipeline/build_radio_collections.py --collection radio_signal_index
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from updater._db import get_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


def _to_python(obj):
    """Recursively convert numpy types to native Python for MongoDB."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


# ── 1. radio_signal_index ────────────────────────────────────────────────


def build_radio_signal_index(db, years: list[int]) -> int:
    """Flatten tagged transcripts into one document per signal occurrence."""
    src = db["team_radio_transcripts"]
    dst = db["radio_signal_index"]

    query = {"signal_tags": {"$exists": True, "$ne": []}}
    if years:
        query["year"] = {"$in": years}

    docs = list(src.find(query))
    logger.info("radio_signal_index: %d transcripts with signals", len(docs))

    ops = []
    for doc in docs:
        for signal_type in doc["signal_tags"]:
            key = f"{doc['file_key']}::{signal_type}"
            record = {
                "year": doc["year"],
                "meeting": doc["meeting"],
                "session": doc["session"],
                "driver_name": doc.get("driver_name"),
                "driver_number": doc.get("driver_number"),
                "signal_type": signal_type,
                "severity": doc.get("urgency_level", 0),
                "transcript_snippet": doc["transcript"][:200],
                "sentiment_score": doc.get("sentiment_score", 0),
                "timestamp": doc.get("time"),
                "date": doc.get("date"),
                "file_key": doc["file_key"],
                "speaker": doc.get("speaker"),
                "mentioned_components": doc.get("mentioned_components", []),
                "mentioned_corners": doc.get("mentioned_corners", []),
                "tyre_compound_mentioned": doc.get("tyre_compound_mentioned"),
                "confidence": doc.get("confidence", 0),
            }
            ops.append(UpdateOne({"_key": key}, {"$set": _to_python(record)}, upsert=True))

    if ops:
        dst.bulk_write(ops, ordered=False)
        dst.create_index([("year", 1), ("meeting", 1), ("session", 1)])
        dst.create_index([("driver_number", 1), ("signal_type", 1)])
        dst.create_index([("signal_type", 1), ("year", 1)])

    logger.info("radio_signal_index: upserted %d signal records", len(ops))
    return len(ops)


# ── 2. driver_communication_profiles ─────────────────────────────────────


def build_driver_communication_profiles(db, years: list[int]) -> int:
    """Per-driver, per-season communication aggregates."""
    src = db["team_radio_transcripts"]
    dst = db["driver_communication_profiles"]

    query = {"signal_tags": {"$exists": True}}
    if years:
        query["year"] = {"$in": years}

    docs = list(src.find(query))
    logger.info("driver_communication_profiles: %d tagged transcripts", len(docs))

    # Group by (driver, season)
    groups = defaultdict(list)
    for doc in docs:
        driver = doc.get("driver_name", "Unknown")
        driver_num = doc.get("driver_number")
        season = doc.get("year")
        if driver and season:
            groups[(driver, driver_num, season)].append(doc)

    ops = []
    for (driver, driver_num, season), driver_docs in groups.items():
        total = len(driver_docs)

        # Sentiment distribution
        neg = sum(1 for d in driver_docs if d.get("sentiment_score", 0) < -0.3)
        pos = sum(1 for d in driver_docs if d.get("sentiment_score", 0) > 0.3)
        neu = total - neg - pos

        # Signal frequency
        tag_counter = Counter()
        for d in driver_docs:
            for t in d.get("signal_tags", []):
                tag_counter[t] += 1

        # Session type distribution
        sessions = Counter(d.get("session", "Unknown") for d in driver_docs)
        session_count = dict(sessions)

        # Frustration rate by session type (negative sentiment)
        frustration = defaultdict(lambda: {"neg": 0, "total": 0})
        for d in driver_docs:
            s = d.get("session", "Unknown")
            frustration[s]["total"] += 1
            if d.get("sentiment_score", 0) < -0.3:
                frustration[s]["neg"] += 1
        frustration_rates = {
            s: round(v["neg"] / v["total"], 2) if v["total"] > 0 else 0
            for s, v in frustration.items()
        }

        # Messages per session average
        unique_sessions = len(set((d.get("meeting"), d.get("session")) for d in driver_docs))
        msgs_per_session = round(total / max(unique_sessions, 1), 1)

        # Complaint frequency (negative sentiment or urgency >= 2)
        complaints = sum(1 for d in driver_docs
                         if d.get("sentiment_score", 0) < -0.3 or d.get("urgency_level", 0) >= 2)

        # Driver code from name
        driver_code = driver[:3].upper()

        profile = {
            "driver_code": driver_code,
            "driver_name": driver,
            "driver_number": driver_num,
            "season": season,
            "total_messages": total,
            "messages_per_session_avg": msgs_per_session,
            "sentiment_distribution": {
                "negative": round(neg / total, 2),
                "neutral": round(neu / total, 2),
                "positive": round(pos / total, 2),
            },
            "top_signal_types": [t for t, _ in tag_counter.most_common(5)],
            "signal_counts": dict(tag_counter),
            "complaint_frequency": round(complaints / total, 2),
            "frustration_rate_by_session_type": frustration_rates,
            "messages_by_session_type": session_count,
            "updated_at": datetime.now(timezone.utc),
        }

        ops.append(UpdateOne(
            {"driver_name": driver, "season": season},
            {"$set": _to_python(profile)},
            upsert=True,
        ))

    if ops:
        dst.bulk_write(ops, ordered=False)
        dst.create_index([("driver_code", 1), ("season", 1)], unique=True)

    logger.info("driver_communication_profiles: upserted %d profiles", len(ops))
    return len(ops)


# ── 3. car_balance_timeline ──────────────────────────────────────────────


def build_car_balance_timeline(db, years: list[int]) -> int:
    """Per-circuit balance feedback aggregated across sessions."""
    src = db["team_radio_transcripts"]
    dst = db["car_balance_timeline"]

    query = {"signal_tags": "car_balance"}
    if years:
        query["year"] = {"$in": years}

    docs = list(src.find(query))
    logger.info("car_balance_timeline: %d car_balance messages", len(docs))

    # Group by (circuit, season)
    groups = defaultdict(list)
    for doc in docs:
        groups[(doc["meeting"], doc["year"])].append(doc)

    # Balance keywords to detect
    balance_keywords = {
        "understeer": ["understeer", "pushing", "front sliding", "won't turn", "front locking"],
        "oversteer": ["oversteer", "rear snapping", "rear sliding", "rear stepping", "snap"],
        "rear_instability": ["rear instability", "rear end", "twitchy", "nervous rear"],
        "front_locking": ["front locking", "locking up", "lock up"],
        "entry_issue": ["entry", "turn in", "initial turn"],
        "exit_issue": ["exit", "traction", "getting on the power"],
    }

    ops = []
    for (circuit, season), circuit_docs in groups.items():
        reports = []
        issue_counter = Counter()

        for d in circuit_docs:
            text_lower = d["transcript"].lower()

            # Detect balance issue type
            detected_issue = "general_balance"
            for issue, keywords in balance_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    detected_issue = issue
                    break

            issue_counter[detected_issue] += 1

            reports.append({
                "session": d["session"],
                "driver": d.get("driver_name"),
                "issue": detected_issue,
                "corners": d.get("mentioned_corners", []),
                "severity": d.get("urgency_level", 0),
                "snippet": d["transcript"][:120],
            })

        dominant = issue_counter.most_common(1)[0][0] if issue_counter else "unknown"

        record = {
            "circuit": circuit,
            "season": season,
            "balance_reports": _to_python(reports),
            "report_count": len(reports),
            "dominant_issue": dominant,
            "issue_distribution": dict(issue_counter),
            "updated_at": datetime.now(timezone.utc),
        }

        ops.append(UpdateOne(
            {"circuit": circuit, "season": season},
            {"$set": _to_python(record)},
            upsert=True,
        ))

    if ops:
        dst.bulk_write(ops, ordered=False)
        dst.create_index([("circuit", 1), ("season", 1)], unique=True)

    logger.info("car_balance_timeline: upserted %d circuit-season records", len(ops))
    return len(ops)


# ── 4. reliability_precursors ────────────────────────────────────────────


def build_reliability_precursors(db, years: list[int]) -> int:
    """Radio messages flagged as reliability concerns — ground truth for anomaly detection."""
    src = db["team_radio_transcripts"]
    dst = db["reliability_precursors"]

    query = {"signal_tags": "reliability_flag"}
    if years:
        query["year"] = {"$in": years}

    docs = list(src.find(query))
    logger.info("reliability_precursors: %d reliability_flag messages", len(docs))

    # Group by (year, meeting, driver)
    groups = defaultdict(list)
    for doc in docs:
        groups[(doc["year"], doc["meeting"], doc.get("driver_name", "Unknown"))].append(doc)

    ops = []
    for (year, meeting, driver), flag_docs in groups.items():
        precursor_messages = []
        for d in sorted(flag_docs, key=lambda x: x.get("time", "")):
            precursor_messages.append({
                "time": d.get("time"),
                "session": d["session"],
                "transcript": d["transcript"][:200],
                "signal": "reliability_flag",
                "urgency": d.get("urgency_level", 0),
                "mentioned_components": d.get("mentioned_components", []),
                "confidence": d.get("confidence", 0),
            })

        # Component frequency across all flags for this event
        comp_counter = Counter()
        for d in flag_docs:
            for c in d.get("mentioned_components", []):
                comp_counter[c] += 1

        record = {
            "year": year,
            "meeting": meeting,
            "driver": driver,
            "precursor_messages": _to_python(precursor_messages),
            "message_count": len(precursor_messages),
            "max_urgency": max(d.get("urgency_level", 0) for d in flag_docs),
            "flagged_components": dict(comp_counter),
            "sessions_affected": list(set(d["session"] for d in flag_docs)),
            "updated_at": datetime.now(timezone.utc),
        }

        ops.append(UpdateOne(
            {"year": year, "meeting": meeting, "driver": driver},
            {"$set": _to_python(record)},
            upsert=True,
        ))

    if ops:
        dst.bulk_write(ops, ordered=False)
        dst.create_index([("year", 1), ("meeting", 1)])
        dst.create_index([("driver", 1), ("year", 1)])
        dst.create_index("max_urgency")

    logger.info("reliability_precursors: upserted %d event records", len(ops))
    return len(ops)


# ── 5. strategy_comms_log ────────────────────────────────────────────────


def build_strategy_comms_log(db, years: list[int]) -> int:
    """Pit window discussions and strategy communications per race."""
    src = db["team_radio_transcripts"]
    dst = db["strategy_comms_log"]

    query = {"signal_tags": "strategy_call"}
    if years:
        query["year"] = {"$in": years}

    docs = list(src.find(query))
    logger.info("strategy_comms_log: %d strategy_call messages", len(docs))

    # Group by (year, meeting, session)
    groups = defaultdict(list)
    for doc in docs:
        groups[(doc["year"], doc["meeting"], doc["session"])].append(doc)

    # Strategy event keywords
    pit_keywords = ["box box", "box this lap", "pit", "pit now", "in this lap", "stay out"]
    compound_keywords = ["soft", "medium", "hard", "intermediate", "inter", "wet"]

    ops = []
    for (year, meeting, session), strat_docs in groups.items():
        events = []
        compounds_mentioned = Counter()

        for d in sorted(strat_docs, key=lambda x: x.get("time", "")):
            text_lower = d["transcript"].lower()

            # Classify event type
            event_type = "strategy_discussion"
            if any(kw in text_lower for kw in pit_keywords):
                event_type = "pit_call"
            elif any(kw in text_lower for kw in compound_keywords):
                event_type = "compound_discussion"

            # Track compound mentions
            if d.get("tyre_compound_mentioned"):
                compounds_mentioned[d["tyre_compound_mentioned"]] += 1

            events.append({
                "type": event_type,
                "speaker": d.get("speaker", "unknown"),
                "driver": d.get("driver_name"),
                "time": d.get("time"),
                "snippet": d["transcript"][:150],
                "urgency": d.get("urgency_level", 0),
                "tyre_compound": d.get("tyre_compound_mentioned"),
            })

        record = {
            "year": year,
            "meeting": meeting,
            "session": session,
            "strategy_events": _to_python(events),
            "event_count": len(events),
            "pit_calls": sum(1 for e in events if e["type"] == "pit_call"),
            "compound_discussions": sum(1 for e in events if e["type"] == "compound_discussion"),
            "compounds_mentioned": dict(compounds_mentioned),
            "drivers_involved": list(set(d.get("driver_name") for d in strat_docs if d.get("driver_name"))),
            "updated_at": datetime.now(timezone.utc),
        }

        ops.append(UpdateOne(
            {"year": year, "meeting": meeting, "session": session},
            {"$set": _to_python(record)},
            upsert=True,
        ))

    if ops:
        dst.bulk_write(ops, ordered=False)
        dst.create_index([("year", 1), ("meeting", 1), ("session", 1)], unique=True)
        dst.create_index([("year", 1)])

    logger.info("strategy_comms_log: upserted %d session records", len(ops))
    return len(ops)


# ── Runner ───────────────────────────────────────────────────────────────


BUILDERS = {
    "radio_signal_index": build_radio_signal_index,
    "driver_communication_profiles": build_driver_communication_profiles,
    "car_balance_timeline": build_car_balance_timeline,
    "reliability_precursors": build_reliability_precursors,
    "strategy_comms_log": build_strategy_comms_log,
}


def build_all(years: list[int], collections: list[str] | None = None):
    db = get_db()
    targets = collections or list(BUILDERS.keys())

    results = {}
    for name in targets:
        if name not in BUILDERS:
            logger.warning("Unknown collection: %s", name)
            continue
        logger.info("── Building %s ──", name)
        count = BUILDERS[name](db, years)
        results[name] = count

    # Log to pipeline_log
    db["pipeline_log"].insert_one({
        "pipeline": "build_radio_collections",
        "years": years,
        "collections_built": results,
        "timestamp": datetime.now(timezone.utc),
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Build derived radio intelligence collections")
    parser.add_argument("--year", type=int, help="Single year")
    parser.add_argument("--from-year", type=int, help="Start year")
    parser.add_argument("--to-year", type=int, help="End year (inclusive)")
    parser.add_argument("--collection", type=str, help="Build only this collection",
                        choices=list(BUILDERS.keys()))
    args = parser.parse_args()

    if args.year:
        years = [args.year]
    elif args.from_year or args.to_year:
        years = list(range(args.from_year or 2023, (args.to_year or 2024) + 1))
    else:
        years = [2023, 2024]

    collections = [args.collection] if args.collection else None
    logger.info("Building radio collections — years=%s, collections=%s", years, collections or "ALL")
    results = build_all(years, collections)

    logger.info("========================================================")
    for name, count in results.items():
        logger.info("  %s: %d records", name, count)
    logger.info("DONE")


if __name__ == "__main__":
    main()
