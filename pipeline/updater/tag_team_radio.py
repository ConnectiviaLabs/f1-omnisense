"""Extract structured signals from team radio transcripts using Groq LLM.

Reads transcripts from `team_radio_transcripts`, runs each through an LLM
to extract signal tags, sentiment, urgency, and metadata. Upserts enrichment
fields back onto the same documents.

Idempotent: skips documents that already have `signal_tags`.

Usage:
    python3 -m pipeline.updater.tag_team_radio                     # all transcripts
    python3 -m pipeline.updater.tag_team_radio --year 2024         # single year
    python3 -m pipeline.updater.tag_team_radio --dry-run           # count only
    python3 -m pipeline.updater.tag_team_radio --limit 10          # first N
    python3 -m pipeline.updater.tag_team_radio --retag             # re-process already tagged
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

REQUEST_DELAY = 3.1  # Groq free tier: ~20 req/min

SIGNAL_CATEGORIES = [
    "tyre_condition",
    "car_balance",
    "reliability_flag",
    "strategy_call",
    "track_condition",
    "emotional_state",
    "competitor_intel",
]

SYSTEM_PROMPT = """You are an F1 team radio analyst. Given a transcript from McLaren team radio, extract structured signals.

Return a JSON object with these fields:

- "signal_tags": array of applicable categories from: tyre_condition, car_balance, reliability_flag, strategy_call, track_condition, emotional_state, competitor_intel. Can be empty [] if the message is purely procedural (e.g. "copy", radio checks).
- "sentiment_score": float from -1.0 (very negative/frustrated) to 1.0 (very positive/celebratory). 0.0 = neutral.
- "urgency_level": integer 0-3. 0 = routine info, 1 = notable, 2 = important/actionable, 3 = critical/safety.
- "speaker": "driver" or "engineer" — infer from context. Engineers give instructions/data, drivers report feelings/conditions.
- "mentioned_components": array of car parts mentioned. Use standardized terms: front_wing, rear_wing, front_tyre, rear_tyre, brakes, engine, gearbox, suspension, floor, diffuser, drs, steering, battery, ers. Empty [] if none.
- "mentioned_corners": array of corners/sectors mentioned, e.g. ["T3", "T9"], ["sector_1"]. Empty [] if none.
- "tyre_compound_mentioned": "soft", "medium", "hard", "intermediate", "wet", or null if none mentioned.
- "confidence": float 0.0-1.0 indicating your confidence in the signal extraction.

Rules:
- A single message can have multiple signal_tags (e.g. tyre complaint + emotional frustration).
- "box box", pit calls, compound discussions, stint plans = strategy_call.
- Complaints about understeer, oversteer, balance, grip = car_balance.
- "graining", "blistering", "deg", tyre wear comments = tyre_condition.
- Vibrations, power issues, mechanical concerns = reliability_flag.
- Weather, track grip, dirty air = track_condition.
- Frustration, celebration, anger, confidence = emotional_state.
- Mentions of other drivers/teams, pace gaps = competitor_intel.
- Radio checks, "copy", acknowledgements with no substance = empty signal_tags, urgency 0.

Return ONLY the JSON object, no other text."""

USER_PROMPT_TEMPLATE = """Driver: {driver_name}
Session: {session} — {meeting} {year}
Transcript: "{transcript}"
"""


def _extract_signals(client, transcript: str, driver_name: str,
                     session: str, meeting: str, year: int) -> dict | None:
    """Call Groq LLM to extract structured signals from a transcript."""
    user_msg = USER_PROMPT_TEMPLATE.format(
        driver_name=driver_name,
        session=session,
        meeting=meeting,
        year=year,
        transcript=transcript,
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)

        # Validate and sanitize
        result = {
            "signal_tags": [t for t in data.get("signal_tags", []) if t in SIGNAL_CATEGORIES],
            "sentiment_score": max(-1.0, min(1.0, float(data.get("sentiment_score", 0)))),
            "urgency_level": max(0, min(3, int(data.get("urgency_level", 0)))),
            "speaker": data.get("speaker", "unknown") if data.get("speaker") in ("driver", "engineer") else "unknown",
            "mentioned_components": data.get("mentioned_components", []),
            "mentioned_corners": data.get("mentioned_corners", []),
            "tyre_compound_mentioned": data.get("tyre_compound_mentioned"),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
        }
        return result

    except json.JSONDecodeError as e:
        logger.error("  JSON parse error: %s — raw: %s", e, raw[:200])
        return None
    except Exception as e:
        logger.error("  LLM call failed: %s", e)
        return None


def tag_all(
    years: list[int],
    dry_run: bool = False,
    limit: int = 0,
    retag: bool = False,
) -> dict:
    """Tag all transcripts with LLM-extracted signals."""
    from groq import Groq

    from ._db import get_db

    db = get_db()
    col = db["team_radio_transcripts"]

    # Build query
    query: dict = {}
    if years:
        query["year"] = {"$in": years}
    if not retag:
        query["signal_tags"] = {"$exists": False}

    total = col.count_documents(query)
    logger.info("Found %d transcripts to tag (years=%s, retag=%s)", total, years, retag)

    stats = {"total": total, "tagged": 0, "skipped": 0, "errors": 0}

    if dry_run:
        logger.info("DRY RUN — would tag %d transcripts", total)
        return stats

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    cursor = col.find(query, {"_id": 1, "file_key": 1, "transcript": 1,
                               "driver_name": 1, "session": 1, "meeting": 1, "year": 1})
    if limit:
        cursor = cursor.limit(limit)
        stats["total"] = min(total, limit)

    for i, doc in enumerate(cursor, 1):
        transcript = doc.get("transcript", "")
        if not transcript or len(transcript.strip()) < 3:
            stats["skipped"] += 1
            continue

        file_key = doc.get("file_key", "?")
        logger.info("  [%d/%d] %s", i, stats["total"], file_key)

        time.sleep(REQUEST_DELAY)
        signals = _extract_signals(
            client,
            transcript,
            doc.get("driver_name", "Unknown"),
            doc.get("session", "Unknown"),
            doc.get("meeting", "Unknown"),
            doc.get("year", 0),
        )

        if signals is None:
            stats["errors"] += 1
            continue

        col.update_one({"_id": doc["_id"]}, {"$set": signals})
        stats["tagged"] += 1

        tags = ", ".join(signals["signal_tags"]) if signals["signal_tags"] else "(none)"
        logger.info("    → tags=[%s] sentiment=%.1f urgency=%d",
                     tags, signals["sentiment_score"], signals["urgency_level"])

    return stats


def main():
    parser = argparse.ArgumentParser(description="Tag team radio transcripts with LLM-extracted signals")
    parser.add_argument("--year", type=int, help="Single year")
    parser.add_argument("--from-year", type=int, help="Start year")
    parser.add_argument("--to-year", type=int, help="End year (inclusive)")
    parser.add_argument("--dry-run", action="store_true", help="Count transcripts without tagging")
    parser.add_argument("--limit", type=int, default=0, help="Max transcripts to process (0=unlimited)")
    parser.add_argument("--retag", action="store_true", help="Re-process already tagged transcripts")
    args = parser.parse_args()

    if args.year:
        years = [args.year]
    elif args.from_year or args.to_year:
        years = list(range(args.from_year or 2023, (args.to_year or 2024) + 1))
    else:
        years = [2023, 2024]

    logger.info("Tagging team radio — years: %s", years)
    stats = tag_all(years, dry_run=args.dry_run, limit=args.limit, retag=args.retag)

    logger.info("========================================================")
    logger.info(
        "DONE: %d total, %d tagged, %d skipped, %d errors",
        stats["total"], stats["tagged"], stats["skipped"], stats["errors"],
    )


if __name__ == "__main__":
    main()
