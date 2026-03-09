"""Transcribe McLaren team radio MP3s using Groq Whisper API → MongoDB.

Walks data/team_radio/{year}/{meeting}/{session}/*.mp3, transcribes each
via Groq whisper-large-v3, and upserts into the `team_radio_transcripts`
collection in MongoDB.

Usage:
    python3 -m pipeline.updater.transcribe_team_radio                    # all years
    python3 -m pipeline.updater.transcribe_team_radio --year 2024        # single year
    python3 -m pipeline.updater.transcribe_team_radio --dry-run          # count only
    python3 -m pipeline.updater.transcribe_team_radio --limit 10         # first N clips
"""

from __future__ import annotations

import argparse
import logging
import os
import re
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

DATA_DIR = PROJECT_ROOT / "data" / "team_radio"

# Groq free tier: ~20 req/min → 3.1s delay is safe
REQUEST_DELAY = 3.1

# Map driver numbers → names (same as download script)
MCLAREN_DRIVERS: dict[int, dict[int, str]] = {
    2018: {14: "Alonso", 2: "Vandoorne"},
    2019: {55: "Sainz", 4: "Norris"},
    2020: {55: "Sainz", 4: "Norris"},
    2021: {3: "Ricciardo", 4: "Norris"},
    2022: {3: "Ricciardo", 4: "Norris"},
    2023: {81: "Piastri", 4: "Norris"},
    2024: {81: "Piastri", 4: "Norris"},
}

# Filename pattern: LANNOR01_4_20240302_142714.mp3
_FILENAME_RE = re.compile(
    r"^(?P<prefix>[A-Z]+)(?P<seq>\d+)_(?P<number>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.mp3$"
)


def _parse_filename(filename: str, year: int) -> dict:
    """Extract metadata from MP3 filename."""
    m = _FILENAME_RE.match(filename)
    if not m:
        return {}
    driver_num = int(m.group("number"))
    drivers = MCLAREN_DRIVERS.get(year, {})
    date_str = m.group("date")
    time_str = m.group("time")
    return {
        "driver_prefix": m.group("prefix"),
        "sequence": int(m.group("seq")),
        "driver_number": driver_num,
        "driver_name": drivers.get(driver_num, f"#{driver_num}"),
        "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
        "time": f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}",
    }


def _transcribe_file(client, mp3_path: Path) -> str | None:
    """Transcribe a single MP3 via Groq Whisper. Returns text or None."""
    try:
        with open(mp3_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(mp3_path.name, f.read()),
                model="whisper-large-v3",
                language="en",
            )
        return result.text.strip() if result.text else None
    except Exception as e:
        logger.error("  Transcription failed for %s: %s", mp3_path.name, e)
        return None


def _collect_mp3s(years: list[int]) -> list[tuple[int, str, str, Path]]:
    """Collect all MP3 files as (year, meeting, session, path) tuples."""
    files = []
    for year in years:
        year_dir = DATA_DIR / str(year)
        if not year_dir.is_dir():
            continue
        for meeting_dir in sorted(year_dir.iterdir()):
            if not meeting_dir.is_dir():
                continue
            for session_dir in sorted(meeting_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                for mp3 in sorted(session_dir.glob("*.mp3")):
                    files.append((year, meeting_dir.name, session_dir.name, mp3))
    return files


def transcribe_all(
    years: list[int],
    dry_run: bool = False,
    limit: int = 0,
) -> dict:
    """Transcribe all MP3s and upsert into MongoDB."""
    from groq import Groq

    from ._db import get_db

    db = get_db()
    col = db["team_radio_transcripts"]

    # Ensure indexes
    col.create_index("file_key", unique=True)
    col.create_index([("year", 1), ("meeting", 1), ("session", 1)])
    col.create_index([("driver_number", 1), ("year", 1)])

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    all_files = _collect_mp3s(years)
    logger.info("Found %d MP3 files across years %s", len(all_files), years)

    if limit:
        all_files = all_files[:limit]

    stats = {"total": len(all_files), "transcribed": 0, "skipped": 0, "errors": 0}

    if dry_run:
        stats["skipped"] = len(all_files)
        logger.info("DRY RUN — would transcribe %d files", len(all_files))
        return stats

    for i, (year, meeting, session, mp3_path) in enumerate(all_files, 1):
        file_key = f"{year}/{meeting}/{session}/{mp3_path.name}"

        # Skip if already transcribed
        if col.find_one({"file_key": file_key}, {"_id": 1}):
            stats["skipped"] += 1
            continue

        logger.info("  [%d/%d] %s", i, len(all_files), file_key)

        meta = _parse_filename(mp3_path.name, year)
        time.sleep(REQUEST_DELAY)
        text = _transcribe_file(client, mp3_path)

        if text is None:
            stats["errors"] += 1
            continue

        doc = {
            "file_key": file_key,
            "year": year,
            "meeting": meeting,
            "session": session,
            "filename": mp3_path.name,
            "transcript": text,
            "driver_number": meta.get("driver_number"),
            "driver_name": meta.get("driver_name"),
            "date": meta.get("date"),
            "time": meta.get("time"),
            "driver_prefix": meta.get("driver_prefix"),
            "sequence": meta.get("sequence"),
        }

        col.update_one({"file_key": file_key}, {"$set": doc}, upsert=True)
        stats["transcribed"] += 1
        logger.info("    → \"%s\"", text[:80] + ("..." if len(text) > 80 else ""))

    return stats


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Transcribe McLaren team radio via Groq Whisper")
    parser.add_argument("--year", type=int, help="Single year")
    parser.add_argument("--from-year", type=int, help="Start year")
    parser.add_argument("--to-year", type=int, help="End year (inclusive)")
    parser.add_argument("--dry-run", action="store_true", help="Count files without transcribing")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0=unlimited)")
    args = parser.parse_args()

    if args.year:
        years = [args.year]
    elif args.from_year or args.to_year:
        years = list(range(args.from_year or 2018, (args.to_year or 2024) + 1))
    else:
        years = list(range(2018, 2025))

    logger.info("Transcribing team radio — years: %s", years)
    stats = transcribe_all(years, dry_run=args.dry_run, limit=args.limit)

    logger.info("========================================================")
    logger.info(
        "DONE: %d total, %d transcribed, %d already existed, %d errors",
        stats["total"], stats["transcribed"], stats["skipped"], stats["errors"],
    )


if __name__ == "__main__":
    main()
