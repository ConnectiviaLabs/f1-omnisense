"""Download McLaren team radio MP3s from the F1 livetiming server.

Strategy:
  1. Fetch meetings + sessions from OpenF1 to get meeting names
  2. For each session, try TeamRadio.json from the livetiming server
  3. Filter captures for McLaren driver numbers only
  4. Download MP3s to local directory

The livetiming path convention is:
    {year}/{session_date}_{Meeting_Name}/{session_date}_{Session_Type}/TeamRadio.json

For 2018-2022 where OpenF1 has no data, we brute-force known race paths.

Output structure:
    data/team_radio/{year}/{Meeting_Name}/{Session_Type}/{filename}.mp3

Usage:
    python3 -m pipeline.updater.download_team_radio                   # download all
    python3 -m pipeline.updater.download_team_radio --year 2024       # single year
    python3 -m pipeline.updater.download_team_radio --dry-run         # show what would download
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "team_radio"

LIVETIMING_BASE = "https://livetiming.formula1.com/static"
OPENF1_BASE = "https://api.openf1.org/v1"

REQUEST_DELAY = 0.4  # seconds between requests

# McLaren driver numbers by season
MCLAREN_DRIVERS: dict[int, dict[int, str]] = {
    2018: {14: "Alonso", 2: "Vandoorne"},
    2019: {55: "Sainz", 4: "Norris"},
    2020: {55: "Sainz", 4: "Norris"},
    2021: {3: "Ricciardo", 4: "Norris"},
    2022: {3: "Ricciardo", 4: "Norris"},
    2023: {81: "Piastri", 4: "Norris"},
    2024: {81: "Piastri", 4: "Norris"},
}

# Session name -> livetiming folder name
SESSION_NAME_MAP = {
    "Practice 1": "Practice_1",
    "Practice 2": "Practice_2",
    "Practice 3": "Practice_3",
    "Qualifying": "Qualifying",
    "Sprint": "Sprint",
    "Sprint Qualifying": "Sprint_Qualifying",
    "Sprint Shootout": "Sprint_Shootout",
    "Race": "Race",
}

# ── Historical race calendars (2018-2022) ────────────────────────────────
# OpenF1 only covers 2023+. For earlier years we hardcode the race dates
# and meeting names as they appear on livetiming.formula1.com.

HISTORICAL_RACES: dict[int, list[tuple[str, str]]] = {
    2018: [
        ("2018-03-25", "Australian_Grand_Prix"),
        ("2018-04-08", "Bahrain_Grand_Prix"),
        ("2018-04-15", "Chinese_Grand_Prix"),
        ("2018-04-29", "Azerbaijan_Grand_Prix"),
        ("2018-05-13", "Spanish_Grand_Prix"),
        ("2018-05-27", "Monaco_Grand_Prix"),
        ("2018-06-10", "Canadian_Grand_Prix"),
        ("2018-06-24", "French_Grand_Prix"),
        ("2018-07-01", "Austrian_Grand_Prix"),
        ("2018-07-08", "British_Grand_Prix"),
        ("2018-07-22", "German_Grand_Prix"),
        ("2018-07-29", "Hungarian_Grand_Prix"),
        ("2018-08-26", "Belgian_Grand_Prix"),
        ("2018-09-02", "Italian_Grand_Prix"),
        ("2018-09-16", "Singapore_Grand_Prix"),
        ("2018-09-30", "Russian_Grand_Prix"),
        ("2018-10-07", "Japanese_Grand_Prix"),
        ("2018-10-21", "United_States_Grand_Prix"),
        ("2018-10-28", "Mexican_Grand_Prix"),
        ("2018-11-11", "Brazilian_Grand_Prix"),
        ("2018-11-25", "Abu_Dhabi_Grand_Prix"),
    ],
    2019: [
        ("2019-03-17", "Australian_Grand_Prix"),
        ("2019-03-31", "Bahrain_Grand_Prix"),
        ("2019-04-14", "Chinese_Grand_Prix"),
        ("2019-04-28", "Azerbaijan_Grand_Prix"),
        ("2019-05-12", "Spanish_Grand_Prix"),
        ("2019-05-26", "Monaco_Grand_Prix"),
        ("2019-06-09", "Canadian_Grand_Prix"),
        ("2019-06-23", "French_Grand_Prix"),
        ("2019-06-30", "Austrian_Grand_Prix"),
        ("2019-07-14", "British_Grand_Prix"),
        ("2019-07-28", "German_Grand_Prix"),
        ("2019-08-04", "Hungarian_Grand_Prix"),
        ("2019-09-01", "Belgian_Grand_Prix"),
        ("2019-09-08", "Italian_Grand_Prix"),
        ("2019-09-22", "Singapore_Grand_Prix"),
        ("2019-09-29", "Russian_Grand_Prix"),
        ("2019-10-13", "Japanese_Grand_Prix"),
        ("2019-10-27", "Mexican_Grand_Prix"),
        ("2019-11-03", "United_States_Grand_Prix"),
        ("2019-11-17", "Brazilian_Grand_Prix"),
        ("2019-12-01", "Abu_Dhabi_Grand_Prix"),
    ],
    2020: [
        ("2020-07-05", "Austrian_Grand_Prix"),
        ("2020-07-12", "Styrian_Grand_Prix"),
        ("2020-07-19", "Hungarian_Grand_Prix"),
        ("2020-08-02", "British_Grand_Prix"),
        ("2020-08-09", "70th_Anniversary_Grand_Prix"),
        ("2020-08-16", "Spanish_Grand_Prix"),
        ("2020-08-30", "Belgian_Grand_Prix"),
        ("2020-09-06", "Italian_Grand_Prix"),
        ("2020-09-13", "Tuscan_Grand_Prix"),
        ("2020-09-27", "Russian_Grand_Prix"),
        ("2020-10-11", "Eifel_Grand_Prix"),
        ("2020-10-25", "Portuguese_Grand_Prix"),
        ("2020-11-01", "Emilia_Romagna_Grand_Prix"),
        ("2020-11-15", "Turkish_Grand_Prix"),
        ("2020-11-29", "Bahrain_Grand_Prix"),
        ("2020-12-06", "Sakhir_Grand_Prix"),
        ("2020-12-13", "Abu_Dhabi_Grand_Prix"),
    ],
    2021: [
        ("2021-03-28", "Bahrain_Grand_Prix"),
        ("2021-04-18", "Emilia_Romagna_Grand_Prix"),
        ("2021-05-02", "Portuguese_Grand_Prix"),
        ("2021-05-09", "Spanish_Grand_Prix"),
        ("2021-05-23", "Monaco_Grand_Prix"),
        ("2021-06-06", "Azerbaijan_Grand_Prix"),
        ("2021-06-20", "French_Grand_Prix"),
        ("2021-06-27", "Styrian_Grand_Prix"),
        ("2021-07-04", "Austrian_Grand_Prix"),
        ("2021-07-18", "British_Grand_Prix"),
        ("2021-08-01", "Hungarian_Grand_Prix"),
        ("2021-08-29", "Belgian_Grand_Prix"),
        ("2021-09-05", "Dutch_Grand_Prix"),
        ("2021-09-12", "Italian_Grand_Prix"),
        ("2021-09-26", "Russian_Grand_Prix"),
        ("2021-10-10", "Turkish_Grand_Prix"),
        ("2021-10-24", "United_States_Grand_Prix"),
        ("2021-11-07", "Mexico_City_Grand_Prix"),
        ("2021-11-14", "S\u00e3o_Paulo_Grand_Prix"),
        ("2021-11-21", "Qatar_Grand_Prix"),
        ("2021-12-05", "Saudi_Arabian_Grand_Prix"),
        ("2021-12-12", "Abu_Dhabi_Grand_Prix"),
    ],
    2022: [
        ("2022-03-20", "Bahrain_Grand_Prix"),
        ("2022-03-27", "Saudi_Arabian_Grand_Prix"),
        ("2022-04-10", "Australian_Grand_Prix"),
        ("2022-04-24", "Emilia_Romagna_Grand_Prix"),
        ("2022-05-08", "Miami_Grand_Prix"),
        ("2022-05-22", "Spanish_Grand_Prix"),
        ("2022-05-29", "Monaco_Grand_Prix"),
        ("2022-06-12", "Azerbaijan_Grand_Prix"),
        ("2022-06-19", "Canadian_Grand_Prix"),
        ("2022-07-03", "British_Grand_Prix"),
        ("2022-07-10", "Austrian_Grand_Prix"),
        ("2022-07-24", "French_Grand_Prix"),
        ("2022-07-31", "Hungarian_Grand_Prix"),
        ("2022-08-28", "Belgian_Grand_Prix"),
        ("2022-09-04", "Dutch_Grand_Prix"),
        ("2022-09-11", "Italian_Grand_Prix"),
        ("2022-10-02", "Singapore_Grand_Prix"),
        ("2022-10-09", "Japanese_Grand_Prix"),
        ("2022-10-23", "United_States_Grand_Prix"),
        ("2022-10-30", "Mexico_City_Grand_Prix"),
        ("2022-11-13", "S\u00e3o_Paulo_Grand_Prix"),
        ("2022-11-20", "Abu_Dhabi_Grand_Prix"),
    ],
}

# Session date offsets from race date (standard weekend)
# Monaco 2018-2021 had FP1/FP2 on Thursday (-3), but we just try all offsets
STANDARD_SESSION_OFFSETS: dict[str, list[int]] = {
    "Practice_1": [-2, -3],       # Friday, or Thursday (Monaco)
    "Practice_2": [-2, -3],       # Friday, or Thursday (Monaco)
    "Practice_3": [-1],           # Saturday
    "Qualifying": [-1, -2],       # Saturday, or Friday (sprint weekends)
    "Race": [0],                  # Sunday
    "Sprint": [-1],               # Saturday (2022+)
    "Sprint_Qualifying": [-1],    # Saturday (2021)
    "Practice": [-1],             # 2020 Emilia Romagna two-day weekend
}


# ── Helpers ──────────────────────────────────────────────────────────────


def _get_json(url: str, timeout: int = 30) -> dict | list | None:
    """Fetch JSON, return None on 403/404/error.

    Handles UTF-8 BOM that the F1 livetiming server includes in some responses.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code in (403, 404):
            return None
        resp.raise_for_status()
        # Livetiming server sends UTF-8 BOM — decode with utf-8-sig
        text = resp.content.decode("utf-8-sig")
        return json.loads(text)
    except (requests.RequestException, ValueError) as e:
        logger.debug("GET failed %s: %s", url, e)
        return None


def _download_mp3(url: str, dest: Path) -> bool:
    """Download MP3. Returns True on success, skips if exists."""
    if dest.exists():
        return True
    try:
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code in (403, 404):
            return False
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        logger.warning("Download failed %s: %s", url, e)
        if dest.exists():
            dest.unlink()
        return False


# ── OpenF1 Discovery ─────────────────────────────────────────────────────


def _get_meetings(year: int) -> list[dict]:
    """Get all meetings for a year from OpenF1."""
    data = _get_json(f"{OPENF1_BASE}/meetings?year={year}")
    return data if isinstance(data, list) else []


def _get_sessions(year: int) -> list[dict]:
    """Get all sessions for a year from OpenF1."""
    data = _get_json(f"{OPENF1_BASE}/sessions?year={year}")
    return data if isinstance(data, list) else []


def _build_session_index(year: int) -> list[dict]:
    """Build enriched session list with meeting names and race dates.

    The livetiming meeting folder uses the Race session date, not the meeting
    start date. Each session subfolder uses its own date.

    Returns list of dicts with keys:
        meeting_name, session_name, session_date, race_date, session_key, year
    """
    meetings = _get_meetings(year)
    sessions = _get_sessions(year)

    if not meetings or not sessions:
        return []

    # Map meeting_key -> meeting_name (skip testing)
    meeting_map = {}
    for m in meetings:
        mk = m.get("meeting_key")
        name = m.get("meeting_name", "")
        if mk and name and "Testing" not in name:
            meeting_map[mk] = name

    # Group sessions by meeting_key
    meeting_sessions: dict[int, list[dict]] = {}
    for s in sessions:
        mk = s.get("meeting_key")
        if mk and mk in meeting_map:
            meeting_sessions.setdefault(mk, []).append(s)

    # Find the Race date for each meeting (used for the meeting folder name)
    race_dates: dict[int, str] = {}
    for mk, sess_list in meeting_sessions.items():
        for s in sess_list:
            if s.get("session_name") == "Race":
                race_dates[mk] = s["date_start"][:10]
                break
        # Sprint weekends without a "Race" label — fall back to latest session date
        if mk not in race_dates:
            latest = max(sess_list, key=lambda x: x.get("date_start", ""))
            race_dates[mk] = latest["date_start"][:10]

    enriched = []
    for s in sessions:
        mk = s.get("meeting_key")
        meeting_name = meeting_map.get(mk)
        session_name = s.get("session_name", "")
        date_start = s.get("date_start", "")
        race_date = race_dates.get(mk)

        if not meeting_name or not session_name or not date_start or not race_date:
            continue

        if session_name not in SESSION_NAME_MAP:
            continue

        enriched.append({
            "meeting_name": meeting_name,
            "session_name": session_name,
            "session_date": date_start[:10],
            "race_date": race_date,
            "session_key": s.get("session_key"),
            "year": year,
        })

    return enriched


# ── Livetiming Path Resolution ───────────────────────────────────────────


def _try_radio_json(year: int, meeting_slug: str, session_slug: str,
                    session_date: str, race_date: str) -> tuple[str | None, dict | None]:
    """Fetch TeamRadio.json from livetiming.

    Path convention:
        {year}/{race_date}_{Meeting}_{GP}/{session_date}_{Session}/TeamRadio.json

    The meeting folder always uses the Race date; the session subfolder
    uses the session's own date.

    Returns (base_url, radio_data) or (None, None).
    """
    path = f"{year}/{race_date}_{meeting_slug}/{session_date}_{session_slug}"
    url = f"{LIVETIMING_BASE}/{path}/TeamRadio.json"

    time.sleep(REQUEST_DELAY)
    data = _get_json(url)
    if data and data.get("Captures"):
        return f"{LIVETIMING_BASE}/{path}", data

    return None, None


# ── Main Download Logic ──────────────────────────────────────────────────


def download_year(year: int, dry_run: bool = False) -> dict:
    """Download all McLaren team radio for a year."""
    drivers = MCLAREN_DRIVERS.get(year)
    if not drivers:
        logger.warning("No McLaren driver mapping for %d", year)
        return {"year": year, "sessions": 0, "clips": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    driver_numbers = set(drivers.keys())
    logger.info("Year %d — McLaren drivers: %s", year, drivers)

    # Build session index from OpenF1
    session_index = _build_session_index(year)
    if not session_index:
        logger.warning("No sessions found for %d via OpenF1", year)
        return {"year": year, "sessions": 0, "clips": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    logger.info("Found %d sessions for %d", len(session_index), year)

    stats = {"year": year, "sessions": 0, "clips": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    for si in session_index:
        meeting_name = si["meeting_name"]
        session_name = si["session_name"]
        session_date = si["session_date"]
        meeting_slug = meeting_name.replace(" ", "_")
        session_slug = SESSION_NAME_MAP[session_name]

        label = f"{meeting_name} — {session_name}"

        race_date = si["race_date"]

        # Try to fetch TeamRadio.json
        base_url, radio_data = _try_radio_json(year, meeting_slug, session_slug, session_date, race_date)
        if not radio_data:
            logger.debug("  No radio: %s", label)
            continue

        captures = radio_data.get("Captures", [])

        # Filter for McLaren only
        mclaren_clips = [
            c for c in captures
            if int(c.get("RacingNumber", 0)) in driver_numbers
        ]
        if not mclaren_clips:
            continue

        stats["sessions"] += 1
        stats["clips"] += len(mclaren_clips)

        logger.info("  %s: %d McLaren clips (of %d total)", label, len(mclaren_clips), len(captures))

        if dry_run:
            stats["skipped"] += len(mclaren_clips)
            continue

        # Download
        out_dir = OUTPUT_DIR / str(year) / meeting_slug / session_slug
        for clip in mclaren_clips:
            mp3_path = clip.get("Path", "")
            if not mp3_path:
                continue

            mp3_url = f"{base_url}/{mp3_path}"
            dest = out_dir / Path(mp3_path).name

            if dest.exists():
                stats["skipped"] += 1
                continue

            time.sleep(REQUEST_DELAY)
            if _download_mp3(mp3_url, dest):
                stats["downloaded"] += 1
            else:
                stats["errors"] += 1

    return stats


def download_year_historical(year: int, dry_run: bool = False) -> dict:
    """Download team radio for 2018-2022 using hardcoded race calendar.

    Brute-forces session dates using offsets from the race date.
    """
    drivers = MCLAREN_DRIVERS.get(year)
    races = HISTORICAL_RACES.get(year)
    if not drivers or not races:
        return {"year": year, "sessions": 0, "clips": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    from datetime import date, timedelta

    driver_numbers = set(drivers.keys())
    logger.info("Year %d — McLaren drivers: %s", year, drivers)
    logger.info("Calendar: %d races", len(races))

    stats = {"year": year, "sessions": 0, "clips": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    for race_date_str, meeting_slug in races:
        race_date = date.fromisoformat(race_date_str)
        meeting_label = meeting_slug.replace("_", " ")

        # Try each session type with its possible date offsets
        for session_type, offsets in STANDARD_SESSION_OFFSETS.items():
            for offset in offsets:
                session_date = race_date + timedelta(days=offset)
                session_date_str = session_date.isoformat()

                base_url, radio_data = _try_radio_json(
                    year, meeting_slug, session_type, session_date_str, race_date_str
                )
                if not radio_data:
                    continue

                captures = radio_data.get("Captures", [])
                mclaren_clips = [
                    c for c in captures
                    if int(c.get("RacingNumber", 0)) in driver_numbers
                ]
                if not mclaren_clips:
                    break  # found the session but no McLaren clips

                stats["sessions"] += 1
                stats["clips"] += len(mclaren_clips)

                label = f"{meeting_label} — {session_type}"
                logger.info("  %s: %d McLaren clips (of %d total)",
                            label, len(mclaren_clips), len(captures))

                if dry_run:
                    stats["skipped"] += len(mclaren_clips)
                    break

                out_dir = OUTPUT_DIR / str(year) / meeting_slug / session_type
                for clip in mclaren_clips:
                    mp3_path = clip.get("Path", "")
                    if not mp3_path:
                        continue
                    mp3_url = f"{base_url}/{mp3_path}"
                    dest = out_dir / Path(mp3_path).name
                    if dest.exists():
                        stats["skipped"] += 1
                        continue
                    time.sleep(REQUEST_DELAY)
                    if _download_mp3(mp3_url, dest):
                        stats["downloaded"] += 1
                    else:
                        stats["errors"] += 1

                break  # found the right offset, move to next session type

    return stats


def download_year_openf1_radio(year: int, dry_run: bool = False) -> dict:
    """Fallback: use OpenF1 team_radio endpoint for exact URLs (2023+ only)."""
    drivers = MCLAREN_DRIVERS.get(year)
    if not drivers:
        return {"year": year, "sessions": 0, "clips": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    # Build session index for folder naming
    session_index = _build_session_index(year)

    stats = {"year": year, "sessions_set": set(), "clips": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    for driver_num, driver_name in drivers.items():
        logger.info("  OpenF1 radio: %s (#%d)", driver_name, driver_num)

        for si in session_index:
            sk = si["session_key"]
            time.sleep(REQUEST_DELAY)
            clips = _get_json(f"{OPENF1_BASE}/team_radio?session_key={sk}&driver_number={driver_num}")
            if not clips:
                continue

            meeting_slug = si["meeting_name"].replace(" ", "_")
            session_slug = SESSION_NAME_MAP[si["session_name"]]
            out_dir = OUTPUT_DIR / str(year) / meeting_slug / session_slug

            stats["sessions_set"].add(sk)
            stats["clips"] += len(clips)

            label = f"{si['meeting_name']} — {si['session_name']}"
            logger.info("    %s: %d clips for %s", label, len(clips), driver_name)

            if dry_run:
                stats["skipped"] += len(clips)
                continue

            for clip in clips:
                rec_url = clip.get("recording_url", "")
                if not rec_url:
                    continue
                dest = out_dir / Path(rec_url).name
                if dest.exists():
                    stats["skipped"] += 1
                    continue
                time.sleep(REQUEST_DELAY)
                if _download_mp3(rec_url, dest):
                    stats["downloaded"] += 1
                else:
                    stats["errors"] += 1

    stats["sessions"] = len(stats.pop("sessions_set"))
    return stats


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Download McLaren team radio MP3s")
    parser.add_argument("--year", type=int, help="Single year (default: all 2018-2024)")
    parser.add_argument("--from-year", type=int, help="Start year for range")
    parser.add_argument("--to-year", type=int, help="End year for range (inclusive)")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without downloading")
    parser.add_argument("--openf1-fallback", action="store_true",
                        help="Use OpenF1 team_radio API as fallback (2023+ only)")
    args = parser.parse_args()

    if args.year:
        years = [args.year]
    elif args.from_year or args.to_year:
        years = list(range(args.from_year or 2018, (args.to_year or 2024) + 1))
    else:
        years = list(range(2018, 2025))

    if args.dry_run:
        logger.info("DRY RUN — no files will be downloaded")
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Years: %s", years)

    total = {"sessions": 0, "clips": 0, "downloaded": 0, "skipped": 0, "errors": 0}

    for year in years:
        logger.info("=== %d ================================================", year)

        if year <= 2022:
            # Pre-OpenF1 era: use hardcoded race calendar
            stats = download_year_historical(year, dry_run=args.dry_run)
        else:
            # 2023+: use OpenF1 for session discovery
            stats = download_year(year, dry_run=args.dry_run)

            # Fallback to OpenF1 team_radio endpoint if needed
            if stats["clips"] == 0 and args.openf1_fallback:
                logger.info("  Trying OpenF1 fallback for %d...", year)
                stats = download_year_openf1_radio(year, dry_run=args.dry_run)

        for k in ("sessions", "clips", "downloaded", "skipped", "errors"):
            total[k] += stats.get(k, 0)

        logger.info(
            "  %d: %d sessions, %d clips, %d new, %d existed, %d errors",
            year, stats.get("sessions", 0), stats.get("clips", 0),
            stats.get("downloaded", 0), stats.get("skipped", 0), stats.get("errors", 0),
        )

    logger.info("========================================================")
    logger.info(
        "TOTAL: %d sessions, %d McLaren clips, %d downloaded, %d existed, %d errors",
        total["sessions"], total["clips"], total["downloaded"], total["skipped"], total["errors"],
    )


if __name__ == "__main__":
    main()
