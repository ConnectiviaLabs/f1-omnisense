"""
Build Telemetry Summaries
─────────────────────────
Pre-aggregates telemetry_compressed into compact summary collections
that chat_server can query directly instead of scanning 60M raw rows.

Collections created:
  - telemetry_lap_summary   : per-driver per-race per-lap aggregated stats
                              (~200K docs, serves laps/positions/pits/stints endpoints)
  - telemetry_race_summary  : per-driver per-race aggregated car/biometric stats
                              (~2K docs, serves mccar-summary/mcdriver-summary)

Usage:
    python -m pipeline.enrichment.build_telemetry_summaries
"""

from __future__ import annotations

import gzip
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from updater._db import get_db


def load_race_telemetry(db) -> pd.DataFrame:
    """Load race telemetry from telemetry_compressed."""
    filenames = sorted(db["telemetry_compressed"].distinct("filename"))
    race_files = [f for f in filenames if "_R." in f]

    # Build driver number → code mapping
    num_to_code = {}
    for doc in db["openf1_drivers"].find({}, {"driver_number": 1, "name_acronym": 1, "_id": 0}):
        num_to_code[str(doc["driver_number"])] = doc["name_acronym"]

    frames = []
    for fi, fname in enumerate(race_files):
        chunks = list(db["telemetry_compressed"].find(
            {"filename": fname},
            {"data": 1, "chunk": 1, "_id": 0},
        ))
        chunks.sort(key=lambda d: d.get("chunk", 0))

        for doc in chunks:
            try:
                df = pickle.loads(gzip.decompress(doc["data"]))
                frames.append(df)
            except Exception:
                pass

        if (fi + 1) % 10 == 0:
            print(f"  [{fi + 1}/{len(race_files)}] loaded...")

    if not frames:
        return pd.DataFrame()

    tel = pd.concat(frames, ignore_index=True)
    tel["Driver"] = tel["Driver"].astype(str).map(num_to_code)
    tel = tel.dropna(subset=["Driver"])
    return tel


def build_lap_summaries(tel: pd.DataFrame, db) -> int:
    """Build telemetry_lap_summary: per-driver per-race per-lap stats."""
    print("\n  Building lap summaries...")

    # Generate _source_file
    tel["_source_file"] = tel.apply(
        lambda r: f"{int(r['Year'])}_{str(r['Race']).replace(' ', '_')}_Race.csv", axis=1
    )
    source_files = sorted(tel["_source_file"].unique())

    grouped = tel.groupby(["Driver", "Year", "Race", "LapNumber"])

    lap_docs = []
    for (driver, year, race, lap_num), grp in grouped:
        doc = {
            "Driver": driver,
            "Year": int(year),
            "Race": race,
            "LapNumber": int(lap_num),
            "top_speed": round(float(grp["Speed"].max()), 1) if "Speed" in grp.columns else None,
            "avg_speed": round(float(grp["Speed"].mean()), 1) if "Speed" in grp.columns else None,
            "_source_file": f"{int(year)}_{race.replace(' ', '_')}_Race.csv",
        }

        if "LapTime_s" in grp.columns:
            lt = grp["LapTime_s"].iloc[0]
            if pd.notna(lt):
                m, s = divmod(float(lt), 60)
                doc["LapTime"] = f"0 days 00:{int(m):02d}:{s:06.3f}"
                doc["LapTime_s"] = round(float(lt), 3)

        if "Date" in grp.columns:
            doc["Date"] = str(grp["Date"].iloc[0])

        if "Compound" in grp.columns:
            compounds = grp["Compound"].dropna().unique()
            doc["Compound"] = str(compounds[0]) if len(compounds) > 0 else None

        if "TyreLife" in grp.columns:
            tl = grp["TyreLife"].dropna()
            doc["TyreLife"] = int(tl.min()) if len(tl) > 0 else None

        if "Stint" in grp.columns:
            st = grp["Stint"].dropna()
            doc["Stint"] = int(st.iloc[0]) if len(st) > 0 else None

        doc["sample_count"] = len(grp)
        lap_docs.append(doc)

        if len(lap_docs) % 50000 == 0:
            print(f"    {len(lap_docs):,} laps processed...")

    print(f"  Total: {len(lap_docs):,} lap summaries")

    if lap_docs:
        db["telemetry_lap_summary"].drop()
        # Insert in batches
        BATCH = 10000
        for i in range(0, len(lap_docs), BATCH):
            db["telemetry_lap_summary"].insert_many(lap_docs[i:i + BATCH])

        db["telemetry_lap_summary"].create_index("_source_file")
        db["telemetry_lap_summary"].create_index([("Driver", 1), ("Year", 1), ("Race", 1)])
        db["telemetry_lap_summary"].create_index([("Year", 1), ("Race", 1), ("LapNumber", 1)])

    return len(lap_docs)


def build_race_summaries(tel: pd.DataFrame, db) -> int:
    """Build telemetry_race_summary: per-driver per-race car/biometric stats."""
    print("\n  Building race summaries...")

    grouped = tel.groupby(["Driver", "Year", "Race"])

    race_docs = []
    for (driver, year, race), grp in grouped:
        speed = grp["Speed"].dropna() if "Speed" in grp.columns else pd.Series(dtype=float)
        rpm = grp["RPM"].dropna() if "RPM" in grp.columns else pd.Series(dtype=float)
        throttle = grp["Throttle"].dropna() if "Throttle" in grp.columns else pd.Series(dtype=float)
        brake = grp["Brake"] if "Brake" in grp.columns else pd.Series(dtype=bool)
        drs = grp["DRS"] if "DRS" in grp.columns else pd.Series(dtype=int)

        doc = {
            "Driver": driver,
            "Year": int(year),
            "Race": race,
            "samples": len(grp),
            "_source_file": f"{int(year)}_{race.replace(' ', '_')}_Race.csv",
        }

        if len(speed) > 0:
            doc["avg_speed"] = round(float(speed.mean()), 1)
            doc["top_speed"] = round(float(speed.quantile(0.99)), 1)

        if len(rpm) > 0:
            doc["avg_rpm"] = round(float(rpm.mean()), 0)
            doc["max_rpm"] = round(float(rpm.quantile(0.99)), 0)

        if len(throttle) > 0:
            doc["avg_throttle"] = round(float(throttle.mean()), 1)

        if len(brake) > 0:
            brake_bool = brake.astype(bool) if brake.dtype != bool else brake
            doc["brake_pct"] = round(float(brake_bool.sum() / len(brake_bool) * 100), 1)

        if len(drs) > 0:
            drs_num = pd.to_numeric(drs, errors="coerce").dropna()
            if len(drs_num) > 0:
                doc["drs_pct"] = round(float((drs_num >= 10).sum() / len(drs_num) * 100), 1)

        if "Compound" in grp.columns:
            compounds = sorted(grp["Compound"].dropna().unique().tolist())
            doc["compounds"] = compounds

        race_docs.append(doc)

    print(f"  Total: {len(race_docs):,} race summaries")

    if race_docs:
        db["telemetry_race_summary"].drop()
        db["telemetry_race_summary"].insert_many(race_docs)
        db["telemetry_race_summary"].create_index([("Driver", 1), ("Year", 1)])
        db["telemetry_race_summary"].create_index("_source_file")

    return len(race_docs)


def main():
    db = get_db()
    print("✅ Connected to MongoDB")

    count = db["telemetry_compressed"].count_documents({})
    if count == 0:
        print("  ⚠ telemetry_compressed is empty, nothing to summarize")
        return

    print(f"\nLoading race telemetry from {count} compressed chunks...")
    tel = load_race_telemetry(db)

    if tel.empty:
        print("  ⚠ No race telemetry found")
        return

    print(f"  Loaded {len(tel):,} rows, {tel['Driver'].nunique()} drivers")
    now = datetime.now(timezone.utc)

    lap_count = build_lap_summaries(tel, db)
    race_count = build_race_summaries(tel, db)

    # Stats
    print(f"\n✅ Pre-aggregation complete:")
    print(f"  telemetry_lap_summary:  {lap_count:,} docs")
    print(f"  telemetry_race_summary: {race_count:,} docs")

    # Size check
    for coll in ["telemetry_lap_summary", "telemetry_race_summary"]:
        try:
            cs = db.command("collStats", coll)
            size_mb = cs["storageSize"] / (1024 * 1024)
            print(f"  {coll}: {size_mb:.1f} MB on disk")
        except Exception:
            pass


if __name__ == "__main__":
    main()
