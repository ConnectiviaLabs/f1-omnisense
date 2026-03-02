"""Push existing McMedia pipeline results (JSON files) into MongoDB collections.

Usage:
    python pipeline/push_media_results_to_mongo.py

Reads from f1data/McMedia/*_results/*.json and inserts into:
  - pipeline_gdino_results
  - pipeline_fused_results  (same as gdino for now)
  - pipeline_minicpm_results
  - pipeline_videomae_results
  - pipeline_timesformer_results
  - media_videos  (video catalog for the picker UI)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGODB_URI", "")
DB_NAME = os.environ.get("MONGODB_DB", "marip_f1")
MEDIA_DIR = Path(__file__).resolve().parent.parent / "f1data" / "McMedia"


def main():
    if not MONGO_URI:
        print("MONGODB_URI not set")
        sys.exit(1)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    ts = datetime.now(timezone.utc)

    # ── Push result JSONs ────────────────────────────────────────────
    mapping = {
        "gdino_results/gdino_results.json": "pipeline_gdino_results",
        "videomae_results/videomae_results.json": "pipeline_videomae_results",
        "minicpm_results/minicpm_results.json": "pipeline_minicpm_results",
        "timesformer_results/timesformer_results.json": "pipeline_timesformer_results",
    }

    for rel_path, collection_name in mapping.items():
        fpath = MEDIA_DIR / rel_path
        if not fpath.exists():
            print(f"  skip {rel_path} (not found)")
            continue

        with open(fpath) as f:
            data = json.load(f)

        col = db[collection_name]
        inserted = 0
        for video_name, result in data.items():
            # Skip if already inserted for this video
            if col.find_one({"filename": video_name}):
                print(f"  {collection_name}/{video_name} already exists, skipping")
                continue

            doc = {"filename": video_name, "timestamp": ts}
            if isinstance(result, list):
                # gdino/minicpm: list of frames
                doc["frames"] = result
            elif isinstance(result, dict):
                # videomae/timesformer: single result object
                doc.update(result)
            col.insert_one(doc)
            inserted += 1

        print(f"  {collection_name}: inserted {inserted} docs")

    # ── Push video catalog ───────────────────────────────────────────
    video_col = db["media_videos"]
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = sorted(
        p for p in MEDIA_DIR.iterdir()
        if p.suffix.lower() in video_exts and p.stat().st_size > 0
    )

    for vp in videos:
        if video_col.find_one({"filename": vp.name}):
            print(f"  media_videos/{vp.name} already exists, skipping")
            continue
        video_col.insert_one({
            "filename": vp.name,
            "size_mb": round(vp.stat().st_size / (1024 * 1024), 1),
            "added": ts,
            "analyzed": False,
        })
        print(f"  media_videos: added {vp.name} ({vp.stat().st_size / (1024*1024):.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
