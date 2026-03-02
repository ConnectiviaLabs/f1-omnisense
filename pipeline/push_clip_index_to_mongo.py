"""Push the local clip_index.json to MongoDB so RunPod can access it.

Usage:
    python pipeline/push_clip_index_to_mongo.py
"""

import json
import os
from pathlib import Path

from pymongo import MongoClient

CLIP_INDEX_PATH = Path(__file__).parent.parent / "f1data" / "McMedia" / "clip_index.json"

def main():
    if not CLIP_INDEX_PATH.exists():
        print(f"ERROR: {CLIP_INDEX_PATH} not found. Run: python pipeline/clip_index.py")
        return

    uri = os.getenv("MONGODB_URI", "")
    db_name = os.getenv("MONGODB_DB", "marip_f1")
    if not uri:
        print("ERROR: MONGODB_URI env var not set")
        return

    print(f"Loading {CLIP_INDEX_PATH} ...")
    with open(CLIP_INDEX_PATH) as f:
        data = json.load(f)

    print(f"  {len(data['images'])} images, {len(data['categories'])} categories")

    client = MongoClient(uri)
    db = client[db_name]
    coll = db["clip_index"]

    # Store as a single document (570KB fits well within 16MB BSON limit)
    coll.delete_many({})
    coll.insert_one({"_type": "clip_index", **data})

    print(f"  Pushed to {db_name}.clip_index collection")
    print("  Done!")

if __name__ == "__main__":
    main()
