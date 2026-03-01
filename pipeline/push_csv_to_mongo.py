#!/usr/bin/env python3
"""Push McCar CSV telemetry files into MongoDB telemetry collection.
Skips files that are already in MongoDB (by _source_file).
"""
import os, sys, csv
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

URI = os.environ["MONGODB_URI"]
DB_NAME = os.environ.get("MONGODB_DB", "marip_f1")

client = MongoClient(URI)
db = client[DB_NAME]
coll = db["telemetry"]

# Check what's already in MongoDB
existing_sources = set(coll.distinct("_source_file"))
print(f"Already in MongoDB: {len(existing_sources)} source files")

MCCAR_DIR = Path(__file__).resolve().parent.parent / "f1data" / "McCar"
NUMERIC_FIELDS = {"RPM", "Speed", "nGear", "Throttle", "Distance", "TyreLife", "LapNumber"}
BOOL_FIELDS = {"Brake"}

total_inserted = 0

for year_dir in sorted(MCCAR_DIR.iterdir()):
    if not year_dir.is_dir():
        continue
    for csv_file in sorted(year_dir.glob("*.csv")):
        if csv_file.name.startswith("ALL"):
            continue
        if csv_file.name in existing_sources:
            print(f"  SKIP {csv_file.name} (already in MongoDB)")
            continue

        print(f"  Loading {csv_file.name}...", end=" ", flush=True)
        docs = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc = {"_source_file": csv_file.name}
                for k, v in row.items():
                    if not k or not v or v == "":
                        continue
                    if k in NUMERIC_FIELDS:
                        try:
                            doc[k] = float(v)
                        except (ValueError, TypeError):
                            doc[k] = v
                    elif k in BOOL_FIELDS:
                        doc[k] = v == "True" or v == "true" or v == "1"
                    else:
                        doc[k] = v
                docs.append(doc)

        if docs:
            # Insert in batches of 10000
            for i in range(0, len(docs), 10000):
                batch = docs[i:i+10000]
                coll.insert_many(batch, ordered=False)
            total_inserted += len(docs)
            print(f"{len(docs)} rows inserted")
        else:
            print("0 rows (empty)")

print(f"\nDone! Total inserted: {total_inserted}")
print(f"Total in telemetry collection: {coll.count_documents({})}")
