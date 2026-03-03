#!/usr/bin/env python3
"""Migrate all collections from MongoDB Atlas to self-hosted MongoDB.

Usage:
    # Set both URIs, then run:
    ATLAS_URI="mongodb+srv://user:pass@cluster.mongodb.net/marip_f1" \
    LOCAL_URI="mongodb://admin:maripf1admin@localhost:27017/marip_f1?authSource=admin" \
    python scripts/migrate_atlas_to_local.py

    # Or with defaults from .env (reads MONGODB_URI_ATLAS):
    python scripts/migrate_atlas_to_local.py
"""

import os
import sys
import time
from pathlib import Path

from pymongo import MongoClient


def load_env():
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


def migrate(atlas_uri: str, local_uri: str, db_name: str = "marip_f1"):
    print(f"Connecting to Atlas: {atlas_uri[:50]}...")
    atlas_client = MongoClient(atlas_uri)
    atlas_db = atlas_client[db_name]

    print(f"Connecting to local: {local_uri[:50]}...")
    local_client = MongoClient(local_uri)
    local_db = local_client[db_name]

    collections = atlas_db.list_collection_names()
    print(f"\nFound {len(collections)} collections to migrate\n")

    total_docs = 0
    batch_size = 1000

    for i, coll_name in enumerate(sorted(collections), 1):
        atlas_coll = atlas_db[coll_name]
        local_coll = local_db[coll_name]

        doc_count = atlas_coll.count_documents({})
        if doc_count == 0:
            print(f"  [{i}/{len(collections)}] {coll_name}: empty, skipping")
            continue

        # Drop local collection if it exists (fresh migration)
        local_coll.drop()

        print(f"  [{i}/{len(collections)}] {coll_name}: {doc_count:,} docs ... ", end="", flush=True)
        start = time.time()

        migrated = 0
        cursor = atlas_coll.find({}, batch_size=batch_size)
        batch = []

        for doc in cursor:
            batch.append(doc)
            if len(batch) >= batch_size:
                local_coll.insert_many(batch)
                migrated += len(batch)
                batch = []

        if batch:
            local_coll.insert_many(batch)
            migrated += len(batch)

        elapsed = time.time() - start
        total_docs += migrated
        print(f"{migrated:,} migrated ({elapsed:.1f}s)")

        # Recreate indexes (skip _id and search indexes)
        try:
            for idx in atlas_coll.list_indexes():
                idx_name = idx.get("name", "")
                if idx_name == "_id_":
                    continue
                # Skip Atlas-specific search indexes
                if "vectorSearch" in str(idx) or "search" in idx_name.lower():
                    continue
                key = list(idx["key"].items())
                opts = {}
                if idx.get("unique"):
                    opts["unique"] = True
                if idx.get("sparse"):
                    opts["sparse"] = True
                try:
                    local_coll.create_index(key, name=idx_name, **opts)
                except Exception as e:
                    print(f"    Index '{idx_name}' skipped: {e}")
        except Exception:
            pass  # Some collections may not support list_indexes

    print(f"\nMigration complete: {total_docs:,} documents across {len(collections)} collections")

    # Verify counts
    print("\nVerification:")
    mismatches = 0
    for coll_name in sorted(collections):
        atlas_count = atlas_db[coll_name].count_documents({})
        local_count = local_db[coll_name].count_documents({})
        status = "OK" if atlas_count == local_count else "MISMATCH"
        if status == "MISMATCH":
            mismatches += 1
            print(f"  {coll_name}: Atlas={atlas_count:,} Local={local_count:,} [{status}]")

    if mismatches == 0:
        print(f"  All {len(collections)} collections match!")
    else:
        print(f"\n  {mismatches} collection(s) have mismatches")

    atlas_client.close()
    local_client.close()


if __name__ == "__main__":
    load_env()

    atlas_uri = (
        os.environ.get("ATLAS_URI")
        or os.environ.get("MONGODB_URI_ATLAS")
        or "mongodb+srv://connectivia_db_user:x4u2mpiSN3CDA1aD@omni.qwxleog.mongodb.net/marip_f1"
    )
    local_uri = (
        os.environ.get("LOCAL_URI")
        or os.environ.get("MONGODB_URI")
        or "mongodb://admin:maripf1admin@localhost:27017/marip_f1?authSource=admin"
    )
    db_name = os.environ.get("MONGODB_DB", "marip_f1")

    if "mongodb.net" not in atlas_uri:
        print("ERROR: ATLAS_URI doesn't look like an Atlas URI")
        sys.exit(1)

    migrate(atlas_uri, local_uri, db_name)
