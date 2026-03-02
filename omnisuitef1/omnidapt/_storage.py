"""Storage backends — LocalStorage (JSON) and MongoStorage (pymongo + GridFS).

LocalStorage works without any external dependencies.
MongoStorage is optional, activated via OMNIDAPT_USE_MONGO=true.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Protocol ─────────────────────────────────────────────────────────────

@runtime_checkable
class StorageBackend(Protocol):
    """Abstract storage interface for omnidapt."""

    def save_document(self, collection: str, document: Dict[str, Any]) -> str:
        """Save a document, return its ID."""
        ...

    def find_documents(
        self, collection: str, query: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Find documents matching query."""
        ...

    def find_one(
        self, collection: str, query: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Find a single document."""
        ...

    def update_document(
        self, collection: str, query: Dict[str, Any], update: Dict[str, Any],
    ) -> bool:
        """Update matching document. Returns True if found."""
        ...

    def delete_document(self, collection: str, query: Dict[str, Any]) -> bool:
        """Delete matching document. Returns True if found."""
        ...

    def count_documents(
        self, collection: str, query: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count matching documents."""
        ...

    def save_artifact(self, name: str, data: bytes) -> str:
        """Save binary artifact, return its path/ID."""
        ...

    def load_artifact(self, name: str) -> Optional[bytes]:
        """Load binary artifact by name."""
        ...


# ── Local Storage ────────────────────────────────────────────────────────

class LocalStorage:
    """JSON-file based storage. Each collection is a JSON file."""

    def __init__(self, base_dir: Path):
        self._base = base_dir
        self._collections_dir = base_dir / "collections"
        self._artifacts_dir = base_dir / "artifacts"
        self._collections_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _collection_path(self, collection: str) -> Path:
        return self._collections_dir / f"{collection}.json"

    def _load_collection(self, collection: str) -> List[Dict[str, Any]]:
        path = self._collection_path(collection)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt collection %s, resetting", collection)
            return []

    def _save_collection(self, collection: str, docs: List[Dict[str, Any]]) -> None:
        path = self._collection_path(collection)
        path.write_text(json.dumps(docs, indent=2, default=str))

    @staticmethod
    def _matches(doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Simple key-value match."""
        for k, v in query.items():
            if doc.get(k) != v:
                return False
        return True

    def save_document(self, collection: str, document: Dict[str, Any]) -> str:
        with self._lock:
            docs = self._load_collection(collection)
            doc_id = document.get("_id") or f"{int(time.time() * 1000)}_{len(docs)}"
            document["_id"] = doc_id
            docs.append(document)
            self._save_collection(collection, docs)
        return doc_id

    def find_documents(
        self, collection: str, query: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        docs = self._load_collection(collection)
        if not query:
            return docs
        return [d for d in docs if self._matches(d, query)]

    def find_one(
        self, collection: str, query: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        docs = self._load_collection(collection)
        for d in docs:
            if self._matches(d, query):
                return d
        return None

    def update_document(
        self, collection: str, query: Dict[str, Any], update: Dict[str, Any],
    ) -> bool:
        with self._lock:
            docs = self._load_collection(collection)
            for d in docs:
                if self._matches(d, query):
                    d.update(update)
                    self._save_collection(collection, docs)
                    return True
        return False

    def delete_document(self, collection: str, query: Dict[str, Any]) -> bool:
        with self._lock:
            docs = self._load_collection(collection)
            original_len = len(docs)
            docs = [d for d in docs if not self._matches(d, query)]
            if len(docs) < original_len:
                self._save_collection(collection, docs)
                return True
        return False

    def count_documents(
        self, collection: str, query: Optional[Dict[str, Any]] = None,
    ) -> int:
        return len(self.find_documents(collection, query))

    def save_artifact(self, name: str, data: bytes) -> str:
        path = self._artifacts_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def load_artifact(self, name: str) -> Optional[bytes]:
        path = self._artifacts_dir / name
        if path.exists():
            return path.read_bytes()
        return None


# ── Mongo Storage ────────────────────────────────────────────────────────

class MongoStorage:
    """MongoDB + GridFS storage backend (optional)."""

    def __init__(self, mongo_uri: str | None = None, db_name: str = "omnidapt"):
        import os
        uri = mongo_uri or os.environ.get("OMNIDAPT_MONGO_URI") or os.environ.get("MONGO_URI", "")
        if not uri:
            raise ValueError("No MongoDB URI. Set OMNIDAPT_MONGO_URI or MONGO_URI.")

        import pymongo
        import gridfs
        self._client = pymongo.MongoClient(uri)
        self._db = self._client[db_name]
        self._fs = gridfs.GridFS(self._db)
        logger.info("MongoStorage connected to %s/%s", uri.split("@")[-1], db_name)

    def save_document(self, collection: str, document: Dict[str, Any]) -> str:
        result = self._db[collection].insert_one(document)
        return str(result.inserted_id)

    def find_documents(
        self, collection: str, query: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        cursor = self._db[collection].find(query or {})
        docs = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            docs.append(doc)
        return docs

    def find_one(
        self, collection: str, query: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        doc = self._db[collection].find_one(query)
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    def update_document(
        self, collection: str, query: Dict[str, Any], update: Dict[str, Any],
    ) -> bool:
        result = self._db[collection].update_one(query, {"$set": update})
        return result.modified_count > 0

    def delete_document(self, collection: str, query: Dict[str, Any]) -> bool:
        result = self._db[collection].delete_one(query)
        return result.deleted_count > 0

    def count_documents(
        self, collection: str, query: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self._db[collection].count_documents(query or {})

    def save_artifact(self, name: str, data: bytes) -> str:
        # Delete existing with same name
        existing = self._fs.find_one({"filename": name})
        if existing:
            self._fs.delete(existing._id)
        file_id = self._fs.put(data, filename=name)
        return str(file_id)

    def load_artifact(self, name: str) -> Optional[bytes]:
        f = self._fs.find_one({"filename": name})
        if f:
            return f.read()
        return None


# ── Factory ──────────────────────────────────────────────────────────────

_storage: StorageBackend | None = None


def get_storage() -> StorageBackend:
    """Get or create the global storage backend."""
    global _storage
    if _storage is None:
        from omnidapt._config import get_config
        cfg = get_config()
        if cfg.use_mongo:
            _storage = MongoStorage()
        else:
            _storage = LocalStorage(cfg.base_dir)
    return _storage


def reset_storage() -> None:
    """Reset storage singleton (for testing)."""
    global _storage
    _storage = None
