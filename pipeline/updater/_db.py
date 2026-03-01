"""MongoDB connection helper for the updater module."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

DB_NAME = "marip_f1"

_client: MongoClient | None = None


def get_db() -> Database:
    global _client
    if _client is None:
        uri = os.environ.get(
            "MONGODB_URI",
            "mongodb+srv://connectivia_db_user:Vq7agrxoA5vRxzmO@omni.qwxleog.mongodb.net/",
        )
        _client = MongoClient(uri)
    return _client[DB_NAME]
