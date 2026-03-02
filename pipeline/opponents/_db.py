"""MongoDB connection helper for the opponents module."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

DB_NAME = "marip_f1"

# Collection names
COL_PIT_LOSS = "circuit_pit_loss_times"
COL_PROFILES = "opponent_profiles"
COL_CIRCUIT = "opponent_circuit_profiles"
COL_COMPOUND = "opponent_compound_profiles"

_client: MongoClient | None = None


def get_db() -> Database:
    """Return the marip_f1 database, reusing a single client."""
    global _client
    if _client is None:
        uri = os.environ["MONGODB_URI"]
        _client = MongoClient(uri)
    return _client[DB_NAME]
