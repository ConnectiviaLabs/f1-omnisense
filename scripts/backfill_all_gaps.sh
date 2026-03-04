#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════
#  F1 OmniSense — Backfill All Data Gaps
#  Run on RunPod after initial setup to fill missing 2025 data
# ══════════════════════════════════════════════════════════════

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/pipeline:$ROOT/omnisuitef1"
export MONGODB_URI="${MONGODB_URI:-mongodb://localhost:27017/marip_f1}"
export MONGODB_DB="${MONGODB_DB:-marip_f1}"

O='\033[0;33m'
G='\033[0;32m'
R='\033[0;31m'
C='\033[0m'

echo -e "${O}══════════════════════════════════════════════════════${C}"
echo -e "${O}  F1 OmniSense — Backfill All Data Gaps${C}"
echo -e "${O}══════════════════════════════════════════════════════${C}"
echo -e "  MongoDB: $MONGODB_URI"
echo ""

# ── 0. Install deps if needed ─────────────────────────────────
echo -e "${O}[0/6] Checking dependencies...${C}"
pip install -q fastf1 pymongo[srv] dnspython pandas numpy python-dotenv 2>/dev/null
echo -e "${G}  [done]${C}"
echo ""

# ── 1. Compress 2025 telemetry (if not already done) ──────────
echo -e "${O}[1/6] Checking 2025 telemetry_compressed...${C}"
CHUNKS_2025=$(python3 -c "
from pymongo import MongoClient; import os
db = MongoClient(os.environ['MONGODB_URI'])['marip_f1']
print(db['telemetry_compressed'].count_documents({'filename': '2025_R.parquet'}))
")
if [ "$CHUNKS_2025" -gt 0 ]; then
    echo -e "${G}  Already done: $CHUNKS_2025 chunks exist. Skipping.${C}"
else
    echo -e "  Fetching 2025 FastF1 telemetry (~30-60 min)..."
    python3 scripts/compress_telemetry_2025.py
    echo -e "${G}  [done]${C}"
fi
echo ""

# ── 2. Build telemetry_lap_summary + telemetry_race_summary ──
echo -e "${O}[2/6] Building telemetry summaries (lap + race)...${C}"
python3 -m pipeline.enrichment.build_telemetry_summaries --year 2025
echo -e "${G}  [done]${C}"
echo ""

# ── 3. Sync OpenF1 data (backfill 2025 gaps) ─────────────────
echo -e "${O}[3/6] Syncing OpenF1 data for 2025...${C}"
python3 -c "
from pipeline.updater._db import get_db
from pipeline.updater import openf1_fetcher
db = get_db()
result = openf1_fetcher.sync(db, 2025, full_refresh=False)
print(f'  OpenF1 sync: {result}')
"
echo -e "${G}  [done]${C}"
echo ""

# ── 4. Sync FastF1 laps + weather for 2025 ───────────────────
echo -e "${O}[4/6] Fetching FastF1 laps & weather for 2025...${C}"
python3 -c "
from pipeline.updater._db import get_db
from pipeline.updater import fastf1_fetcher
db = get_db()
result = fastf1_fetcher.sync(db, 2025)
print(f'  FastF1 sync: {result}')
"
echo -e "${G}  [done]${C}"
echo ""

# ── 5. Sync Jolpica historical data ──────────────────────────
echo -e "${O}[5/6] Syncing Jolpica data (2018-2025)...${C}"
python3 -c "
from pipeline.updater._db import get_db
from pipeline.updater import jolpica_fetcher
db = get_db()
result = jolpica_fetcher.sync(db, years=list(range(2018, 2026)))
print(f'  Jolpica sync: {result}')
"
echo -e "${G}  [done]${C}"
echo ""

# ── 6. Refresh derived profiles ──────────────────────────────
echo -e "${O}[6/6] Refreshing driver & opponent profiles...${C}"
python3 -c "
from pipeline.updater._db import get_db
from pipeline.updater import profile_refresher
db = get_db()
result = profile_refresher.refresh(db)
print(f'  Profiles: {result}')
" 2>&1 || echo -e "${R}  Profile refresh had errors (non-fatal)${C}"
echo ""

# ── Verify ────────────────────────────────────────────────────
echo -e "${O}══════════════════════════════════════════════════════${C}"
echo -e "${O}  VERIFICATION${C}"
echo -e "${O}══════════════════════════════════════════════════════${C}"
python3 -c "
from pymongo import MongoClient; import os
db = MongoClient(os.environ['MONGODB_URI'])['marip_f1']

checks = [
    ('telemetry_compressed (2025)', {'filename': '2025_R.parquet'}),
    ('telemetry_race_summary (2025)', {'Year': 2025}),
    ('telemetry_lap_summary (2025)', {'Year': 2025}),
    ('fastf1_laps (2025)', {'Year': 2025}),
    ('fastf1_weather (2025)', {'Year': 2025}),
    ('jolpica_race_results (2025)', {'season': 2025}),
    ('openf1_sessions (2025)', {}),
    ('opponent_profiles', {}),
    ('driver_telemetry_profiles', {}),
]

for label, filt in checks:
    coll_name = label.split(' (')[0]
    count = db[coll_name].count_documents(filt)
    status = 'OK' if count > 0 else 'MISSING'
    print(f'  [{status:7s}] {label}: {count:,} docs')

print()
total = sum(db[c].estimated_document_count() for c in db.list_collection_names())
print(f'  Total documents: {total:,}')
"
echo ""
echo -e "${G}  Backfill complete! Restart the server to pick up new data.${C}"
echo ""
