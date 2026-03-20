#!/usr/bin/env bash
set -euo pipefail

# RunPod FULL setup script for F1 OmniSense
# Installs: Python deps, MongoDB 7, mongo-express, migrates Atlas data, starts app.
# Run once on a fresh RunPod pod. Idempotent — safe to re-run.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "════════════════════════════════════════════════════"
echo "  F1 OmniSense — RunPod Full Setup"
echo "════════════════════════════════════════════════════"

# ── 0. Detect volume mount ──────────────────────────────────────
VOLUME_PATH=""
for p in /workspace /runpod-volume; do
    [ -d "$p" ] && VOLUME_PATH="$p" && break
done
if [ -z "$VOLUME_PATH" ]; then
    echo "WARNING: No RunPod volume found. Data will NOT persist."
    VOLUME_PATH="/tmp"
fi
echo "  Volume: $VOLUME_PATH"
MONGO_DATA="$VOLUME_PATH/mongodb_data"
mkdir -p "$MONGO_DATA"

# ── 1. Install Python dependencies ─────────────────────────────
echo ""
echo "[1/5] Installing Python dependencies..."
pip install --upgrade pip -q

# Fix known conflicts first
pip uninstall fitz -y 2>/dev/null || true
pip install PyMuPDF -q 2>/dev/null || true

# Install from requirements
if [ -f "$ROOT/pipeline/requirements.txt" ]; then
    pip install -r "$ROOT/pipeline/requirements.txt" -q 2>&1 | tail -3
    echo "  [done] pipeline/requirements.txt"
fi

# Extras not in requirements.txt
pip install pymongo python-dotenv groq pdfplumber python-docx -q 2>/dev/null || true
echo "  [done] extra deps"

# ── 2. Install MongoDB 7 ───────────────────────────────────────
echo ""
echo "[2/5] Setting up MongoDB..."
if ! command -v mongod &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq gnupg curl > /dev/null

    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
        gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg 2>/dev/null

    . /etc/os-release
    if [[ "$ID" == "ubuntu" ]]; then
        echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu ${VERSION_CODENAME}/mongodb-org/7.0 multiverse" \
            > /etc/apt/sources.list.d/mongodb-org-7.0.list
    else
        echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main" \
            > /etc/apt/sources.list.d/mongodb-org-7.0.list
    fi

    apt-get update -qq
    apt-get install -y -qq mongodb-org > /dev/null
    echo "  Installed: $(mongod --version | head -1)"
else
    echo "  Already installed: $(mongod --version | head -1)"
fi

# ── 3. Start MongoDB ───────────────────────────────────────────
echo ""
echo "[3/5] Starting MongoDB..."
if pgrep -x mongod &>/dev/null; then
    echo "  Already running"
else
    mongod --dbpath "$MONGO_DATA" --bind_ip_all --port 27017 --fork --logpath /var/log/mongod.log

    for i in $(seq 1 20); do
        if mongosh --quiet --eval "db.runCommand({ping:1})" &>/dev/null; then
            echo "  MongoDB is ready!"
            break
        fi
        [ "$i" -eq 20 ] && { echo "ERROR: MongoDB did not start"; exit 1; }
        sleep 1
    done
fi

# ── 4. Install & start mongo-express ───────────────────────────
echo ""
echo "[4/5] Setting up mongo-express..."
if ! command -v npx &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
    apt-get install -y -qq nodejs > /dev/null
fi

if ! npm list -g mongo-express &>/dev/null 2>&1; then
    npm install -g mongo-express --silent 2>/dev/null
fi

pkill -f "mongo-express" 2>/dev/null || true
sleep 1

ME_CONFIG_MONGODB_URL="mongodb://localhost:27017/" \
ME_CONFIG_BASICAUTH="false" \
ME_CONFIG_SITE_BASEURL="/" \
    npx mongo-express &>/var/log/mongo-express.log &
echo "  mongo-express started on port 8081 (PID: $!)"

# ── 5. Migrate Atlas → local (first time only) ────────────────
echo ""
echo "[5/5] Data migration..."
MARKER="$MONGO_DATA/.migration_complete"
if [ ! -f "$MARKER" ]; then
    echo "  Migrating all data from Atlas to local MongoDB..."

    LOCAL_URI="mongodb://localhost:27017/marip_f1" \
        python3 "$ROOT/scripts/migrate_atlas_to_local.py"

    touch "$MARKER"
    echo "  Migration complete. Marker set."
else
    echo "  Already migrated (marker found). Skipping."
fi

# ── 6. Update .env to use local MongoDB ────────────────────────
echo ""
if grep -q "^MONGODB_URI=mongodb://localhost" "$ROOT/.env" 2>/dev/null; then
    echo ".env already points to localhost"
else
    echo "Updating .env to use local MongoDB..."
    sed -i 's|^MONGODB_URI=.*|MONGODB_URI=mongodb://localhost:27017/marip_f1|' "$ROOT/.env"
    echo "  Done"
fi

# ── 7. Build frontend if needed ────────────────────────────────
if [ ! -d "$ROOT/frontend/dist" ]; then
    echo ""
    echo "Building frontend..."
    cd "$ROOT/frontend" && npm install --silent && npm run build
    cd "$ROOT"
fi

# ── Done ───────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  Setup complete!"
echo "════════════════════════════════════════════════════"
echo "  MongoDB:       localhost:27017 (data: $MONGO_DATA)"
echo "  Mongo Express: port 8081"
echo ""
echo "  Start the app:"
echo "    cd $ROOT && ./runpod_start.sh"
echo ""
