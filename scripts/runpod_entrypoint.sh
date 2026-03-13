#!/usr/bin/env bash
# F1 OmniSense — RunPod unified entrypoint
# Handles both first boot (installs deps) and restarts (fast path).
# Set as RunPod "Start Command": bash /workspace/marip-f1/scripts/runpod_entrypoint.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────
O='\033[0;33m'  # Orange
G='\033[0;32m'  # Green
R='\033[0;31m'  # Red
C='\033[0m'     # Reset

PORT="${API_PORT:-8300}"
VOLUME="/workspace"
ROOT="$VOLUME/marip-f1"
VENV="$VOLUME/venv"
MONGO_DATA="$VOLUME/mongodb_data"
LOG_DIR="$VOLUME/logs"
HF_CACHE="$VOLUME/huggingface_cache"

echo -e "${O}════════════════════════════════════════════════════${C}"
echo -e "${O}  F1 OmniSense — RunPod Entrypoint (port $PORT)${C}"
echo -e "${O}════════════════════════════════════════════════════${C}"

# ── Validate volume ───────────────────────────────────────────────
if [ ! -d "$VOLUME" ]; then
    echo -e "${R}  [✗] Network volume not found at $VOLUME${C}"
    echo -e "${R}      Attach a network volume and mount at /workspace${C}"
    exit 1
fi

if [ ! -d "$ROOT" ]; then
    echo -e "${R}  [✗] Repo not found at $ROOT${C}"
    echo -e "${R}      Clone the repo: git clone <url> $ROOT${C}"
    exit 1
fi

cd "$ROOT"

# Create persistent directories
mkdir -p "$MONGO_DATA" "$LOG_DIR" "$HF_CACHE"

# ══════════════════════════════════════════════════════════════════
# FIRST BOOT — install system deps and create venv
# ══════════════════════════════════════════════════════════════════
if [ ! -d "$VENV" ]; then
    echo -e "\n${O}  ── First Boot Detected ──${C}"

    # ── Install MongoDB 7 ─────────────────────────────────────────
    if ! command -v mongod &>/dev/null; then
        echo -e "${O}  [1/4] Installing MongoDB 7...${C}"
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
        echo -e "${G}  [✓] MongoDB $(mongod --version | head -1)${C}"
    else
        echo -e "${G}  [✓] MongoDB already installed${C}"
    fi

    # ── Install Node.js 20 ────────────────────────────────────────
    if ! command -v node &>/dev/null; then
        echo -e "${O}  [2/4] Installing Node.js 20...${C}"
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
        apt-get install -y -qq nodejs > /dev/null
        echo -e "${G}  [✓] Node $(node --version)${C}"
    else
        echo -e "${G}  [✓] Node already installed: $(node --version)${C}"
    fi

    # ── Create venv + install pip packages ────────────────────────
    echo -e "${O}  [3/4] Creating Python venv + installing packages...${C}"
    python3 -m venv --system-site-packages "$VENV"
    source "$VENV/bin/activate"

    pip install --upgrade pip -q

    # Fix known conflicts
    pip uninstall fitz -y 2>/dev/null || true
    pip install PyMuPDF -q 2>/dev/null || true

    # Install requirements (torch is excluded — using system PyTorch+CUDA)
    pip install -r "$ROOT/pipeline/requirements.txt" -q 2>&1 | tail -5
    pip install pymongo python-dotenv groq pdfplumber python-docx -q 2>/dev/null || true
    echo -e "${G}  [✓] Python packages installed${C}"

    # ── Build frontend ────────────────────────────────────────────
    echo -e "${O}  [4/4] Building frontend...${C}"
    cd "$ROOT/frontend"
    npm install --silent 2>&1 | tail -3
    npm run build
    cd "$ROOT"
    echo -e "${G}  [✓] Frontend built${C}"

    # ── Install mongo-express globally ────────────────────────────
    if ! npm list -g mongo-express &>/dev/null 2>&1; then
        npm install -g mongo-express --silent 2>/dev/null || true
    fi

    echo -e "\n${G}  ── First Boot Complete ──${C}"
fi

# ══════════════════════════════════════════════════════════════════
# EVERY BOOT — activate venv, start services, launch app
# ══════════════════════════════════════════════════════════════════
echo -e "\n${O}  ── Starting Services ──${C}"

# ── Activate venv ─────────────────────────────────────────────────
source "$VENV/bin/activate"

# ── Environment variables ─────────────────────────────────────────
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export PYTHONPATH="$ROOT:$ROOT/omnisuitef1:$ROOT/pipeline"
export API_PORT="$PORT"

# ── Start MongoDB ─────────────────────────────────────────────────
if command -v mongod &>/dev/null; then
    if ! pgrep -x mongod &>/dev/null; then
        echo -e "${O}  Starting MongoDB (data: $MONGO_DATA)...${C}"

        # Try --fork first, fall back to nohup
        if ! mongod --dbpath "$MONGO_DATA" --bind_ip 127.0.0.1 --port 27017 \
                    --fork --logpath "$LOG_DIR/mongod.log" 2>/dev/null; then
            echo -e "${O}  --fork failed, using nohup fallback...${C}"
            nohup mongod --dbpath "$MONGO_DATA" --bind_ip 127.0.0.1 --port 27017 \
                         --logpath "$LOG_DIR/mongod.log" &>/dev/null &
        fi

        # Wait for ready
        for i in $(seq 1 20); do
            if mongosh --quiet --eval "db.runCommand({ping:1})" &>/dev/null; then
                echo -e "${G}  [✓] MongoDB ready${C}"
                break
            fi
            if [ "$i" -eq 20 ]; then
                echo -e "${R}  [✗] MongoDB failed to start after 20s${C}"
                echo -e "${R}      Check: $LOG_DIR/mongod.log${C}"
                exit 1
            fi
            sleep 1
        done
    else
        echo -e "${G}  [✓] MongoDB already running${C}"
    fi
else
    echo -e "${R}  [✗] mongod not found — run first boot or install manually${C}"
    exit 1
fi

# ── Start mongo-express (background) ─────────────────────────────
pkill -f "mongo-express" 2>/dev/null || true
sleep 1
ME_CONFIG_MONGODB_URL="mongodb://localhost:27017/" \
ME_CONFIG_BASICAUTH="false" \
ME_CONFIG_SITE_BASEURL="/" \
    npx mongo-express &>"$LOG_DIR/mongo-express.log" &
echo -e "${G}  [✓] mongo-express on port 8081 (PID: $!)${C}"

# ── Kill existing process on our port ─────────────────────────────
PIDS=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    echo -e "${O}  [~] Killing existing process on port $PORT${C}"
    echo "$PIDS" | xargs kill 2>/dev/null || true
    sleep 1
fi

# ── Build frontend if dist missing ────────────────────────────────
if [ ! -d "$ROOT/frontend/dist" ]; then
    echo -e "${O}  Building frontend...${C}"
    cd "$ROOT/frontend" && npm run build
    cd "$ROOT"
    echo -e "${G}  [✓] Frontend built${C}"
else
    echo -e "${G}  [✓] Frontend dist/ exists${C}"
fi

# ── Launch FastAPI (foreground — RunPod tracks this process) ──────
echo -e "\n${O}════════════════════════════════════════════════════${C}"
echo -e "${G}  F1 OmniSense starting on port $PORT${C}"
echo -e "${G}  MongoDB:       localhost:27017${C}"
echo -e "${G}  Mongo Express: port 8081${C}"
echo -e "${G}  Logs:          $LOG_DIR/${C}"
echo -e "${O}════════════════════════════════════════════════════${C}"

exec python3 "$ROOT/pipeline/chat_server.py"
