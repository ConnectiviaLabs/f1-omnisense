#!/usr/bin/env bash
# F1 OmniSense — RunPod startup (backend serves frontend too)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

O='\033[0;33m'
G='\033[0;32m'
R='\033[0;31m'
C='\033[0m'

# Use API_PORT env var, default to 8300
PORT="${API_PORT:-8300}"

echo -e "${O}════════════════════════════════════════════════════${C}"
echo -e "${O}  F1 OmniSense — RunPod Startup (port $PORT)${C}"
echo -e "${O}════════════════════════════════════════════════════${C}"

# ── Ensure MongoDB is running ────────────────────────────────
if command -v mongod &>/dev/null; then
  if ! pgrep -x mongod &>/dev/null; then
    MONGO_DATA="/workspace/mongodb_data"
    mkdir -p "$MONGO_DATA"
    echo -e "${O}  Starting MongoDB (data: $MONGO_DATA)...${C}"
    mongod --dbpath "$MONGO_DATA" --bind_ip_all --port 27017 --fork --logpath /var/log/mongod.log
    # Wait for ready
    for i in $(seq 1 15); do
      if mongosh --quiet --eval "db.runCommand({ping:1})" &>/dev/null; then
        echo -e "${G}  [✓] MongoDB ready${C}"
        break
      fi
      [ "$i" -eq 15 ] && echo -e "${R}  [✗] MongoDB failed to start${C}"
      sleep 1
    done
  else
    echo -e "${G}  [✓] MongoDB already running${C}"
  fi
fi

# Kill existing process on our port
PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo -e "${O}  [~] Killing existing process on port $PORT${C}"
  echo "$PIDS" | xargs kill 2>/dev/null || true
  sleep 1
fi

# Build frontend if dist doesn't exist
if [ ! -d "$ROOT/frontend/dist" ]; then
  echo -e "\n${O}  Building frontend...${C}"
  cd "$ROOT/frontend" && npm run build
  cd "$ROOT"
  echo -e "${G}  [✓] Frontend built${C}"
else
  echo -e "${G}  [✓] Frontend dist/ exists${C}"
fi

# Start backend (serves API + frontend SPA on same port)
echo -e "\n${O}  Starting server (port $PORT)...${C}"
export API_PORT="$PORT"
export PYTHONPATH="$ROOT:$ROOT/omnisuitef1:$ROOT/pipeline"
exec python3 "$ROOT/pipeline/chat_server.py"
