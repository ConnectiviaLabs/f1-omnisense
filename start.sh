#!/usr/bin/env bash
# F1 OmniSense — start all services
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Colors
O='\033[0;33m'  # Orange
G='\033[0;32m'  # Green
R='\033[0;31m'  # Red
C='\033[0m'     # Clear

echo -e "${O}════════════════════════════════════════════════════${C}"
echo -e "${O}  F1 OmniSense — Starting Services${C}"
echo -e "${O}════════════════════════════════════════════════════${C}"

# Kill any existing instances on our ports
for PORT in 8300 5173; do
  PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo -e "${O}  [~] Killing existing process on port $PORT${C}"
    echo "$PIDS" | xargs kill 2>/dev/null || true
    sleep 1
  fi
done

cleanup() {
  echo -e "\n${O}Shutting down...${C}"
  kill $API_PID $VITE_PID 2>/dev/null
  wait $API_PID $VITE_PID 2>/dev/null
  echo -e "${G}All services stopped.${C}"
}
trap cleanup EXIT INT TERM

# 1. Activate venv
if [ -d "fienv" ]; then
  source fienv/bin/activate
  echo -e "${G}  [✓] Python venv activated${C}"
else
  echo -e "${R}  [✗] fienv not found — run: python -m venv fienv${C}"
  exit 1
fi

# 2. Check required API keys
if [ -z "$NOMIC_API_KEY" ]; then
  # Try loading from .env
  if [ -f "$ROOT/.env" ]; then
    export $(grep -v '^#' "$ROOT/.env" | grep NOMIC_API_KEY | xargs 2>/dev/null)
  fi
fi
if [ -z "$NOMIC_API_KEY" ]; then
  echo -e "${R}  [✗] NOMIC_API_KEY not set — add it to .env${C}"
  exit 1
fi
echo -e "${G}  [✓] NOMIC_API_KEY set${C}"

# 3. Start API server (Knowledge Agent + 3D Model Gen on port 8300)
echo -e "\n${O}  Starting API server (port 8300)...${C}"
PYTHONPATH="$ROOT/omnisuitef1:$ROOT/pipeline:$ROOT" python -m uvicorn pipeline.chat_server:app --host 0.0.0.0 --port 8300 &
API_PID=$!

for i in $(seq 1 15); do
  if curl -s http://localhost:8300/health >/dev/null 2>&1; then
    DOCS=$(curl -s http://localhost:8300/health | python3 -c "import sys,json; print(json.load(sys.stdin)['documents'])" 2>/dev/null)
    echo -e "${G}  [✓] API server ready — ${DOCS} documents indexed${C}"
    break
  fi
  sleep 1
done

if ! curl -s http://localhost:8300/health >/dev/null 2>&1; then
  echo -e "${R}  [✗] API server failed to start${C}"
  exit 1
fi

# 4. Start Vite frontend
echo -e "\n${O}  Starting Frontend (Vite dev server)...${C}"
cd "$ROOT/frontend"
npx vite --host &
VITE_PID=$!
cd "$ROOT"

# Wait for Vite
for i in $(seq 1 15); do
  if curl -s http://localhost:5173 >/dev/null 2>&1; then
    echo -e "${G}  [✓] Frontend ready${C}"
    break
  fi
  sleep 1
done

echo -e "\n${O}════════════════════════════════════════════════════${C}"
echo -e "${G}  All services running:${C}"
echo -e "    Frontend:  ${O}http://localhost:5173${C}"
echo -e "    API:       ${O}http://localhost:8300${C}  (Knowledge Agent + 3D Gen)"
echo -e "${O}════════════════════════════════════════════════════${C}"
echo -e "  Press ${R}Ctrl+C${C} to stop all services"
echo ""

wait
