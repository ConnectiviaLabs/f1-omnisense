#!/usr/bin/env bash
# F1 OmniSense — RunPod startup (backend + frontend)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

O='\033[0;33m'
G='\033[0;32m'
R='\033[0;31m'
C='\033[0m'

echo -e "${O}════════════════════════════════════════════════════${C}"
echo -e "${O}  F1 OmniSense — RunPod Startup${C}"
echo -e "${O}════════════════════════════════════════════════════${C}"

# Kill existing processes on our ports
for PORT in 8300 8080; do
  PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo -e "${O}  [~] Killing existing process on port $PORT${C}"
    echo "$PIDS" | xargs kill 2>/dev/null || true
    sleep 1
  fi
done

cleanup() {
  echo -e "\n${O}Shutting down...${C}"
  kill $API_PID $FRONTEND_PID 2>/dev/null || true
  wait $API_PID $FRONTEND_PID 2>/dev/null || true
  echo -e "${G}All services stopped.${C}"
}
trap cleanup EXIT INT TERM

# 1. Start backend (port 8300)
echo -e "\n${O}  Starting backend (port 8300)...${C}"
PYTHONPATH="$ROOT:$ROOT/pipeline" python3 "$ROOT/pipeline/chat_server.py" &
API_PID=$!

for i in $(seq 1 30); do
  if curl -s http://localhost:8300/health >/dev/null 2>&1; then
    echo -e "${G}  [✓] Backend ready${C}"
    break
  fi
  sleep 2
done

if ! curl -s http://localhost:8300/health >/dev/null 2>&1; then
  echo -e "${R}  [✗] Backend failed to start${C}"
  exit 1
fi

# 2. Build frontend if needed
if [ ! -d "$ROOT/frontend/dist" ]; then
  echo -e "\n${O}  Building frontend...${C}"
  cd "$ROOT/frontend" && npm run build
  cd "$ROOT"
  echo -e "${G}  [✓] Frontend built${C}"
fi

# 3. Start frontend (port 8080, proxies /api to 8300)
echo -e "\n${O}  Starting frontend (port 8080)...${C}"
cd "$ROOT/frontend"
npx vite preview --host 0.0.0.0 --port 8080 &
FRONTEND_PID=$!
cd "$ROOT"

for i in $(seq 1 10); do
  if curl -s http://localhost:8080 >/dev/null 2>&1; then
    echo -e "${G}  [✓] Frontend ready${C}"
    break
  fi
  sleep 1
done

echo -e "\n${O}════════════════════════════════════════════════════${C}"
echo -e "${G}  All services running:${C}"
echo -e "    Frontend:  ${O}http://localhost:8080${C}"
echo -e "    Backend:   ${O}http://localhost:8300${C}"
echo -e "${O}════════════════════════════════════════════════════${C}"
echo -e "  Press ${R}Ctrl+C${C} to stop all services"
echo ""

wait
