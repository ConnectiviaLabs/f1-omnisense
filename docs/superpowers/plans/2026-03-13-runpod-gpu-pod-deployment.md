# RunPod GPU Pod Deployment — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a unified entrypoint script that deploys the F1 OmniSense app (FastAPI + React) on a RunPod GPU Pod with local MongoDB, handling both first boot and restarts.

**Architecture:** Single bash entrypoint script on the network volume. First boot installs MongoDB 7, Node.js 20, and pip packages into a persistent venv. Every boot starts MongoDB, mongo-express, and FastAPI (which serves the API + SPA on port 8300).

**Tech Stack:** Bash, MongoDB 7, Node.js 20, Python 3.11 (RunPod image), FastAPI, Vite/React

**Spec:** `docs/superpowers/specs/2026-03-13-runpod-gpu-pod-deployment-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `scripts/runpod_entrypoint.sh` | Unified boot script — first boot setup + every-boot startup |
| Keep | `runpod_start.sh` | Deprecated, kept for reference |
| Keep | `scripts/runpod_setup.sh` | Deprecated, kept for reference |
| None | `pipeline/chat_server.py` | No changes needed — already serves SPA + API on configurable port |

---

## Chunk 1: Unified Entrypoint Script

### Task 1: Create the entrypoint script — header and config

**Files:**
- Create: `scripts/runpod_entrypoint.sh`

- [ ] **Step 1: Create the script with header, color codes, and volume detection**

```bash
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
```

- [ ] **Step 2: Save file, make executable, verify it parses**

Run: `chmod +x scripts/runpod_entrypoint.sh && bash -n scripts/runpod_entrypoint.sh`
Expected: No output (syntax OK)

- [ ] **Step 3: Commit**

```bash
git add scripts/runpod_entrypoint.sh
git commit -m "feat: add runpod entrypoint script — header and config"
```

---

### Task 2: Add first-boot installation logic

**Files:**
- Modify: `scripts/runpod_entrypoint.sh`

- [ ] **Step 1: Add the first-boot detection and MongoDB install section**

Append after the `mkdir -p` line:

```bash
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
```

- [ ] **Step 2: Verify syntax**

Run: `bash -n scripts/runpod_entrypoint.sh`
Expected: No output (syntax OK)

- [ ] **Step 3: Commit**

```bash
git add scripts/runpod_entrypoint.sh
git commit -m "feat: add first-boot install logic to entrypoint"
```

---

### Task 3: Add every-boot startup logic

**Files:**
- Modify: `scripts/runpod_entrypoint.sh`

- [ ] **Step 1: Add the every-boot section — venv activation, MongoDB start, FastAPI launch**

Append after the first-boot `fi`:

```bash
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
```

- [ ] **Step 2: Verify syntax**

Run: `bash -n scripts/runpod_entrypoint.sh`
Expected: No output (syntax OK)

- [ ] **Step 3: Commit**

```bash
git add scripts/runpod_entrypoint.sh
git commit -m "feat: add every-boot startup logic to entrypoint"
```

---

## Chunk 2: Local Testing & Deployment

### Task 4: Test the entrypoint script locally (dry run)

**Files:**
- Read only: `scripts/runpod_entrypoint.sh`

- [ ] **Step 1: Verify the full script parses without errors**

Run: `bash -n scripts/runpod_entrypoint.sh`
Expected: No output (clean parse)

- [ ] **Step 2: Review the final script end-to-end**

Run: `cat -n scripts/runpod_entrypoint.sh | head -200`
Expected: Complete script with header, first-boot, and every-boot sections

- [ ] **Step 3: Verify no references to bind_ip_all remain**

Run: `grep -r "bind_ip_all" scripts/`
Expected: No matches (only `bind_ip 127.0.0.1` should appear)

- [ ] **Step 4: Verify no hardcoded secrets in the script**

Run: `grep -iE "(api_key|password|secret|token)=" scripts/runpod_entrypoint.sh`
Expected: No matches (all secrets come from .env)

---

### Task 5: Push to remote and deploy on RunPod

**Files:**
- None modified

- [ ] **Step 1: Push the branch to remote**

Run: `git push origin mongodb-migration`
Expected: Branch pushed successfully

- [ ] **Step 2: SSH into RunPod pod**

Run: `ssh root@<runpod-ip>` (or use RunPod web terminal)

- [ ] **Step 3: Pull latest code on the pod**

```bash
cd /workspace/marip-f1
git pull origin mongodb-migration
```

- [ ] **Step 4: Verify .env exists with correct MongoDB URI**

```bash
grep "^MONGODB_URI" /workspace/marip-f1/.env
```
Expected: `MONGODB_URI=mongodb://localhost:27017/marip_f1`

- [ ] **Step 5: Set RunPod Start Command**

In RunPod UI → Pod Settings → Start Command:
```
bash /workspace/marip-f1/scripts/runpod_entrypoint.sh
```

- [ ] **Step 6: Add port 8300 to Expose HTTP Ports**

In RunPod UI → Pod Template Overrides → Expose HTTP Ports:
Add `8300` (keep existing `8888`)

- [ ] **Step 7: Restart the pod and verify**

Restart pod, then check logs. Once running, test:
```bash
# Health check
curl http://localhost:8300/health

# MongoDB check
mongosh --eval "db.runCommand({ping:1})"

# Frontend check (should return HTML)
curl -s http://localhost:8300/ | head -5
```
Expected: Health endpoint returns JSON, MongoDB pings, frontend returns HTML

---

### Task 6: Final verification and cleanup

- [ ] **Step 1: Verify the app is accessible via RunPod proxy URL**

Open in browser: `https://<pod-id>-8300.proxy.runpod.net/`
Expected: F1 OmniSense frontend loads

- [ ] **Step 2: Test API endpoint through proxy**

Open: `https://<pod-id>-8300.proxy.runpod.net/health`
Expected: JSON health response

- [ ] **Step 3: Test a full restart cycle**

Stop the pod, start it again. Verify:
- Boot time < 30 seconds (fast path, no installs)
- MongoDB data persists
- Frontend loads
- API responds

- [ ] **Step 4: Commit any final adjustments**

```bash
git add -A
git commit -m "fix: finalize runpod entrypoint after deployment testing"
git push origin mongodb-migration
```
