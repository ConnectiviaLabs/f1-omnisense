# RunPod GPU Pod Deployment — F1 OmniSense

**Date:** 2026-03-13
**Status:** Approved
**Approach:** Enhanced script-based (single entrypoint)

---

## Overview

Deploy the F1 OmniSense application (FastAPI backend + React frontend) on a RunPod GPU Pod with local MongoDB, using a single entrypoint script that handles both first boot and restarts.

## RunPod Configuration

| Setting | Value |
|---------|-------|
| Container Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| GPU | NVIDIA L4 (24GB VRAM, Ada Lovelace) |
| Container Disk | 50GB (ephemeral) |
| Network Volume | `maripdata` — 100GB at `/workspace` (persistent) |
| Exposed HTTP Ports | 8300 (app), 8888 (default) |
| Exposed TCP Ports | 22 (SSH) |
| Start Command | `bash /workspace/marip-f1/scripts/runpod_entrypoint.sh` |

## Architecture

```
RunPod GPU Pod (L4)
│
├─ Port 8300 — FastAPI (API + frontend SPA)
│   ├─ /api/*          → Backend API routes (9 OmniSuite routers + extras)
│   └─ /*              → Frontend SPA (React, served from dist/)
│
├─ Port 27017 — MongoDB 7 (local, localhost only)
├─ Port 8081  — Mongo Express (background, DB admin UI)
└─ Port 22    — SSH access
```

### GPU vs API Split

**Runs on L4 GPU (local inference):**
- Sentence Transformers (Nomic/BGE embeddings, 768-dim vectors)
- YOLO (ultralytics — car/component detection)
- CLIP (image-text similarity via OmniVis)

**Runs via API (remote):**
- LLM reasoning — Groq `llama-3.3-70b-versatile` (primary)
- Quick model — Groq `llama-3.1-8b-instant`
- Fallbacks: OpenAI `gpt-4o-mini`, Anthropic `claude-3-haiku`

## Volume Layout

All persistent data lives on the network volume at `/workspace`:

```
/workspace/                              ← Network Volume (100GB, persistent)
├── marip-f1/                            ← Git repo
│   ├── frontend/dist/                   ← Built SPA
│   ├── pipeline/                        ← Backend code
│   ├── omnisuitef1/                     ← OmniSuite packages
│   └── .env                             ← Secrets (not in git)
├── mongodb_data/                        ← MongoDB data directory
│   └── .migration_complete              ← Atlas migration marker (already done)
├── huggingface_cache/                   ← HF model weights (~5-8GB)
├── venv/                                ← Python virtual environment
└── logs/                                ← App & MongoDB logs
```

### Persistence Map

| What | Location | Survives restart? |
|------|----------|-------------------|
| MongoDB data | `/workspace/mongodb_data` | Yes |
| HF model weights | `/workspace/huggingface_cache` | Yes |
| App code (git repo) | `/workspace/marip-f1` | Yes |
| Python venv + packages | `/workspace/venv` | Yes |
| .env secrets | `/workspace/marip-f1/.env` | Yes |
| Frontend build | `/workspace/marip-f1/frontend/dist` | Yes |
| Logs | `/workspace/logs` | Yes |

## Boot Sequence

### Unified Entrypoint: `scripts/runpod_entrypoint.sh`

Replaces the current two-script approach (`runpod_setup.sh` + `runpod_start.sh`).

```
Pod starts (runpod/pytorch image)
│
├─ Start Command: bash /workspace/marip-f1/scripts/runpod_entrypoint.sh
│
├─ FIRST BOOT ONLY (detects no /workspace/venv):
│   ├─ Install MongoDB 7 (apt)
│   ├─ Install Node.js 20 (nodesource — tested version)
│   ├─ Create /workspace/venv (inherits system PyTorch+CUDA)
│   ├─ pip install -r requirements.txt into venv
│   │   NOTE: Skip torch/torchvision in requirements — use system PyTorch 2.4+CUDA
│   ├─ npm install + npm run build (frontend)
│   ├─ mkdir -p /workspace/logs
│   └─ ~10-15 min one-time setup
│
├─ EVERY BOOT:
│   ├─ Activate /workspace/venv
│   ├─ Export HF_HOME=/workspace/huggingface_cache
│   ├─ Export PYTHONPATH (project root + omnisuitef1 + pipeline)
│   ├─ mkdir -p /workspace/logs
│   ├─ Start MongoDB (bind_ip 127.0.0.1) → /workspace/mongodb_data
│   │   mongod --dbpath /workspace/mongodb_data --bind_ip 127.0.0.1 \
│   │          --port 27017 --logpath /workspace/logs/mongod.log --fork
│   │   Fallback: if --fork fails, use nohup mongod ... &
│   ├─ Wait for MongoDB health check (20s timeout, exit 1 on fail)
│   ├─ Start mongo-express on 8081 (background)
│   ├─ Rebuild frontend if dist/ missing
│   └─ exec python3 /workspace/marip-f1/pipeline/chat_server.py
│       (chat_server.py reads API_PORT from .env, runs uvicorn internally)
│
└─ Ready — accessible via RunPod proxy on port 8300
```

### Key Design Decisions

1. **Single entrypoint** — one script set as RunPod "Start Command", handles all cases
2. **Idempotent** — safe to re-run; checks `command -v` and version before installing
3. **Fast restarts** — skips all installs on restart (~10s to boot)
4. **Venv on volume** — pip packages persist across restarts, no re-install. Created with `--system-site-packages` to inherit the image's PyTorch 2.4 + CUDA 12.4 (no need to reinstall torch)
5. **FastAPI runs in foreground** — `exec python3 pipeline/chat_server.py` replaces shell, so RunPod can track the process. `chat_server.py` calls `uvicorn.run()` internally, reading `API_PORT` from `.env`
6. **No containers-in-containers** — direct process management, simpler than Docker-in-pod
7. **No nginx** — FastAPI serves both API and static SPA on port 8300
8. **MongoDB binds to localhost only** — no auth needed since it's not network-accessible. RunPod pod network is isolated.
9. **First-boot detection** — uses `/workspace/venv` existence. If setup fails partway, delete `/workspace/venv` and restart to retry from scratch.
10. **Node.js 20** — matches the tested version from existing `runpod_setup.sh`

## Environment Variables

Stored in `/workspace/marip-f1/.env`:

```env
# MongoDB (local)
MONGODB_URI=mongodb://localhost:27017/marip_f1
MONGODB_DB=marip_f1

# Server
API_PORT=8300

# LLM Providers
GROQ_API_KEY=<secret>
GROQ_REASONING_MODEL=llama-3.3-70b-versatile
GROQ_QUICK_MODEL=llama-3.1-8b-instant
PRIMARY_LLM_PROVIDER=groq
OPENAI_API_KEY=<secret>
ANTHROPIC_API_KEY=<secret>

# 3D Generation
MESHY_API_KEY=<secret>
TRIPO_API_KEY=<secret>
FAL_KEY=<secret>

# HuggingFace
HF_TOKEN=<secret>

# Other
ROBOFLOW=<secret>
DEBUG=false
LOG_LEVEL=INFO
```

## What Changes

### New File
- `scripts/runpod_entrypoint.sh` — unified boot script

### Modified Files
- None required. Existing `chat_server.py` already serves frontend SPA on the same port.

### Deprecated (kept for reference)
- `runpod_start.sh` — replaced by entrypoint
- `scripts/runpod_setup.sh` — replaced by entrypoint

## Deployment Steps (Manual, One-Time)

1. Create RunPod GPU Pod with L4, the pytorch image, and `maripdata` volume
2. SSH into pod
3. Clone repo to `/workspace/marip-f1` (if not already there)
4. Copy `.env` with real secrets to `/workspace/marip-f1/.env`
5. Set RunPod Start Command to `bash /workspace/marip-f1/scripts/runpod_entrypoint.sh`
6. Add port 8300 to "Expose HTTP Ports"
7. Restart pod — entrypoint handles everything

## Updating the App

To deploy new code:
```bash
cd /workspace/marip-f1
git pull
cd frontend && npm run build && cd ..
# Restart FastAPI (or just restart the pod)
```

## Monitoring

- **Logs:** `/workspace/logs/mongod.log`, `/workspace/logs/mongo-express.log`
- **FastAPI:** stdout (visible in RunPod logs)
- **Health check:** `curl http://localhost:8300/health`
- **MongoDB:** `mongosh --eval "db.runCommand({ping:1})"`
- **Mongo Express:** port 8081 via RunPod proxy (if exposed)
