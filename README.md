<p align="center">
  <img src="https://img.shields.io/badge/F1-OmniSense-ff6600?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJMMyAyMGgyMEwxMiAyeiIgZmlsbD0iI2ZmNjYwMCIvPjwvc3ZnPg==" alt="F1 OmniSense"/>
  <img src="https://img.shields.io/badge/React-18.3-61dafb?style=for-the-badge&logo=react" alt="React"/>
  <img src="https://img.shields.io/badge/FastAPI-0.128-009688?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb" alt="MongoDB"/>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python" alt="Python"/>
</p>

# F1 OmniSense

**Real-time Formula 1 intelligence platform** — telemetry analysis, race strategy simulation, driver profiling, circuit intelligence, and AI-powered insights across the entire F1 grid.

Built for McLaren-focused analytics with full grid coverage from 2018–2025.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                         │
│  Vite · TailwindCSS · Recharts · MapLibre · Vercel AI SDK   │
├─────────────────────────────────────────────────────────────┤
│                     FastAPI Backend                          │
│  9 OmniSuite Routers · REST API · WebSocket · LLM Gateway   │
├─────────────────────────────────────────────────────────────┤
│                    MongoDB Atlas                             │
│  44 Collections · Telemetry · Race Data · Knowledge Base     │
├─────────────────────────────────────────────────────────────┤
│                   Data Ingestion Layer                       │
│  OpenF1 API · FastF1 · Jolpica API · PDF Extraction          │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Race Intelligence
- **Live Dashboard** — Real-time race positions, gaps, intervals, and race control messages
- **Car Telemetry** — Speed, RPM, throttle, brake, DRS, and tyre data across all 20 drivers
- **Race Strategy** — Stint timelines, optimal pit windows, tyre degradation curves, safety car probability
- **Circuit Intelligence** — Track maps with speed gradients, DRS zones, sector analysis, pit loss times

### Driver & Team Analytics
- **Driver Intel** — Career stats, head-to-head comparisons, performance markers, overtake profiles
- **McLaren Analytics** — Team-specific deep dives with constructor profile analysis
- **Fleet Overview** — Multi-car telemetry comparison and health monitoring
- **Driver Biometrics** — Physiological data visualization and health gauges

### AI & Knowledge
- **AI Chatbot** — RAG-powered F1 knowledge assistant with session memory and multi-LLM support (Groq, OpenAI, Anthropic)
- **Media Intelligence** — Video analysis with YOLO object detection, CLIP embeddings, action classification
- **Regulations** — Searchable FIA regulation database with AI-powered interpretation

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite, TailwindCSS, Recharts, MapLibre GL |
| Backend | FastAPI, Uvicorn, Python 3.12 |
| Database | MongoDB Atlas (44 collections) |
| AI/LLM | Groq, LangChain, Sentence Transformers, CLIP |
| Vision | YOLOv8, VideoMAE, TimeSformer, GroundingDINO |
| Data Sources | OpenF1 API, FastF1, Jolpica API |
| Deployment | Docker Compose, Railway, RunPod |

## OmniSuite Routers

The backend is organized into 9 specialized routers:

| Router | Endpoint Prefix | Purpose |
|--------|----------------|---------|
| **OmniRAG** | `/api/omni/rag` | RAG chatbot with vector search and session memory |
| **OmniKEx** | `/api/omni/kex` | Knowledge extraction and McLaren briefings |
| **OmniVis** | `/api/omni/vis` | Computer vision and video analysis |
| **OmniBedding** | `/api/omni/bedding` | Embedding generation, clustering, t-SNE |
| **OmniHealth** | `/api/omni/health` | Vehicle health monitoring and risk scoring |
| **OmniAnalytics** | `/api/omni/analytics` | Fleet analytics and live dashboards |
| **OmniData** | `/api/omni/data` | Data ingestion and profiling |
| **OmniDoc** | `/api/omni/doc` | Document parsing (PDF, DOCX) and storage |
| **OmniDapt** | `/api/omni/dapt` | Model adaptation and anomaly detection |

## Data Pipeline

```
OpenF1 API ──┐
FastF1 ──────┤── Fetchers ──► MongoDB ──► Enrichment ──► Summary Collections
Jolpica API ─┘                              │
                                            ├── Telemetry Summaries
                                            ├── Driver Profiles
                                            ├── Constructor Profiles
                                            ├── Circuit Intelligence
                                            ├── Tyre Degradation Curves
                                            ├── Strategy Simulations
                                            └── Overtake Profiles
```

### Enrichment Scripts

| Script | Output Collection |
|--------|------------------|
| `build_telemetry_summaries.py` | `telemetry_lap_summary`, `telemetry_race_summary` |
| `build_openf1_race_summaries.py` | `telemetry_race_summary` (2025+) |
| `build_telemetry_profiles.py` | `driver_telemetry_profiles` |
| `build_constructor_profiles.py` | `constructor_profiles` |
| `fetch_circuits.py` | `circuit_intelligence` |
| `fetch_overtakes.py` | `driver_overtake_profiles` |
| `fetch_air_density.py` | `race_air_density` |
| `fetch_jolpica_full.py` | `jolpica_*` collections |

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- MongoDB Atlas connection string

### Backend

```bash
cd pipeline
pip install -r requirements.txt
cp ../.env.example ../.env  # Add your MongoDB URI and API keys
python chat_server.py
```

The API server starts on `http://localhost:8300`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The dev server starts on `http://localhost:5173`.

### Docker

```bash
docker-compose up --build
```

### Data Ingestion

Sync the latest race data from OpenF1:

```bash
cd pipeline
python -m updater.updater --year 2025
```

Run all enrichment scripts:

```bash
python -m pipeline.enrichment.run_all
```

## Environment Variables

```env
MONGODB_URI=mongodb+srv://...
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

See `.env.example` for the full list.

## Project Structure

```
f1/
├── frontend/                 # React + Vite frontend
│   ├── src/app/
│   │   ├── components/       # Page components (18 views)
│   │   ├── api/              # API client modules
│   │   └── types/            # TypeScript type definitions
│   └── public/circuits/      # GeoJSON track maps (24 circuits)
│
├── pipeline/                 # FastAPI backend
│   ├── chat_server.py        # Main server (200+ endpoints)
│   ├── omni_*_router.py      # 9 OmniSuite routers
│   ├── enrichment/           # Data enrichment scripts
│   ├── updater/              # Live data fetchers
│   ├── opponents/            # Competitor analysis
│   ├── texture/              # 3D model texturing
│   └── embeddings.py         # Nomic embedding client
│
├── colabModels/              # Colab notebooks (ML models)
│   └── tyre_degradation/     # Tyre deg polynomial fitting
│
├── docker-compose.yml        # Production deployment
└── .env.example              # Environment template
```

## License

Private — all rights reserved.
