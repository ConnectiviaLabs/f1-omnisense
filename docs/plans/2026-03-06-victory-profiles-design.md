# VictoryProfiles Design

## Purpose

A 3-layer knowledge base for competitive intelligence and internal improvement analysis across F1 teams, drivers, and cars. Combines structured metrics with semantic embeddings for both similarity search and structured drill-down.

## Architecture: 3-Layer Hierarchy

### Layer 1: Driver Profiles
- **Collection**: `victory_driver_profiles`
- **Key**: `(driver_code, team, season)`
- **Sources**:
  - `driver_performance_markers` — throttle smoothness, consistency, late race pace, top speed
  - `driver_overtake_profiles` — overtake success rate, defensive rating
  - `driver_telemetry_profiles` — speed/throttle/brake distributions
  - `anomaly_scores_snapshot` — driver-level anomaly summary
- **Output**: Merged metrics + Nomic 768-dim embedding of generated narrative

### Layer 2: Car Profiles
- **Collection**: `victory_car_profiles`
- **Key**: `(team, season)`
- **Sources**:
  - `anomaly_scores_snapshot` — 7 system health scores (Power Unit, Brakes, Drivetrain, Suspension, Thermal, Electronics, Tyre Management), aggregated by team from driver-level snapshot
  - `telemetry_race_summary` — avg speed, RPM, throttle %, brake %, DRS usage
  - `constructor_profiles` — DNF rate, mechanical failures, reliability stats
- **Output**: Merged metrics + Nomic 768-dim embedding of generated narrative

### Layer 3: Team Knowledge Base
- **Collection**: `victory_team_kb`
- **Key**: `(team, season)`
- **Sources**: All driver profiles + car profile for that team-season
- **Output**: Combined narrative embedding + structured metadata blob (raw metrics for filtering/comparison)

## Embedding Strategy: Hybrid (Approach 3)

Each document stores:
1. **Structured metadata** — raw numeric metrics for filtering, diffing, and drill-down queries
2. **768-dim Nomic embedding** — of a ~200-word LLM-generated narrative summarizing the profile

This enables:
- **Semantic similarity search** — "find teams like McLaren" via cosine similarity on embeddings
- **Structured comparison** — "compare Power Unit health across teams" via metadata fields

### Narrative Generation

Ollama (qwen3.5:9b) generates a ~200-word summary per profile. Example:

> "McLaren 2024: Two-driver lineup of Norris and Piastri. Norris shows exceptional late-race pace (-0.3s delta) with high throttle smoothness (0.92). Car reliability strong — Power Unit 94% health, zero mechanical DNFs. Brakes show moderate degradation trend..."

The narrative is the embedding input. Structured metadata is stored alongside for drill-down.

## Temporal Model: Season Snapshots

One profile per entity per season. Enables:
- Season-over-season comparison for internal improvement analysis
- Historical competitive intelligence

Aligns with existing `constructor_profiles` (team+season key) and VectorProfiles direction.

## Pipeline

**Script**: `pipeline/build_victory_profiles.py`

**Execution flow**:
1. Build driver profiles — query 4 source collections, merge per (driver_code, team, season), generate narrative, embed, upsert to `victory_driver_profiles`
2. Build car profiles — query 3 source collections, merge per (team, season), generate narrative, embed, upsert to `victory_car_profiles`
3. Build team KBs — for each (team, season), pull driver + car profiles, generate combined narrative, embed, store structured metadata, upsert to `victory_team_kb`

**Idempotent**: Uses upsert on compound key. Re-running rebuilds from latest source data.

**Triggered**: Manual via CLI, same pattern as `build_vector_profiles.py`.

## API Endpoints

Added to `chat_server.py`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/victory/team/{team}/{season}` | GET | Full team KB with driver + car profiles |
| `/api/victory/compare` | POST | Compare 2+ teams by cosine similarity + structured diff |
| `/api/victory/search` | POST | Semantic search across team KBs |
| `/api/victory/regression/{team}` | GET | Season-over-season diff for internal improvement |

### Compare Flow (Competitive Intelligence)
1. Fetch team KB embeddings for requested teams
2. Compute cosine similarity matrix
3. Return similarity scores + structured metadata diff highlighting biggest gaps

### Regression Flow (Internal Improvement)
1. Fetch same team across two seasons
2. Diff structured metadata fields
3. Return ranked list of regressions/improvements with magnitude

### Search Flow
1. Embed query text with Nomic
2. Vector similarity search across `victory_team_kb`
3. Return top-K teams with similarity scores + metadata highlights

## Use Cases

1. **Competitive intelligence** — "How does McLaren's car reliability compare to Red Bull's?" / "Which team has the strongest late-race driver performance?"
2. **Internal improvement** — "Where did McLaren regress from 2023 to 2024?" / "Which car system needs investment?"

## Tyre Strategy Note

Tyre strategy behavior (compound preferences, stint lengths, degradation slopes) is excluded from car profiles — it belongs in a separate strategy layer.

## Future Integration

No frontend changes in v1. Endpoints power Team Intelligence view and future Gen UI integration.
