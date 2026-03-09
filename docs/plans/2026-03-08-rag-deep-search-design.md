# RAG Deep Search for Agent Chain — Design

## Problem

All 7 agents in the omniagents chain generate LLM-powered insights using `self.reason()`, but none have access to the RAG knowledge base (`f1_knowledge`). The RAG system (BGE embeddings + Atlas `$vectorSearch` + `RAGRetriever`) exists and powers the chat interface, but agents can't use it. This means agent-generated analysis lacks grounding in technical documentation — anomaly explanations, maintenance recommendations, and forecast interpretations are based solely on structured data and agent events, not domain knowledge.

## Solution

Add `deep_search()` and `deep_search_context()` methods to the `F1Agent` base class, backed by a lazy singleton `RAGRetriever`. This gives every agent opt-in access to vector-searched document context for enriching LLM prompts. Agents decide when to activate deep search based on their own logic (e.g., severity thresholds), and users can force it via an API parameter.

## Architecture

### Base Class Integration

Add two methods to `F1Agent` in `omniagents/base.py`:

- `deep_search(query, k=5)` — returns `List[SearchResult]` (raw results with scores + metadata)
- `deep_search_context(query, k=5)` — returns formatted markdown string ready for prompt injection

Both delegate to a module-level singleton `RAGRetriever`, lazy-initialized on first call. The singleton reuses the existing `get_embedder()` (BGE-large, 1024-dim, text-only — CLIP disabled) and `get_vectorstore()` (auto-detects Atlas vs ChromaDB).

### Singleton Retriever

```python
_retriever: Optional[RAGRetriever] = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        from omnidoc.embedder import get_embedder
        from omnirag.vectorstore import get_vectorstore
        from omnirag.retriever import RAGRetriever
        embedder = get_embedder(enable_clip=False)
        store = get_vectorstore()
        _retriever = RAGRetriever(store, embedder.embed_query)
    return _retriever
```

Lazy init ensures:
- Zero overhead if no agent uses deep search
- BGE model (~1.3GB) loads once, shared across all agents
- Graceful degradation if dependencies are missing

### Activation Strategy

**Agent-decided (primary):** Each agent has its own logic for when to use deep search.

| Agent | When to deep search |
|---|---|
| TridentInsight (07) | Always — enriches all 4 report sections |
| Telemetry Anomaly (01) | HIGH/CRITICAL severity only |
| Predictive Maintenance (06) | Always — maintenance docs are high value |
| Weather Adapt (02) | When wet/intermediate conditions detected |
| Pit Window (03) | Circuit-specific tyre data lookups |
| Knowledge Convergence (04) | When fusing cross-agent intelligence |
| Visual Inspect (05) | Not initially — image-based, less doc-relevant |

**User-triggered (override):** API accepts `deep_search=true` parameter on run endpoints. When set, agents that don't normally deep search will use it for that run.

```
POST /api/omni/agents/run/telemetry_anomaly
{"session_key": 9523, "driver_number": 4, "deep_search": true}
```

### Data Flow

```
Agent needs LLM insight
    |
    v
Agent calls self.deep_search_context("McLaren brake degradation specs")
    |
    v
Singleton RAGRetriever.get_relevant_context()
    |-- BGE embed_query() -> 1024-dim vector
    |-- VectorStore.similarity_search() -> top-k docs
    |-- Cliff detection + dedup + min-score filter
    |
    v
Formatted markdown with source attribution
    |
    v
Agent injects into self.reason() prompt as "Relevant documentation:" section
    |
    v
Groq LLM generates doc-grounded analysis
```

### Graceful Degradation

If the retriever can't initialize (missing BGE model, empty vectorstore, no MONGO_URI):
- `deep_search()` returns `[]`
- `deep_search_context()` returns `"No relevant context found."`
- Agent continues without RAG context — identical to current behavior
- Warning logged once on first failed init

## Files

| File | Action |
|---|---|
| `omnisuitef1/omniagents/base.py` | Modify — add `deep_search()`, `deep_search_context()`, singleton `_get_retriever()` |
| `omnisuitef1/omniagents/agents/trident_insight.py` | Modify — call `deep_search_context()` before LLM prompts |
| `omnisuitef1/omniagents/agents/telemetry_anomaly.py` | Modify — call for HIGH/CRITICAL anomalies |
| `pipeline/omni_agents_router.py` | Modify — pass `deep_search` flag from API to `run_analysis()` |

## Existing Infrastructure Reused

| Component | Module | What it provides |
|---|---|---|
| BGE Embedder | `omnidoc/embedder.py` | `embed_query()` — 1024-dim text vectors |
| VectorStore | `omnirag/vectorstore.py` | `get_vectorstore()` — Atlas or ChromaDB |
| RAGRetriever | `omnirag/retriever.py` | `search_enhanced()`, `get_relevant_context()` — cliff detection, dedup, formatting |
| Groq LLM | `omniagents/base.py` | `self.reason()` — existing LLM call, unchanged |

## Not In Scope

- Indexing new documents (existing ingestion pipeline handles this)
- Image search via CLIP (agents don't need cross-modal search)
- Conversation memory (agents are stateless per-run, not multi-turn)
- Caching deep search results (retriever is fast enough; agents already cache their outputs)
