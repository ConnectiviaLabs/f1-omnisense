# RAG Deep Search for Agent Chain — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `deep_search()` and `deep_search_context()` to the F1Agent base class so all 7 agents can opt-in to RAG-grounded LLM reasoning.

**Architecture:** Lazy singleton `RAGRetriever` in `base.py`, initialized on first `deep_search()` call. Reuses existing `BGEEmbedder` (1024-dim) + `get_vectorstore()` (Atlas/$vectorSearch or ChromaDB). Each agent decides when to search; API exposes `deep_search=true` override.

**Tech Stack:** Python, BGE-large-en-v1.5, MongoDB Atlas $vectorSearch, RAGRetriever (cliff detection + dedup), Groq LLM

---

### Task 1: Add deep_search to F1Agent base class

**Files:**
- Modify: `omnisuitef1/omniagents/base.py`

**Step 1: Add singleton retriever function**

Add after the `_GROQ_MODEL` line (~line 41), before the `F1Agent` class:

```python
# ── RAG deep search (lazy singleton) ─────────────────────────────────────

_retriever = None
_retriever_init_failed = False


def _get_retriever():
    """Lazy singleton: BGE embedder + vectorstore → RAGRetriever."""
    global _retriever, _retriever_init_failed
    if _retriever is not None:
        return _retriever
    if _retriever_init_failed:
        return None
    try:
        from omnidoc.embedder import get_embedder
        from omnirag.vectorstore import get_vectorstore
        from omnirag.retriever import RAGRetriever

        embedder = get_embedder(enable_bge=True, enable_clip=False)
        store = get_vectorstore()
        _retriever = RAGRetriever(store, embedder.embed_query)
        logger.info("RAG deep search retriever initialized")
        return _retriever
    except Exception:
        _retriever_init_failed = True
        logger.warning("RAG deep search unavailable — missing dependencies or vectorstore")
        return None
```

**Step 2: Add deep_search methods to F1Agent class**

Add after the `reason()` method (~line 145), before the state management section:

```python
    # ── RAG deep search ─────────────────────────────────────────────────

    async def deep_search(self, query: str, k: int = 5) -> list:
        """Vector search against f1_knowledge. Returns List[SearchResult] or []."""
        retriever = _get_retriever()
        if retriever is None:
            return []
        try:
            return await asyncio.to_thread(retriever.search_enhanced, query, k=k)
        except Exception:
            logger.exception("[%s] deep_search failed for query: %s", self.name, query[:80])
            return []

    async def deep_search_context(self, query: str, k: int = 5) -> str:
        """Vector search returning formatted markdown for prompt injection."""
        retriever = _get_retriever()
        if retriever is None:
            return "No relevant context found."
        try:
            return await asyncio.to_thread(retriever.get_relevant_context, query, k=k)
        except Exception:
            logger.exception("[%s] deep_search_context failed for query: %s", self.name, query[:80])
            return "No relevant context found."
```

**Step 3: Verify import works**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from omniagents.base import F1Agent; print('deep_search' in dir(F1Agent)); print('OK')"
```
Expected: `True` then `OK`

**Step 4: Commit**

```bash
git add omnisuitef1/omniagents/base.py
git commit -m "feat: add deep_search() and deep_search_context() to F1Agent base class"
```

---

### Task 2: Add deep_search flag to API and RunRequest

**Files:**
- Modify: `pipeline/omni_agents_router.py`

**Step 1: Add deep_search to RunRequest model**

In `RunRequest` class (~line 77), add the field:

```python
class RunRequest(BaseModel):
    session_key: int
    driver_number: Optional[int] = None
    year: Optional[int] = None
    deep_search: bool = False
```

**Step 2: Store deep_search flag on agent before execution**

In the `run_agent` endpoint (~line 86), after getting the agent but before creating the task, set the flag:

```python
    agent = _registry.get(agent_name)
    if agent is None:
        raise HTTPException(404, f"Agent '{agent_name}' not found. Available: {_registry.agent_names}")

    # Pass deep_search flag to agent for this run
    agent._deep_search_override = req.deep_search
```

Also in the `run_all_agents` endpoint (~line 111), before creating the task:

```python
    # Set deep_search override on all agents
    for name in _registry.agent_names:
        ag = _registry.get(name)
        if ag:
            ag._deep_search_override = req.deep_search
```

**Step 3: Add _deep_search_override attribute to F1Agent.__init__**

In `omnisuitef1/omniagents/base.py`, in `F1Agent.__init__` (~line 52), add:

```python
    def __init__(self, bus: EventBus, db=None):
        self._bus = bus
        self._db = db
        self._groq = Groq() if _HAS_GROQ else None
        self._state = AgentState(agent_id=self.name, name=self.name)
        self._deep_search_override = False

        # Wire subscriptions
        for topic in self.subscriptions:
            self._bus.subscribe(topic, self._handle_event)
```

**Step 4: Verify router loads**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from pipeline.omni_agents_router import router; print(len(router.routes), 'routes'); print('OK')"
```
Expected: route count and `OK`

**Step 5: Commit**

```bash
git add omnisuitef1/omniagents/base.py pipeline/omni_agents_router.py
git commit -m "feat: add deep_search flag to RunRequest and F1Agent"
```

---

### Task 3: Integrate deep search into TridentInsightAgent

**Files:**
- Modify: `omnisuitef1/omniagents/agents/trident_insight.py`

**Step 1: Add RAG context retrieval in _build_report**

In `_build_report()`, after gathering MongoDB context (~line 122) and before building prompts (~line 131), add the RAG deep search:

```python
        # --- RAG deep search for document grounding ---
        rag_context = ""
        if self._deep_search_override or True:  # TridentInsight always uses deep search
            rag_queries = [
                f"McLaren {scope} race analysis intelligence",
                f"McLaren anomaly detection telemetry {entity or 'grid'}",
                f"McLaren forecast maintenance schedule",
            ]
            rag_blocks = await asyncio.gather(
                *(self.deep_search_context(q, k=3) for q in rag_queries)
            )
            # Merge and deduplicate
            rag_context = "\n---\n".join(
                block for block in rag_blocks
                if block != "No relevant context found."
            )
```

**Step 2: Inject RAG context into the 3 section prompts**

Append to each of the 3 prompts. After the existing `insights_prompt`, `anomaly_prompt`, and `forecast_prompt` strings, add RAG context:

```python
        if rag_context:
            rag_section = f"\n\nRelevant technical documentation:\n{rag_context}"
            insights_prompt += rag_section
            anomaly_prompt += rag_section
            forecast_prompt += rag_section
```

**Step 3: Add rag_sources to report metadata**

In the report dict metadata section (~line 191), add:

```python
            "metadata": {
                "model_used": _GROQ_MODEL,
                "generation_time_s": round(time.time() - start, 2),
                "deep_search": bool(rag_context),
            },
```

**Step 4: Verify import works**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from omniagents.agents.trident_insight import TridentInsightAgent; print('OK')"
```
Expected: `OK`

**Step 5: Commit**

```bash
git add omnisuitef1/omniagents/agents/trident_insight.py
git commit -m "feat: integrate RAG deep search into TridentInsightAgent reports"
```

---

### Task 4: Integrate deep search into TelemetryAnomalyAgent

**Files:**
- Modify: `omnisuitef1/omniagents/agents/telemetry_anomaly.py`

**Step 1: Add deep search for HIGH/CRITICAL anomalies**

In `run_analysis()`, after extracting critical anomalies (~line 62) and before the existing `self.reason()` call (~line 65), add RAG context:

```python
        # 4. LLM reasoning on critical/high anomalies
        critical = [a for a in anomalies if a["severity"] in ("critical", "high")]
        insight = None
        if critical:
            # Deep search for relevant documentation (HIGH/CRITICAL or user override)
            rag_context = ""
            if self._deep_search_override or len(critical) > 0:
                top_features = []
                for a in critical[:5]:
                    for feat in TELEMETRY_FEATURES:
                        if a.get(feat) is not None and feat not in top_features:
                            top_features.append(feat)
                query = f"McLaren {' '.join(top_features[:3])} anomaly specifications normal ranges"
                rag_context = await self.deep_search_context(query, k=4)

            prompt = (
                f"Analyze these {len(critical)} anomalies detected in session {session_key}"
                f"{f' for driver #{driver_number}' if driver_number else ''}. "
                "What do they indicate about car health? What actions should the team take?"
            )
            if rag_context and rag_context != "No relevant context found.":
                prompt += f"\n\nRelevant technical documentation:\n{rag_context}"

            insight = await self.reason(prompt, data_context={"anomalies": critical[:10]})
```

This replaces the existing lines 62-70.

**Step 2: Verify import works**

Run:
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "from omniagents.agents.telemetry_anomaly import TelemetryAnomalyAgent; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add omnisuitef1/omniagents/agents/telemetry_anomaly.py
git commit -m "feat: add RAG deep search to TelemetryAnomalyAgent for critical anomalies"
```

---

## Verification

After all 4 tasks:

1. **Import check:**
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -c "
from omniagents.base import F1Agent
from omniagents.agents.trident_insight import TridentInsightAgent
from omniagents.agents.telemetry_anomaly import TelemetryAnomalyAgent
from pipeline.omni_agents_router import router
print('deep_search' in dir(F1Agent))
print('deep_search_context' in dir(F1Agent))
print(len(router.routes), 'routes')
print('ALL OK')
"
```
Expected: `True`, `True`, route count, `ALL OK`

2. **Start backend:**
```bash
PYTHONPATH=omnisuitef1:pipeline:. python3 -m uvicorn pipeline.chat_server:app --host 0.0.0.0 --port 8300
```

3. **Test deep_search flag via API:**
```bash
curl -X POST http://localhost:8300/api/omni/agents/run/trident_insight \
  -H "Content-Type: application/json" \
  -d '{"session_key": 9523, "driver_number": 4, "deep_search": true}'
```

4. **Check output includes deep_search metadata:**
```bash
curl http://localhost:8300/api/omni/agents/trident_insight/output/9523 | python3 -m json.tool | grep deep_search
```
