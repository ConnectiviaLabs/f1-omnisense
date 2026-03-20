"""Groq-powered ReAct agent for tabular AutoML.

Multi-turn tool-calling loop: the Groq LLM autonomously decides which ML
tools to invoke and in what order, adapting strategy based on intermediate
results.  Uses the Groq SDK directly (OpenAI-compatible function calling).

Architecture mirrors omnirag/agent.py but adds:
  - Multi-turn agentic loop (not just 2 calls)
  - Pipeline-aware system prompt
  - PipelineState as shared memory between tool calls

Usage:
    from agentml import run, RunConfig

    result = run(RunConfig(
        data_path="data.csv",
        target_column="Speed",
    ))
    print(result.best_trial.model_name)
    print(result.holdout_metrics)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from onmichine._types import PipelineState, RunConfig
from onmichine.tools import (
    build_features,
    data_fetch,
    evaluate_model,
    get_tool_definitions,
    infer_task,
    knowledge_lookup,
    load_and_profile,
    model_explain,
    package_model,
    rag_search,
    train_models,
    web_search,
)

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert AutoML agent. You build tabular ML pipelines by calling tools
in the right order. Your goal is to produce the best possible model for the
user's dataset and target column.

## Core pipeline tools (call in this order):

1. **load_and_profile** — Load dataset, profile columns, detect issues.
2. **infer_task** — Infer regression/classification from target column.
3. **build_features** — Build preprocessing pipeline, split train/test.
4. **train_models** — Multi-stage model search (baselines → boosting → HPO → ensemble).
   You can select which stages to run via the `stages` parameter.
5. **evaluate_model** — Deep evaluation: holdout metrics, confusion matrix, feature importance.
6. **package_model** — Serialize all artifacts to disk.

## Utility tools (call anytime):

7. **web_search** — Search the web for domain knowledge, feature engineering ideas,
   or modeling strategies. Use after load_and_profile to understand the domain.
8. **data_fetch** — Fetch data from MongoDB collections for enrichment or analysis.
   Pass collection name, optional query filter, and limit.
9. **model_explain** — SHAP-based deep model explanation with feature interactions
   and actionable insights. Call after train_models for deeper understanding.
10. **knowledge_lookup** — Curated ML domain knowledge base. Search by topic
    (feature_engineering, model_selection, data_quality, etc.) or tags.
    Prefer this over web_search for ML methodology questions.
11. **rag_search** — Search the RAG vectorstore for ML reference books and
    domain documents (statistical validation, permutation testing, cross-validation,
    feature selection). Use for advanced methodology questions beyond the curated
    knowledge base.

## Rules:

- Always start with load_and_profile, then infer_task, then build_features.
- After load_and_profile, consider using web_search to learn about the domain.
- After build_features, call train_models. You can call it multiple times
  with different stages if you want to be strategic (e.g., run baselines first,
  inspect results, then decide whether to run HPO).
- After training, use model_explain for deep feature analysis, then evaluate_model,
  then package_model.
- Use data_fetch when you need additional data from MongoDB to enrich features.
- Inspect each tool's result carefully. If issues are found, reason about them
  before proceeding. For example, if data has many issues, note them.
- When the pipeline is complete (package_model called), provide a final summary
  of the results: best model, key metrics, top features, and any issues found.
- Be efficient — don't call tools unnecessarily.
- If a tool returns an error, reason about what went wrong and try to recover.
"""

MAX_AGENT_TURNS = 15

# Available Groq models — user can select via RunConfig.model or AGENTML_MODEL env var
AVAILABLE_MODELS = {
    "gpt-oss-120b": {
        "id": "openai/gpt-oss-120b",
        "temperature": 1,
        "max_tokens": 8192,
        "top_p": 1,
        "reasoning_effort": "medium",
    },
    "qwen3-32b": {
        "id": "qwen/qwen3-32b",
        "temperature": 0.6,
        "max_tokens": 4096,
        "top_p": 0.95,
        "reasoning_effort": "default",
    },
    "llama-4-scout": {
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "temperature": 1,
        "max_tokens": 1024,
        "top_p": 1,
    },
    "llama-3.3-70b": {
        "id": "llama-3.3-70b-versatile",
        "temperature": 0.1,
        "max_tokens": 4096,
        "top_p": 1,
    },
}

DEFAULT_MODEL = os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile")

# ── Tool dispatcher ──────────────────────────────────────────────────────────

TOOL_HANDLERS: Dict[str, Callable] = {
    "load_and_profile": lambda state, args: load_and_profile(state),
    "infer_task": lambda state, args: infer_task(state),
    "build_features": lambda state, args: build_features(state),
    "train_models": lambda state, args: train_models(state, **args),
    "evaluate_model": lambda state, args: evaluate_model(state),
    "package_model": lambda state, args: package_model(state),
    "web_search": lambda state, args: web_search(state, **args),
    "data_fetch": lambda state, args: data_fetch(state, **args),
    "model_explain": lambda state, args: model_explain(state, **args),
    "knowledge_lookup": lambda state, args: knowledge_lookup(state, **args),
    "rag_search": lambda state, args: rag_search(state, **args),
}


# ── Agent loop ───────────────────────────────────────────────────────────────


def run(config: RunConfig) -> PipelineState:
    """Run the full agentic AutoML pipeline.

    The Groq LLM decides which tools to call and in what order.
    Falls back to a deterministic sequential pipeline if Groq is unavailable.
    """
    state = PipelineState(config=config)

    if not HAS_GROQ or not os.getenv("GROQ_API_KEY"):
        logger.warning("Groq SDK not available, falling back to sequential pipeline")
        return _run_sequential(state)

    try:
        return _run_agentic(state)
    except Exception as e:
        logger.warning("Agentic loop failed (%s), falling back to sequential", e)
        # Reset state for clean sequential run
        state = PipelineState(config=config)
        state.errors.append(f"agent_fallback: {e}")
        return _run_sequential(state)


def _resolve_model(config_model: Optional[str] = None) -> dict:
    """Resolve which Groq model to use and return its config.

    Priority: RunConfig.model > AGENTML_MODEL env > GROQ_REASONING_MODEL env > default.
    Accepts full model IDs or short names (e.g. 'qwen3-32b').
    """
    model_id = config_model or os.getenv("AGENTML_MODEL", DEFAULT_MODEL)

    # Check if it matches a short name in AVAILABLE_MODELS
    if model_id in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_id]

    # Check if it matches a full model ID
    for cfg in AVAILABLE_MODELS.values():
        if cfg["id"] == model_id:
            return cfg

    # Unknown model — use as-is with safe defaults
    return {"id": model_id, "temperature": 0.1, "max_tokens": 4096, "top_p": 1}


def _run_agentic(state: PipelineState) -> PipelineState:
    """Multi-turn Groq agent loop with tool calling."""
    cfg = state.config
    model_cfg = _resolve_model(cfg.model)
    model = model_cfg["id"]
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    tools_schema = get_tool_definitions()
    logger.info("Using Groq model: %s", model)

    # Build initial user message
    user_msg = (
        f"Build the best ML model for this dataset:\n"
        f"- Data path: {cfg.data_path}\n"
        f"- Target column: {cfg.target_column}\n"
        f"- Time budget: {cfg.time_budget_s}s\n"
        f"- Test ratio: {cfg.test_ratio}\n"
        f"- Random state: {cfg.random_state}\n"
        f"- Max HPO trials: {cfg.max_hpo_trials}\n"
        f"- Ensemble enabled: {cfg.enable_ensemble}\n"
    )
    if cfg.task_type:
        user_msg += f"- Task type (override): {cfg.task_type.value}\n"
    if cfg.sample_rows:
        user_msg += f"- Sample rows: {cfg.sample_rows}\n"

    user_msg += (
        "\nRun the full pipeline: load → profile → infer task → build features → "
        "train models → evaluate → package. Provide a summary when done."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    pipeline_start = time.time()

    for turn in range(MAX_AGENT_TURNS):
        logger.info("Agent turn %d/%d", turn + 1, MAX_AGENT_TURNS)

        try:
            create_kwargs = {
                "model": model,
                "messages": messages,
                "tools": tools_schema,
                "tool_choice": "auto",
                "temperature": model_cfg.get("temperature", 0.1),
                "max_tokens": model_cfg.get("max_tokens", 4096),
                "top_p": model_cfg.get("top_p", 1),
            }
            if "reasoning_effort" in model_cfg:
                create_kwargs["reasoning_effort"] = model_cfg["reasoning_effort"]
            response = client.chat.completions.create(**create_kwargs)
        except Exception as e:
            logger.error("Groq API call failed: %s", e)
            state.errors.append(f"groq_api: {e}")
            break

        choice = response.choices[0]
        msg = choice.message

        # No tool calls → agent is done (final summary)
        if not msg.tool_calls:
            logger.info("Agent finished with final message")
            if msg.content:
                logger.info("Agent summary:\n%s", msg.content)
            break

        # Append assistant message to history
        messages.append(msg)

        # Dispatch each tool call
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            except json.JSONDecodeError:
                fn_args = {}

            logger.info("  Tool call: %s(%s)", fn_name, fn_args)

            handler = TOOL_HANDLERS.get(fn_name)
            if handler is None:
                result = {"status": "error", "message": f"Unknown tool: {fn_name}"}
            else:
                try:
                    result = handler(state, fn_args)
                except Exception as e:
                    logger.error("  Tool %s failed: %s", fn_name, e, exc_info=True)
                    result = {"status": "error", "message": str(e)}
                    state.errors.append(f"{fn_name}: {e}")

            state.tool_calls.append({
                "tool": fn_name,
                "args": fn_args,
                "result_status": result.get("status", "unknown"),
                "turn": turn,
            })

            # Feed result back to agent
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

        # Check if stop reason indicates completion
        if choice.finish_reason == "stop":
            break

    elapsed = time.time() - pipeline_start
    logger.info("Agentic pipeline completed in %.1fs", elapsed)

    # If no tools ran successfully, raise so run() can fallback to sequential
    successful = [tc for tc in state.tool_calls if tc.get("result_status") == "ok"]
    if not successful:
        raise RuntimeError(
            f"Agentic loop produced no successful tool calls. Errors: {state.errors}"
        )

    return state


def _run_sequential(state: PipelineState) -> PipelineState:
    """Deterministic fallback: run all tools in fixed order."""
    steps = [
        ("load_and_profile", load_and_profile, {}),
        ("infer_task", infer_task, {}),
        ("build_features", build_features, {}),
        ("train_models", train_models, {}),
        ("evaluate_model", evaluate_model, {}),
        ("package_model", package_model, {}),
    ]

    for name, fn, kwargs in steps:
        logger.info("Sequential: %s", name)
        try:
            if kwargs:
                result = fn(state, **kwargs)
            else:
                result = fn(state)
            status = result.get("status", "ok")
            state.tool_calls.append({"tool": name, "result_status": status})

            if status == "error":
                state.errors.append(f"{name}: {result.get('message', 'unknown')}")
                # Fatal for first 4 steps
                if name in ("load_and_profile", "infer_task", "build_features", "train_models"):
                    logger.error("Fatal: %s failed, aborting", name)
                    break

            logger.info("  %s → %s", name, status)
        except Exception as e:
            logger.error("  %s failed: %s", name, e, exc_info=True)
            state.errors.append(f"{name}: {e}")
            if name in ("load_and_profile", "infer_task", "build_features", "train_models"):
                break

    return state
