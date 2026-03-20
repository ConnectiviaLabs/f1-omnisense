"""AgentML — Groq-powered agentic tabular AutoML.

A ReAct-style agent that autonomously orchestrates ML pipeline tools
(load, profile, infer task, build features, train models, evaluate, package)
using Groq LLM function calling.

Quick start:
    from agentml import run, RunConfig

    result = run(RunConfig(
        data_path="data.csv",
        target_column="Speed",
    ))
    print(result.best_trial.model_name)
    print(result.holdout_metrics)
"""

from onmichine.agent import run
from onmichine._types import (
    RunConfig,
    PipelineState,
    TaskType,
    ModelStage,
    MetricName,
    ColumnSchema,
    TrialResult,
)

__all__ = [
    "run",
    "RunConfig",
    "PipelineState",
    "TaskType",
    "ModelStage",
    "MetricName",
    "ColumnSchema",
    "TrialResult",
]
