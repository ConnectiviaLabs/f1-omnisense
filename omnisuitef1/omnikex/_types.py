"""Type definitions for omniKeX — Knowledge Extraction from Analytics Data.

Enums, dataclasses, and constants shared across all omniKeX modules.
All dataclasses implement to_dict() / from_dict() for serialization.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ────────────────────────────────────────────────────────────────────

class InsightPillar(str, Enum):
    REALTIME = "realtime"
    ANOMALY = "anomaly"
    FORECAST = "forecast"


class InsightStatus(str, Enum):
    PENDING = "pending"
    GENERATED = "generated"
    VERIFIED = "verified"
    FAILED = "failed"


class LLMProvider(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AUTO = "auto"


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class WISEWeights:
    """WISE framework emphasis weights (W/I/S/E)."""
    weight: float = 0.25
    infer: float = 0.25
    show: float = 0.25
    exercise: float = 0.25

    def normalized(self) -> "WISEWeights":
        total = self.weight + self.infer + self.show + self.exercise
        if total == 0:
            return WISEWeights()
        return WISEWeights(
            weight=self.weight / total,
            infer=self.infer / total,
            show=self.show / total,
            exercise=self.exercise / total,
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WISEWeights":
        return cls(**{k: d[k] for k in ("weight", "infer", "show", "exercise") if k in d})


@dataclass
class WISEConfig:
    """Configuration for a single WISE extraction."""
    pillar: InsightPillar
    wise_weights: WISEWeights
    response_length: str = "medium"
    additional_instructions: str = ""
    insight_opportunities: List[str] = field(default_factory=list)
    is_dynamic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pillar": self.pillar.value,
            "wise_weights": self.wise_weights.to_dict(),
            "response_length": self.response_length,
            "additional_instructions": self.additional_instructions,
            "insight_opportunities": self.insight_opportunities,
            "is_dynamic": self.is_dynamic,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WISEConfig":
        d = dict(d)
        d["pillar"] = InsightPillar(d["pillar"])
        d["wise_weights"] = WISEWeights.from_dict(d["wise_weights"])
        return cls(**d)


@dataclass
class DataProfile:
    """Comprehensive profile of a DataFrame."""
    row_count: int
    column_count: int
    memory_mb: float
    completeness_pct: float
    numeric_cols: List[str]
    datetime_cols: List[str]
    categorical_cols: List[str]
    entity_cols: List[str]
    column_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    patterns_detected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataProfile":
        return cls(**d)


@dataclass
class DataFingerprint:
    """Verifiable snapshot of source data for grounding."""
    mode: str
    row_count: int
    column_count: int
    columns: List[str]
    numeric_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataFingerprint":
        return cls(**d)


@dataclass
class GroundingResult:
    """Result of verifying LLM claims against source data."""
    grounding_score: float
    total_claims: int
    verified_claims: int
    unverified_claims: int
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GroundingResult":
        return cls(**d)


@dataclass
class Insight:
    """A single generated insight."""
    pillar: InsightPillar
    text: str
    status: InsightStatus = InsightStatus.GENERATED
    model_used: str = ""
    provider_used: str = ""
    prompt_length: int = 0
    generation_time_s: float = 0.0
    wise_config: Optional[WISEConfig] = None
    fingerprint: Optional[DataFingerprint] = None
    grounding: Optional[GroundingResult] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "pillar": self.pillar.value,
            "text": self.text,
            "status": self.status.value,
            "model_used": self.model_used,
            "provider_used": self.provider_used,
            "prompt_length": self.prompt_length,
            "generation_time_s": round(self.generation_time_s, 3),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
        if self.wise_config is not None:
            d["wise_config"] = self.wise_config.to_dict()
        if self.fingerprint is not None:
            d["fingerprint"] = self.fingerprint.to_dict()
        if self.grounding is not None:
            d["grounding"] = self.grounding.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Insight":
        d = dict(d)
        d["pillar"] = InsightPillar(d["pillar"])
        d["status"] = InsightStatus(d["status"])
        wc = d.pop("wise_config", None)
        if wc is not None:
            wc = WISEConfig.from_dict(wc)
        fp = d.pop("fingerprint", None)
        if fp is not None:
            fp = DataFingerprint.from_dict(fp)
        gr = d.pop("grounding", None)
        if gr is not None:
            gr = GroundingResult.from_dict(gr)
        return cls(**d, wise_config=wc, fingerprint=fp, grounding=gr)


@dataclass
class ExtractionResult:
    """Collection of insights from an extraction run."""
    insights: List[Insight]
    source_profile: Optional[DataProfile] = None
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "insights": [i.to_dict() for i in self.insights],
            "generated_at": self.generated_at,
        }
        if self.source_profile is not None:
            d["source_profile"] = self.source_profile.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtractionResult":
        d = dict(d)
        d["insights"] = [Insight.from_dict(i) for i in d["insights"]]
        sp = d.pop("source_profile", None)
        if sp is not None:
            sp = DataProfile.from_dict(sp)
        return cls(**d, source_profile=sp)


# ── Constants ────────────────────────────────────────────────────────────────

PILLAR_BASE_WEIGHTS: Dict[InsightPillar, WISEWeights] = {
    InsightPillar.REALTIME: WISEWeights(0.20, 0.40, 0.30, 0.10),   # I-heavy
    InsightPillar.ANOMALY:  WISEWeights(0.35, 0.20, 0.15, 0.30),   # W+E heavy
    InsightPillar.FORECAST: WISEWeights(0.20, 0.15, 0.35, 0.30),   # S+E heavy
}

PILLAR_INDEX: Dict[InsightPillar, int] = {
    InsightPillar.REALTIME: 0,
    InsightPillar.ANOMALY:  1,
    InsightPillar.FORECAST: 2,
}

TASK_TEMPERATURES: Dict[str, float] = {
    "realtime": 0.1,
    "anomaly": 0.3,
    "forecast": 0.4,
}

PERSONA_TEMPERATURES: Dict[str, float] = {
    "ops": 0.25,
    "operations": 0.25,
    "ceo": 0.35,
    "cfo": 0.30,
    "analyst": 0.15,
    "datascientist": 0.15,
}
DEFAULT_TEMPERATURE: float = 0.20


@dataclass
class KexLLMConfig:
    """Configuration for a KeX LLM call."""
    provider: LLMProvider = LLMProvider.AUTO
    model: str = ""
    temperature: Optional[float] = None  # None = use task default
    max_tokens: int = 2400
    task_type: str = "realtime"  # realtime / anomaly / forecast
    persona: Optional[str] = None
