"""Optional RL: DynamicWISEHead, TransformerPolicyNetwork, InsightFeatureExtractor.

Ported from DataSense optimization_agent.py.
Requires torch — falls back to static PILLAR_BASE_WEIGHTS if unavailable.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from omnikex._types import (
    InsightPillar,
    PILLAR_BASE_WEIGHTS,
    PILLAR_INDEX,
    WISEConfig,
    WISEWeights,
)

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_FEATURE_DIM = 32


# ── Feature Extractor ────────────────────────────────────────────────────────

class InsightFeatureExtractor:
    """Extract 32-dimensional feature vectors from insights and context."""

    def extract(self, insight_text: str, context: Optional[Dict] = None) -> np.ndarray:
        context = context or {}
        text = insight_text or ""
        features: List[float] = []

        # Text metrics (8 features)
        features.extend([
            min(len(text) / 2000, 1.0),
            min(text.count('\n') / 20, 1.0),
            min(text.count('|') / 50, 1.0),
            min(text.count('**') / 20, 1.0),
            min(len(re.findall(r'\d+\.?\d*%', text)) / 10, 1.0),
            min(len(re.findall(r'\d+\.?\d*', text)) / 50, 1.0),
            1.0 if 'critical' in text.lower() else 0.0,
            1.0 if 'action' in text.lower() else 0.0,
        ])

        # WISE component presence (4 features)
        text_lower = text.lower()
        features.extend([
            1.0 if any(w in text_lower for w in ['impact', 'priority', 'severity', 'score']) else 0.0,
            1.0 if any(w in text_lower for w in ['pattern', 'trend', 'correlation', 'hidden']) else 0.0,
            1.0 if '|' in text or 'table' in text_lower else 0.0,
            1.0 if any(w in text_lower for w in ['option', 'recommend', 'action', 'suggest']) else 0.0,
        ])

        # Context features (8 features)
        features.extend([
            min(context.get('anomaly_count', 0) / 100, 1.0),
            min(context.get('data_points', 0) / 10000, 1.0),
            min(context.get('time_range_hours', 24) / 168, 1.0),
            1.0 if context.get('has_forecast', False) else 0.0,
            1.0 if context.get('has_anomalies', False) else 0.0,
            min(context.get('avg_severity', 0) / 3, 1.0),
            min(context.get('sensor_count', 1) / 50, 1.0),
            min(context.get('collection_age_days', 30) / 365, 1.0),
        ])

        # Historical performance (4 features)
        features.extend([
            context.get('prev_rating', 3) / 5,
            context.get('avg_rating_this_page', 3) / 5,
            context.get('avg_rating_this_collection', 3) / 5,
            min(context.get('feedback_count', 0) / 100, 1.0),
        ])

        # Pillar one-hot (3 features)
        page = context.get('page', 'realtime')
        features.extend([
            1.0 if page == 'realtime' else 0.0,
            1.0 if page == 'anomaly' else 0.0,
            1.0 if page == 'forecast' else 0.0,
        ])

        # Strategy one-hot (3 features)
        strategy = context.get('strategy_used', 0)
        features.extend([
            1.0 if strategy == 0 else 0.0,
            1.0 if strategy == 1 else 0.0,
            1.0 if strategy == 2 else 0.0,
        ])

        # Reserved (2 features)
        features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)


# ── Neural network components (torch-only) ───────────────────────────────────

if HAS_TORCH:

    class TransformerPolicyNetwork(nn.Module):
        """Transformer-based policy for WISE strategy selection (actor-critic)."""

        def __init__(
            self,
            feature_dim: int = 32,
            num_heads: int = 4,
            num_layers: int = 2,
            hidden_dim: int = 128,
            action_dim: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.input_projection = nn.Linear(feature_dim, hidden_dim)
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, action_dim),
            )

            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

            self.softmax = nn.Softmax(dim=-1)

        def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
            """Get hidden representation for use by WISE head."""
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.input_projection(x)
            x = x + self.pos_embedding
            x = self.transformer(x)
            return x.squeeze(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.get_hidden(x)
            logits = self.policy_head(hidden)
            return self.softmax(logits)

        def forward_with_value(self, x: torch.Tensor):
            hidden = self.get_hidden(x)
            action_probs = self.softmax(self.policy_head(hidden))
            value = self.value_head(hidden) * 5
            return action_probs, value

    class DynamicWISEHead(nn.Module):
        """Learns context-adaptive W/I/S/E emphasis weights."""

        def __init__(self, hidden_dim: int = 128):
            super().__init__()
            self.wise_net = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 4),
            )
            self.page_base_weights = nn.Parameter(torch.tensor([
                [0.20, 0.40, 0.30, 0.10],  # realtime: I-heavy
                [0.35, 0.20, 0.15, 0.30],  # anomaly: W+E heavy
                [0.20, 0.15, 0.35, 0.30],  # forecast: S+E heavy
            ]))
            self.blend_ratio = nn.Parameter(torch.tensor(0.3))

        def forward(self, hidden: torch.Tensor, page_idx: int) -> torch.Tensor:
            adjustment_logits = self.wise_net(hidden)
            adjustment = torch.softmax(adjustment_logits, dim=-1)
            base = self.page_base_weights[page_idx]
            blend = torch.clamp(self.blend_ratio, 0.1, 0.5)
            blended = (1 - blend) * base + blend * adjustment
            return blended / blended.sum(dim=-1, keepdim=True)

        def get_weights_dict(self, hidden: torch.Tensor, page_idx: int) -> Dict[str, float]:
            """Get WISE weights as a dictionary."""
            with torch.no_grad():
                weights = self.forward(hidden, page_idx)
                if weights.dim() > 1:
                    weights = weights[0]
                return {
                    "weight": float(weights[0]),
                    "infer": float(weights[1]),
                    "show": float(weights[2]),
                    "exercise": float(weights[3]),
                }


# ── WISE Optimizer ───────────────────────────────────────────────────────────

class WISEOptimizer:
    """Wraps policy + head + feature extractor + training loop.

    Works with or without torch:
    - With torch: uses TransformerPolicyNetwork + DynamicWISEHead for dynamic weights
    - Without torch: returns static PILLAR_BASE_WEIGHTS
    """

    def __init__(self):
        self._feature_extractor = InsightFeatureExtractor()
        self._torch_enabled = HAS_TORCH
        self._feedback_count = 0
        self._training_threshold = 10
        self._policy_version = 0

        if self._torch_enabled:
            self._policy = TransformerPolicyNetwork(
                feature_dim=_FEATURE_DIM,
                num_heads=4,
                num_layers=2,
                hidden_dim=128,
                action_dim=3,
                dropout=0.1,
            )
            self._wise_head = DynamicWISEHead(hidden_dim=128)
            self._policy_optimizer = optim.Adam(self._policy.parameters(), lr=0.0001)
            self._head_optimizer = optim.Adam(self._wise_head.parameters(), lr=0.001)
            self._strategy_buffers: Dict[int, deque] = {
                i: deque(maxlen=3000) for i in range(3)
            }
            self._policy.eval()
            self._wise_head.eval()
        else:
            self._policy = None
            self._wise_head = None
            self._policy_optimizer = None
            self._head_optimizer = None
            self._strategy_buffers = {}

    def get_wise_config(
        self,
        pillar: InsightPillar,
        context: Optional[Dict[str, Any]] = None,
    ) -> WISEConfig:
        """Get WISE configuration for a pillar, optionally with dynamic weights."""
        pillar_idx = PILLAR_INDEX[pillar]
        base_weights = PILLAR_BASE_WEIGHTS[pillar]

        dynamic = False
        weights = base_weights

        if self._torch_enabled and self._policy is not None and self._wise_head is not None and context:
            try:
                ctx = dict(context)
                ctx["page"] = pillar.value
                ctx["strategy_used"] = pillar_idx
                features = self._feature_extractor.extract("", ctx)
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                with torch.no_grad():
                    hidden = self._policy.get_hidden(features_tensor)
                    weights_dict = self._wise_head.get_weights_dict(hidden, pillar_idx)
                weights = WISEWeights(**weights_dict)
                dynamic = True
            except Exception as e:
                logger.warning("Dynamic WISE weights failed, using base: %s", e)
                weights = base_weights

        return WISEConfig(
            pillar=pillar,
            wise_weights=weights,
            is_dynamic=dynamic,
            insight_opportunities=[],
        )

    def record_feedback(
        self,
        insight_text: str,
        rating: float,
        pillar: InsightPillar,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record feedback for WISE optimization."""
        if not self._torch_enabled:
            return

        pillar_idx = PILLAR_INDEX[pillar]
        self._feedback_count += 1

        ctx = dict(context) if context else {}
        ctx["page"] = pillar.value
        ctx["strategy_used"] = pillar_idx
        ctx["prev_rating"] = rating
        ctx["feedback_count"] = self._feedback_count

        features = self._feature_extractor.extract(insight_text, ctx)

        self._strategy_buffers[pillar_idx].append({
            "value": rating,
            "features": features,
            "weights_used": None,  # Will be set if we track inference cache
        })

        if self._feedback_count >= self._training_threshold:
            self._train_wise_networks(pillar_idx)
            self._feedback_count = 0
            self._policy_version += 1

    def _train_wise_networks(self, strategy_id: int, epochs: int = 5) -> None:
        """Train TransformerPolicyNetwork + DynamicWISEHead on feedback.

        Good ratings (>= 3.5): reinforce the WISE weights that were used.
        Bad ratings (< 2.5): pull weights back toward the hand-tuned base.
        Medium ratings: skipped (not informative enough).
        """
        if not self._torch_enabled:
            return

        buffer = self._strategy_buffers.get(strategy_id)
        if not buffer or len(buffer) < 5:
            return

        base = self._wise_head.page_base_weights[strategy_id].detach()
        samples = []

        for item in list(buffer)[-50:]:
            features = item.get("features")
            if features is None:
                continue
            rating = item["value"]
            weights_used = item.get("weights_used")

            if rating >= 3.5 and weights_used:
                target = torch.tensor(
                    [weights_used["weight"], weights_used["infer"],
                     weights_used["show"], weights_used["exercise"]]
                )
            elif rating < 2.5:
                target = base
            else:
                continue
            samples.append((features, target))

        if len(samples) < 3:
            return

        self._policy.train()
        self._wise_head.train()

        for _ in range(epochs):
            total_loss = torch.tensor(0.0)
            for features_np, target in samples:
                features_t = torch.FloatTensor(features_np).unsqueeze(0)
                hidden = self._policy.get_hidden(features_t)
                predicted = self._wise_head(hidden, strategy_id)
                total_loss = total_loss + F.mse_loss(predicted, target.unsqueeze(0))

            if total_loss.requires_grad:
                self._policy_optimizer.zero_grad()
                self._head_optimizer.zero_grad()
                total_loss.backward()
                self._policy_optimizer.step()
                self._head_optimizer.step()

        self._policy.eval()
        self._wise_head.eval()
        logger.info(
            "WISE networks trained for pillar %d on %d samples (loss=%.4f)",
            strategy_id, len(samples), total_loss.item(),
        )

    def train(self, pillar: Optional[InsightPillar] = None) -> None:
        """Explicitly trigger training for one or all pillars."""
        if pillar is not None:
            self._train_wise_networks(PILLAR_INDEX[pillar])
        else:
            for idx in range(3):
                self._train_wise_networks(idx)


# ── Singleton ────────────────────────────────────────────────────────────────

_optimizer_instance: Optional[WISEOptimizer] = None


def get_optimizer() -> WISEOptimizer:
    """Get the singleton WISEOptimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = WISEOptimizer()
    return _optimizer_instance
