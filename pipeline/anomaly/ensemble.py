"""Backward-compatible DataFrame-based anomaly ensemble wrapper.

Canonical implementation lives in ``omnianalytics/anomaly.py`` (TabularDataset
API).  This module keeps the legacy DataFrame-in / DataFrame-out API used by
``pipeline.anomaly.run_f1_anomaly`` and
``omnisuitef1.omniagents.agents.telemetry_anomaly``.

Shared utilities (``estimate_contamination``, ``statistical_threshold``,
``MODEL_WEIGHTS``) are imported from omnianalytics so they stay in sync.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import NearestNeighbors

from omnianalytics.anomaly import (
    AnomalyEnsemble,
    estimate_contamination,
    statistical_threshold,
    MODEL_WEIGHTS,
)

logger = logging.getLogger(__name__)

# Re-export canonical symbols so ``from pipeline.anomaly.ensemble import …``
# still works for anything that referenced them directly.
__all__ = [
    "AnomalyDetectionEnsemble",
    "AnomalyStatistics",
    "severity_from_votes",
    "estimate_contamination",
    "statistical_threshold",
    "MODEL_WEIGHTS",
    "AnomalyEnsemble",
]

# MODEL_WEIGHTS in omnianalytics uses "PCA"; legacy pipeline code expects
# "Autoencoder".  Build a compat mapping once at import time.
_LEGACY_WEIGHTS = {
    k if k != "PCA" else "Autoencoder": v for k, v in MODEL_WEIGHTS.items()
}


# ─── Legacy DataFrame-based Ensemble ─────────────────────────────────


class AnomalyDetectionEnsemble:
    """Ensemble of anomaly detection models with consistent scoring.

    This is the *legacy* DataFrame-based API.  New code should prefer
    ``omnianalytics.anomaly.AnomalyEnsemble`` which operates on
    ``TabularDataset`` objects.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # -- internal helpers --------------------------------------------------

    def _normalize_scores(self, df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
        """Percentile-based normalization: normal~0.1, anomaly~0.9."""
        for col in score_cols:
            if col not in df.columns:
                continue
            scores = df[col].values
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            if p95 - p5 > 1e-10:
                normalized = 0.1 + 0.8 * (scores - p5) / (p95 - p5)
            else:
                normalized = np.full_like(scores, 0.2, dtype=float)
            df[col] = np.clip(normalized, 0.0, 1.0)
        return df

    def _add_knn_distance_score(
        self, scaled_data: pd.DataFrame, out_df: pd.DataFrame, k: int = 5,
    ) -> pd.DataFrame:
        """KNN-based anomaly score with Tukey's fence threshold."""
        try:
            n = scaled_data.shape[0]
            if n <= 2:
                raise ValueError("Too few samples for KNN scoring.")
            k = max(2, min(k, n - 1))
            nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
            nn.fit(scaled_data)
            distances, _ = nn.kneighbors(scaled_data)
            avg_dist = distances[:, 1:].mean(axis=1)
            out_df["KNN_AnomalyScore"] = avg_dist
            thr = statistical_threshold(avg_dist, method="tukey")
            out_df["KNN_Anomaly"] = (avg_dist >= thr).astype(int)
        except Exception as e:
            logger.error(f"KNN distance score failed: {e}")
            out_df["KNN_AnomalyScore"] = 0.0
            out_df["KNN_Anomaly"] = 0
        return out_df

    def _fast_reconstruction_error(self, data):
        """Fast PCA-based reconstruction error."""
        try:
            n_components = max(2, min(data.shape[1] // 2, 10))
            pca = PCA(n_components=n_components, random_state=self.random_state)
            transformed = pca.fit_transform(data)
            reconstructed = pca.inverse_transform(transformed)
            error = np.mean((data - reconstructed) ** 2, axis=1)
            threshold_raw = statistical_threshold(error, method="tukey")
            ranks = pd.Series(error).rank(pct=True).values
            normalized_error = np.clip(ranks, 0.0, 1.0)
            threshold = float((error <= threshold_raw).mean())
            return normalized_error, threshold
        except Exception as e:
            logger.error(f"PCA reconstruction failed: {e}")
            return np.zeros(len(data)), 0.5

    # -- public API --------------------------------------------------------

    def sklearn_models(
        self, raw_data: pd.DataFrame, scaled_data: pd.DataFrame,
    ) -> tuple:
        """Run IsolationForest + OneClassSVM + KNN, mutate *raw_data* in place."""
        if len(scaled_data) != len(raw_data):
            min_len = min(len(scaled_data), len(raw_data))
            scaled_data = scaled_data.iloc[:min_len].copy()
            raw_data = raw_data.iloc[:min_len].copy()

        scaled_data = scaled_data.fillna(scaled_data.median()).fillna(0)
        est_contam = estimate_contamination(scaled_data.values)

        algorithms = {
            "IsolationForest": IsolationForest(
                random_state=self.random_state,
                contamination=est_contam,
                n_jobs=-1,
                max_samples=min(256, len(scaled_data)),
            ),
            "OneClassSVM": SGDOneClassSVM(
                nu=min(0.5, max(0.01, est_contam)),
                random_state=self.random_state,
            ),
        }

        for name, algo in algorithms.items():
            try:
                preds = algo.fit_predict(scaled_data)
                scores = -algo.decision_function(scaled_data)
                raw_data[f"{name}_Anomaly"] = np.where(preds == -1, 1, 0)
                raw_data[f"{name}_AnomalyScore"] = scores
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                raw_data[f"{name}_Anomaly"] = 0
                raw_data[f"{name}_AnomalyScore"] = 0.0

        raw_data = self._add_knn_distance_score(scaled_data, raw_data, k=5)

        score_cols = [c for c in raw_data.columns if c.endswith("_AnomalyScore")]
        raw_data = self._normalize_scores(raw_data, score_cols)
        return scaled_data, raw_data

    def run_autoencoder(
        self, scaled_data: pd.DataFrame, raw_data: pd.DataFrame,
    ) -> tuple:
        """PCA-based reconstruction (fast, no TF dependency)."""
        normalized_error, threshold = self._fast_reconstruction_error(scaled_data)
        raw_data["Autoencoder_Anomaly"] = np.where(
            normalized_error > threshold, 1, 0,
        )
        raw_data["Autoencoder_AnomalyScore"] = normalized_error
        return scaled_data, raw_data

    def run_anomaly_detection_models(
        self, raw_data: pd.DataFrame, scaled_data: pd.DataFrame,
    ) -> tuple:
        """Full ensemble pipeline: sklearn models + autoencoder."""
        scaled_data, raw_data = self.sklearn_models(raw_data, scaled_data)
        scaled_data, raw_data = self.run_autoencoder(scaled_data, raw_data)
        return scaled_data, raw_data


# ─── Statistics ───────────────────────────────────────────────────────


class AnomalyStatistics:
    """Aggregate ensemble results into severity levels."""

    def anomaly_insights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute voting, weighted scores, severity levels."""
        anoms = [
            "IsolationForest_Anomaly",
            "OneClassSVM_Anomaly",
            "Autoencoder_Anomaly",
        ]
        scores = [
            "IsolationForest_AnomalyScore",
            "OneClassSVM_AnomalyScore",
            "Autoencoder_AnomalyScore",
        ]

        # Filter to columns that actually exist
        anoms = [a for a in anoms if a in df.columns]
        scores = [s for s in scores if s in df.columns]

        if not anoms or not scores:
            df["Anomaly_Level"] = "normal"
            return df

        anomalies = df[anoms]
        score_vals = df[scores]

        anomaly_sum = anomalies.sum(axis=1)
        score_mean = score_vals.mean(axis=1)
        score_std = score_vals.std(axis=1)

        # KNN columns
        if "KNN_Anomaly" in df.columns:
            anomaly_sum += df["KNN_Anomaly"]
        if "KNN_AnomalyScore" in df.columns:
            score_mean = (
                score_mean * len(scores) + df["KNN_AnomalyScore"]
            ) / (len(scores) + 1)

        df["Voted_Anomaly"] = np.where(anomaly_sum >= 2, 1, 0)
        df["Anomaly_Score_Mean"] = score_mean
        df["Anomaly_Score_STD"] = score_std
        df["Voting_Score"] = anomaly_sum / (
            len(anoms) + (1 if "KNN_Anomaly" in df.columns else 0)
        )

        # Reliability-weighted score
        model_names = [a.replace("_Anomaly", "") for a in anoms]
        if "KNN_Anomaly" in df.columns:
            model_names.append("KNN")
        weights = np.array(
            [_LEGACY_WEIGHTS.get(m, 0.5) for m in model_names],
        )
        weights = weights / weights.sum()

        all_scores_cols = scores + (
            ["KNN_AnomalyScore"] if "KNN_AnomalyScore" in df.columns else []
        )
        df["Reliability_Weighted_Score"] = df[all_scores_cols].values.dot(weights)

        # Enhanced score
        df["Enhanced_Anomaly_Score"] = df["Reliability_Weighted_Score"] - (
            score_std * 0.1
        )
        df["Enhanced_Anomaly_Score"] += np.where(
            anomaly_sum == len(model_names), 0.05, 0,
        )

        # Severity classification
        sq = df["Enhanced_Anomaly_Score"].quantile(
            q=[0.25, 0.50, 0.75, 0.85, 0.95],
        )
        std_q = score_std.quantile(q=[0.25, 0.50, 0.75])

        df["Anomaly_Level"] = df.apply(
            lambda r: self._classify(
                r["Enhanced_Anomaly_Score"], r["Anomaly_Score_STD"], sq, std_q,
            ),
            axis=1,
        )
        return df

    @staticmethod
    def _classify(score, std_dev, sq, std_q):
        adj = 0.1 if std_dev >= std_q[0.75] else (-0.1 if std_dev <= std_q[0.25] else 0)
        if score >= sq[0.95] + adj:
            return "critical"
        elif score >= sq[0.85] + adj:
            return "high"
        elif score >= sq[0.75]:
            return "medium"
        elif score >= sq[0.50]:
            return "low"
        return "normal"


# ─── Vote-based severity (from anomaly_broadcaster.py) ────────────────


def severity_from_votes(vote_count: int, total_models: int = 4) -> str:
    """Data-driven severity based on model voting consensus."""
    if vote_count == 0:
        return "normal"
    elif vote_count == 1:
        return "low"
    elif vote_count == 2:
        return "medium"
    elif vote_count == 3:
        return "high"
    else:
        return "critical"
