"""Anomaly detection ensemble: IsolationForest + OneClassSVM + KNN + PCA.

High/Critical anomalies automatically get SHAP explanations identifying
the top contributing features, which feed into targeted forecasting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from omnidata._types import TabularDataset
from omnianalytics import feature_store
from omnianalytics._types import (
    AnomalyResult, AnomalyScore, SeverityLevel,
)

logger = logging.getLogger(__name__)

MODEL_WEIGHTS = {
    "IsolationForest": 1.0,
    "PCA": 0.9,
    "KNN": 0.8,
    "OneClassSVM": 0.6,
}


def estimate_contamination(data: np.ndarray) -> float:
    """IQR-based contamination estimation. Returns float in [0.01, 0.25]."""
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    IQR = np.where(IQR == 0, 1, IQR)
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_mask = np.any((data < lower) | (data > upper), axis=1)
    contamination = outlier_mask.mean()
    return float(np.clip(contamination, 0.01, 0.25))


def statistical_threshold(scores: np.ndarray, method: str = "tukey") -> float:
    """Tukey's fence threshold."""
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    if method == "tukey":
        return float(Q3 + 1.5 * IQR)
    # percentile fallback
    return float(np.percentile(scores, 95))


class AnomalyEnsemble:
    """4-model unsupervised anomaly ensemble with SHAP for high/critical points.

    Usage:
        ensemble = AnomalyEnsemble()
        result = ensemble.run(dataset)
        # result.scores has severity + SHAP top features for HIGH/CRITICAL
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run(
        self,
        dataset: TabularDataset,
        columns: Optional[List[str]] = None,
        *,
        weights: Optional[Dict[str, float]] = None,
        explain_critical: bool = True,
        top_k_features: int = 3,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None,
        db=None,
        calibrated_thresholds: Optional[Dict[str, float]] = None,
    ) -> AnomalyResult:
        """Run full ensemble on dataset metric columns.

        If explain_critical=True, runs SHAP on HIGH/CRITICAL anomalies
        and stores top_k contributing feature names in each AnomalyScore.model_scores.
        """
        # ── Feature store cache check ──
        if session_key is not None and driver_number is not None and db is not None:
            cached = feature_store.get(db, session_key, driver_number, "anomaly_scores")
            if cached is not None:
                logger.info("feature_store HIT: anomaly_scores session=%s driver=%s", session_key, driver_number)
                return AnomalyResult.from_dict(cached)

        from omnidata.profiler import profile as run_profile

        if not dataset.profile.columns:
            run_profile(dataset)

        cols = columns or dataset.profile.metric_cols
        if not cols:
            raise ValueError("No metric columns found for anomaly detection")

        df = dataset.df[cols].copy()
        df = df.apply(pd.to_numeric, errors="coerce").fillna(df.median()).fillna(0)
        data = df.values
        n_rows = len(data)

        if n_rows < 10:
            raise ValueError(f"Too few rows ({n_rows}) for anomaly detection")

        w = weights or MODEL_WEIGHTS

        # Scale
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        # Estimate contamination
        contam = estimate_contamination(scaled)

        # Run 4 models
        if_labels, if_scores = self._run_isolation_forest(scaled, contam)
        svm_labels, svm_scores = self._run_svm(scaled, contam)
        knn_labels, knn_scores = self._run_knn(scaled)
        pca_labels, pca_scores = self._run_pca(scaled)

        # Normalize all scores to [0, 1]
        if_norm = self._normalize_scores(if_scores)
        svm_norm = self._normalize_scores(svm_scores)
        knn_norm = self._normalize_scores(knn_scores)
        pca_norm = self._normalize_scores(pca_scores)

        # Weighted ensemble
        all_scores = np.column_stack([if_norm, svm_norm, knn_norm, pca_norm])
        all_labels = np.column_stack([if_labels, svm_labels, knn_labels, pca_labels])
        model_names = ["IsolationForest", "OneClassSVM", "KNN", "PCA"]

        weight_arr = np.array([w.get(m, 0.5) for m in model_names])
        weight_arr = weight_arr / weight_arr.sum()

        score_mean = all_scores.dot(weight_arr)
        score_std = all_scores.std(axis=1)
        vote_count = all_labels.sum(axis=1).astype(int)

        # Enhanced score (penalize disagreement)
        enhanced = score_mean - (score_std * 0.1)
        enhanced = np.clip(enhanced, 0, 1)

        # Threshold
        threshold = statistical_threshold(enhanced)
        anomaly_flags = enhanced >= threshold

        # Severity classification (use calibrated thresholds if available)
        severities = self._classify_severity(enhanced, score_std, calibrated_thresholds)

        # SHAP explanations for HIGH/CRITICAL
        shap_features = {}
        if explain_critical:
            shap_features = self._explain_critical(
                data, scaled, cols, severities, top_k_features,
            )

        # Build AnomalyScores
        scores = []
        severity_dist = {s.value: 0 for s in SeverityLevel}

        for i in range(n_rows):
            ms = {model_names[j]: round(float(all_scores[i, j]), 4) for j in range(4)}

            # Add SHAP top features for high/critical
            if i in shap_features:
                ms["shap_top_features"] = shap_features[i]

            s = AnomalyScore(
                row_index=i,
                score_mean=round(float(score_mean[i]), 4),
                score_std=round(float(score_std[i]), 4),
                severity=severities[i],
                vote_count=int(vote_count[i]),
                total_models=4,
                model_scores=ms,
            )
            scores.append(s)
            severity_dist[severities[i].value] += 1

        result = AnomalyResult(
            scores=scores,
            contamination_estimate=round(contam, 4),
            threshold=round(float(threshold), 4),
            anomaly_count=int(anomaly_flags.sum()),
            total_rows=n_rows,
            severity_distribution=severity_dist,
            model_weights={m: round(w.get(m, 0.5), 2) for m in model_names},
        )

        # ── Cache result in feature store ──
        if session_key is not None and driver_number is not None and db is not None:
            try:
                feature_store.put(db, session_key, driver_number, "anomaly_scores", result.to_dict())
            except Exception:
                logger.debug("Failed to cache anomaly_scores for session=%s driver=%s", session_key, driver_number)

        return result

    def _run_isolation_forest(self, scaled: np.ndarray, contamination: float) -> Tuple[np.ndarray, np.ndarray]:
        model = IsolationForest(
            contamination=contamination,
            n_jobs=-1,
            max_samples=min(256, len(scaled)),
            random_state=self.random_state,
        )
        preds = model.fit_predict(scaled)
        raw_scores = -model.score_samples(scaled)
        labels = (preds == -1).astype(int)
        return labels, raw_scores

    def _run_svm(self, scaled: np.ndarray, contamination: float) -> Tuple[np.ndarray, np.ndarray]:
        nu = min(0.5, max(0.01, contamination))
        model = SGDOneClassSVM(nu=nu, random_state=self.random_state)
        model.fit(scaled)
        scores = -model.score_samples(scaled)
        threshold = statistical_threshold(scores)
        labels = (scores >= threshold).astype(int)
        return labels, scores

    def _run_knn(self, scaled: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        k = max(2, min(k, len(scaled) - 1))
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        nn.fit(scaled)
        distances, _ = nn.kneighbors(scaled)
        avg_dist = distances[:, 1:].mean(axis=1)
        threshold = statistical_threshold(avg_dist)
        labels = (avg_dist >= threshold).astype(int)
        return labels, avg_dist

    def _run_pca(self, scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_components = max(2, min(scaled.shape[1] // 2, 10))
        if n_components >= scaled.shape[1]:
            n_components = max(1, scaled.shape[1] - 1)
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(scaled)
        reconstructed = pca.inverse_transform(transformed)
        error = np.mean((scaled - reconstructed) ** 2, axis=1)
        threshold = statistical_threshold(error)
        labels = (error >= threshold).astype(int)
        return labels, error

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Percentile-based: 5th->0.1, 95th->0.9, clipped [0,1]."""
        p5 = np.percentile(scores, 5)
        p95 = np.percentile(scores, 95)
        if p95 - p5 > 1e-10:
            normalized = 0.1 + 0.8 * (scores - p5) / (p95 - p5)
        else:
            normalized = np.full_like(scores, 0.2, dtype=float)
        return np.clip(normalized, 0.0, 1.0)

    def _classify_severity(
        self, enhanced: np.ndarray, score_std: np.ndarray,
        calibrated_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[SeverityLevel]:
        """5-level severity classification.

        If calibrated_thresholds is provided (from backtest calibration),
        uses absolute thresholds keyed by level name. Otherwise falls back
        to percentile-based thresholds (relative to current session).

        Calibrated thresholds should be: {"low": 0.3, "medium": 0.45, "high": 0.6, "critical": 0.75}
        """
        if calibrated_thresholds:
            t_critical = calibrated_thresholds.get("critical", 0.75)
            t_high = calibrated_thresholds.get("high", 0.6)
            t_medium = calibrated_thresholds.get("medium", 0.45)
            t_low = calibrated_thresholds.get("low", 0.3)
        else:
            # Fallback: percentile-based (relative to this session)
            q = np.percentile(enhanced, [75, 85, 93, 97])
            t_low, t_medium, t_high, t_critical = q[0], q[1], q[2], q[3]

        severities = []
        for i, score in enumerate(enhanced):
            if score >= t_critical:
                severities.append(SeverityLevel.CRITICAL)
            elif score >= t_high:
                severities.append(SeverityLevel.HIGH)
            elif score >= t_medium:
                severities.append(SeverityLevel.MEDIUM)
            elif score >= t_low:
                severities.append(SeverityLevel.LOW)
            else:
                severities.append(SeverityLevel.NORMAL)
        return severities

    def _explain_critical(
        self,
        raw_data: np.ndarray,
        scaled_data: np.ndarray,
        feature_names: List[str],
        severities: List[SeverityLevel],
        top_k: int = 3,
    ) -> Dict[int, List[Dict[str, float]]]:
        """Run SHAP on HIGH/CRITICAL anomalies to find top contributing features.

        Returns {row_index: [{"feature": name, "importance": float}, ...]}.
        Falls back to PCA-based feature importance if shap not available.
        """
        critical_indices = [
            i for i, s in enumerate(severities)
            if s in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)
        ]
        if not critical_indices:
            return {}

        try:
            import shap

            # Train a fast IsolationForest for SHAP
            model = IsolationForest(
                contamination=0.1, n_jobs=-1,
                max_samples=min(256, len(scaled_data)),
                random_state=self.random_state,
            )
            model.fit(scaled_data)

            # Use TreeExplainer for speed
            explainer = shap.TreeExplainer(model)
            critical_data = scaled_data[critical_indices]
            shap_values = explainer.shap_values(critical_data)

            result = {}
            for idx_in_batch, row_idx in enumerate(critical_indices):
                abs_shap = np.abs(shap_values[idx_in_batch])
                top_indices = np.argsort(abs_shap)[-top_k:][::-1]
                result[row_idx] = [
                    {
                        "feature": feature_names[fi],
                        "importance": round(float(abs_shap[fi]), 4),
                    }
                    for fi in top_indices
                ]
            return result

        except ImportError:
            logger.info("shap not installed, using PCA feature importance fallback")
            return self._explain_pca_fallback(
                scaled_data, feature_names, critical_indices, top_k,
            )

    def _explain_pca_fallback(
        self,
        scaled_data: np.ndarray,
        feature_names: List[str],
        critical_indices: List[int],
        top_k: int,
    ) -> Dict[int, List[Dict[str, float]]]:
        """PCA reconstruction error per feature as fallback explanation."""
        n_components = max(2, min(scaled_data.shape[1] // 2, 10))
        if n_components >= scaled_data.shape[1]:
            n_components = max(1, scaled_data.shape[1] - 1)
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(scaled_data)
        reconstructed = pca.inverse_transform(transformed)

        result = {}
        for row_idx in critical_indices:
            per_feature_error = (scaled_data[row_idx] - reconstructed[row_idx]) ** 2
            top_indices = np.argsort(per_feature_error)[-top_k:][::-1]
            result[row_idx] = [
                {
                    "feature": feature_names[fi],
                    "importance": round(float(per_feature_error[fi]), 4),
                }
                for fi in top_indices
            ]
        return result
