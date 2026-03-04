"""Training scheduler — event-driven retraining, drift detection, health checks.

Ported from DataSense classifier_trainer.py, simplified for omnidapt.
Uses asyncio background tasks instead of APScheduler.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from omnidapt._config import get_config
from omnidapt._types import (
    HealthReport,
    ModelDomain,
    ModelStatus,
    TrainingConfig,
    TrainingMetrics,
    TrainingStatus,
)

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """Event-driven training scheduler with drift detection."""

    def __init__(self):
        self._config = get_config()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._pending_requests: List[TrainingConfig] = []

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self):
        """Start the scheduler background loop."""
        if self._running:
            logger.info("Scheduler already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Scheduler started (interval: %dh)", self._config.retrain_interval_hours)

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    def request_training(self, config: TrainingConfig):
        """Manually request training for a dataset."""
        self._pending_requests.append(config)
        logger.info("Training requested for %s (%s)", config.dataset_name, config.domain.value)

    async def _loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Process pending manual requests
                while self._pending_requests:
                    config = self._pending_requests.pop(0)
                    await self._execute_training(config)

                # Run scheduled checks
                await self.run_scheduled_training()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler error: %s", e)

            # Sleep for interval
            try:
                await asyncio.sleep(self._config.retrain_interval_hours * 3600)
            except asyncio.CancelledError:
                break

    async def run_scheduled_training(self):
        """Check all models and retrain where needed."""
        from omnidapt.model_registry import get_model_registry

        registry = get_model_registry()

        for domain in ModelDomain:
            # Get all model names in this domain
            all_docs = registry._storage.find_documents("model_versions", {"domain": domain.value})
            model_names = set(d["name"] for d in all_docs)

            for name in model_names:
                if self.should_retrain(name):
                    prod = registry.get_production(name)
                    dataset_name = prod.dataset_name if prod else name

                    config = TrainingConfig(
                        domain=domain,
                        dataset_name=dataset_name,
                        model_name=name,
                        epochs=self._config.cv_epochs if domain == ModelDomain.CV else self._config.audio_epochs,
                        batch_size=self._config.cv_batch_size if domain == ModelDomain.CV else self._config.audio_batch_size,
                    )
                    await self._execute_training(config)

    async def _execute_training(self, config: TrainingConfig):
        """Execute a training job in background thread."""
        from omnidapt.jobs import get_job_manager

        jm = get_job_manager()

        if config.domain == ModelDomain.CV:
            from omnidapt.yolo_trainer import train_yolo
            func = train_yolo
        else:
            from omnidapt.ast_trainer import train_ast
            func = train_ast

        job_id = jm.submit(func, config=config, kwargs={"config": config})
        logger.info("Submitted %s training job %s for %s", config.domain.value, job_id, config.dataset_name)

    def should_retrain(self, model_name: str) -> bool:
        """Check if a model needs retraining."""
        from omnidapt.model_registry import get_model_registry

        registry = get_model_registry()
        prod = registry.get_production(model_name)

        if not prod:
            return False

        # Check age
        age_days = (time.time() - prod.created_at) / 86400
        if age_days > self._config.max_model_age_days:
            logger.info("Model %s is %.1f days old (max %d)", model_name, age_days, self._config.max_model_age_days)
            return True

        # Check performance threshold
        if prod.metrics:
            primary = prod.metrics.mAP50 or prod.metrics.f1_score or prod.metrics.accuracy
            if primary < self._config.perf_threshold:
                logger.info("Model %s below threshold: %.3f < %.3f", model_name, primary, self._config.perf_threshold)
                return True

        return False

    def detect_drift(self, model_name: str, recent_metrics: TrainingMetrics) -> Dict:
        """Detect performance drift by comparing against training baseline."""
        from omnidapt.model_registry import get_model_registry

        registry = get_model_registry()
        prod = registry.get_production(model_name)

        result = {"drift_detected": False, "drift_score": 0.0, "details": {}}

        if not prod or not prod.metrics:
            return result

        baseline = prod.metrics

        # Compare key metrics
        comparisons = {}
        for metric in ("mAP50", "f1_score", "accuracy"):
            baseline_val = getattr(baseline, metric, 0)
            recent_val = getattr(recent_metrics, metric, 0)
            if baseline_val > 0:
                delta = (baseline_val - recent_val) / baseline_val
                comparisons[metric] = {"baseline": baseline_val, "recent": recent_val, "delta": delta}

        if not comparisons:
            return result

        # Average drift
        avg_drift = sum(c["delta"] for c in comparisons.values()) / len(comparisons)
        result["drift_score"] = abs(avg_drift)
        result["details"] = comparisons

        # Drift threshold: >10% degradation
        if avg_drift > 0.10:
            result["drift_detected"] = True
            logger.warning("Drift detected for %s: %.1f%% degradation", model_name, avg_drift * 100)

        return result

    def check_model_health(self, model_name: str) -> HealthReport:
        """Generate a health report for a model."""
        from omnidapt.model_registry import get_model_registry

        registry = get_model_registry()
        prod = registry.get_production(model_name)

        if not prod:
            versions = registry.list_versions(model_name)
            domain = versions[0].domain if versions else ModelDomain.CV
            return HealthReport(
                model_name=model_name,
                domain=domain,
                retrain_reason="No production model",
                needs_retrain=True,
            )

        age_days = (time.time() - prod.created_at) / 86400
        needs_retrain = self.should_retrain(model_name)
        retrain_reason = ""

        if age_days > self._config.max_model_age_days:
            retrain_reason = f"Model age ({age_days:.0f}d) exceeds max ({self._config.max_model_age_days}d)"
        elif prod.metrics:
            primary = prod.metrics.mAP50 or prod.metrics.f1_score or prod.metrics.accuracy
            if primary < self._config.perf_threshold:
                retrain_reason = f"Performance ({primary:.3f}) below threshold ({self._config.perf_threshold})"

        return HealthReport(
            model_name=model_name,
            domain=prod.domain,
            status=prod.status,
            current_version=prod.version,
            age_days=age_days,
            latest_metrics=prod.metrics,
            needs_retrain=needs_retrain,
            retrain_reason=retrain_reason,
        )

    def check_all_health(self) -> List[HealthReport]:
        """Health check all registered models."""
        from omnidapt.model_registry import get_model_registry

        registry = get_model_registry()
        all_docs = registry._storage.find_documents("model_versions")
        model_names = set(d["name"] for d in all_docs)
        return [self.check_model_health(name) for name in model_names]

    def get_status(self) -> Dict:
        """Get scheduler status."""
        return {
            "running": self._running,
            "pending_requests": len(self._pending_requests),
            "retrain_interval_hours": self._config.retrain_interval_hours,
            "perf_threshold": self._config.perf_threshold,
            "max_model_age_days": self._config.max_model_age_days,
        }


# ── Singleton ────────────────────────────────────────────────────────────

_scheduler: Optional[TrainingScheduler] = None


def get_scheduler() -> TrainingScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TrainingScheduler()
    return _scheduler
