"""Model registry — versioning, lifecycle management, omnivis integration.

Manages model versions through: draft → validated → production → archived.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from omnidapt._config import get_config
from omnidapt._storage import get_storage
from omnidapt._types import (
    HealthReport,
    ModelDomain,
    ModelStatus,
    ModelVersion,
    TrainingMetrics,
)

logger = logging.getLogger(__name__)

COLLECTION = "model_versions"


class ModelRegistry:
    """Model version registry with lifecycle management."""

    def __init__(self):
        self._storage = get_storage()
        self._config = get_config()

    def register(
        self,
        name: str,
        domain: ModelDomain,
        artifact_path: str | Path,
        metrics: Optional[TrainingMetrics] = None,
        base_model: str = "",
        dataset_name: str = "",
        metadata: Optional[Dict] = None,
    ) -> ModelVersion:
        """Register a new model version (starts as DRAFT)."""
        existing = self.list_versions(name)
        version = self._next_version(existing)

        # Copy artifact to models dir
        src = Path(artifact_path)
        if domain == ModelDomain.CV:
            dest_dir = self._config.cv_models_dir
        else:
            dest_dir = self._config.audio_models_dir

        if src.is_file():
            dest = dest_dir / f"{name}_v{version}{src.suffix}"
            shutil.copy2(src, dest)
            stored_path = str(dest)
        elif src.is_dir():
            dest = dest_dir / f"{name}_v{version}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            stored_path = str(dest)
        else:
            stored_path = str(artifact_path)

        mv = ModelVersion(
            name=name,
            domain=domain,
            version=version,
            status=ModelStatus.DRAFT,
            artifact_path=stored_path,
            metrics=metrics,
            base_model=base_model,
            dataset_name=dataset_name,
            metadata=metadata or {},
        )

        self._storage.save_document(COLLECTION, mv.to_dict())
        logger.info("Registered %s v%s (%s)", name, version, domain.value)
        return mv

    def promote(self, name: str, version: str, to_status: ModelStatus) -> bool:
        """Promote a model version to a new status.

        When promoting to PRODUCTION, demotes the current production version.
        """
        valid_transitions = {
            ModelStatus.DRAFT: {ModelStatus.VALIDATED, ModelStatus.ARCHIVED},
            ModelStatus.VALIDATED: {ModelStatus.PRODUCTION, ModelStatus.ARCHIVED},
            ModelStatus.PRODUCTION: {ModelStatus.ARCHIVED},
        }

        doc = self._storage.find_one(COLLECTION, {"name": name, "version": version})
        if not doc:
            logger.warning("Model %s v%s not found", name, version)
            return False

        current = ModelStatus(doc["status"])
        if to_status not in valid_transitions.get(current, set()):
            logger.warning("Invalid transition: %s → %s", current.value, to_status.value)
            return False

        # Demote current production if promoting to production
        if to_status == ModelStatus.PRODUCTION:
            prod = self._storage.find_one(
                COLLECTION, {"name": name, "status": ModelStatus.PRODUCTION.value},
            )
            if prod:
                self._storage.update_document(
                    COLLECTION,
                    {"name": name, "version": prod["version"]},
                    {"status": ModelStatus.ARCHIVED.value, "promoted_at": time.time()},
                )

        self._storage.update_document(
            COLLECTION,
            {"name": name, "version": version},
            {"status": to_status.value, "promoted_at": time.time()},
        )
        logger.info("Promoted %s v%s → %s", name, version, to_status.value)
        return True

    def rollback(self, name: str) -> Optional[ModelVersion]:
        """Rollback: archive production, promote latest validated."""
        prod = self._storage.find_one(
            COLLECTION, {"name": name, "status": ModelStatus.PRODUCTION.value},
        )
        if prod:
            self._storage.update_document(
                COLLECTION,
                {"name": name, "version": prod["version"]},
                {"status": ModelStatus.ARCHIVED.value},
            )

        validated = [
            v for v in self.list_versions(name)
            if v.status == ModelStatus.VALIDATED
        ]
        if not validated:
            logger.warning("No validated version to rollback to for %s", name)
            return None

        latest = sorted(validated, key=lambda v: v.created_at, reverse=True)[0]
        self.promote(name, latest.version, ModelStatus.PRODUCTION)
        return latest

    def get_production(self, name: str) -> Optional[ModelVersion]:
        """Get the current production version."""
        doc = self._storage.find_one(
            COLLECTION, {"name": name, "status": ModelStatus.PRODUCTION.value},
        )
        if doc:
            return ModelVersion.from_dict(doc)
        return None

    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        docs = self._storage.find_documents(COLLECTION, {"name": name})
        return [ModelVersion.from_dict(d) for d in docs]

    def get_metrics_history(self, name: str) -> List[Dict]:
        """Get metrics over all versions."""
        versions = self.list_versions(name)
        return [
            {"version": v.version, "status": v.status.value,
             "metrics": v.metrics.to_dict() if v.metrics else None,
             "created_at": v.created_at}
            for v in sorted(versions, key=lambda v: v.created_at)
        ]

    def register_with_omnivis(self, name: str) -> bool:
        """Register the production model with omnivis for inference."""
        prod = self.get_production(name)
        if not prod:
            logger.warning("No production version for %s", name)
            return False

        try:
            from omnivis.model_manager import get_model_manager
            mm = get_model_manager()
            mm.register_model(name, Path(prod.artifact_path))
            logger.info("Registered %s v%s with omnivis", name, prod.version)
            return True
        except ImportError:
            logger.warning("omnivis not available, skipping registration")
            return False
        except Exception as e:
            logger.error("Failed to register with omnivis: %s", e)
            return False

    @staticmethod
    def _next_version(existing: List[ModelVersion]) -> str:
        """Compute next semver (patch bump)."""
        if not existing:
            return "1.0.0"
        versions = sorted(existing, key=lambda v: v.created_at, reverse=True)
        last = versions[0].version
        parts = last.split(".")
        try:
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)
        except ValueError:
            return f"{last}.1"


# ── Singleton ────────────────────────────────────────────────────────────

_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = None
