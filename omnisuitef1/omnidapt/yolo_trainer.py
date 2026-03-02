"""YOLO fine-tuning pipeline via ultralytics.

Exports dataset → trains YOLO model → registers result in model registry.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from omnidapt._config import get_config
from omnidapt._types import (
    ModelDomain,
    TrainingConfig,
    TrainingMetrics,
    TrainingStatus,
)

logger = logging.getLogger(__name__)


class YOLOTrainer:
    """YOLO fine-tuning trainer using ultralytics."""

    def __init__(self):
        self._config = get_config()

    def train(
        self,
        config: TrainingConfig,
        job_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        """Run YOLO fine-tuning.

        1. Export dataset to YOLO format
        2. Train with ultralytics
        3. Parse results
        4. Register model version

        Returns dict with metrics and result_path.
        """
        from omnidapt.dataset_manager import get_dataset_manager
        from omnidapt.jobs import get_job_manager
        from omnidapt.model_registry import get_model_registry

        jm = get_job_manager() if job_id else None
        dm = get_dataset_manager()
        registry = get_model_registry()

        def _progress(pct: float, phase: str = "", detail: str = ""):
            if jm and job_id:
                jm.update_progress(job_id, pct, status=TrainingStatus.TRAINING, phase=phase, detail=detail)

        # 1. Export dataset
        _progress(5, "export", f"Exporting {config.dataset_name} to YOLO format")
        try:
            export_dir = dm.export_yolo(config.dataset_name)
        except Exception as e:
            raise RuntimeError(f"Dataset export failed: {e}") from e

        data_yaml = export_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found in {export_dir}")

        _progress(10, "setup", "Loading YOLO model")

        # 2. Train
        from ultralytics import YOLO

        base_model = config.base_model or self._config.cv_base_model
        model = YOLO(base_model)

        device = config.device or "0" if _cuda_available() else "cpu"
        experiment_dir = self._config.experiments_dir / f"{config.model_name}_{int(time.time())}"

        _progress(15, "training", f"Starting training ({config.epochs} epochs)")

        if cancel_event and cancel_event.is_set():
            return {"cancelled": True}

        results = model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=config.image_size,
            lr0=config.learning_rate,
            patience=config.patience,
            device=device,
            project=str(experiment_dir),
            name="train",
            exist_ok=True,
            verbose=False,
        )

        if cancel_event and cancel_event.is_set():
            return {"cancelled": True}

        _progress(85, "evaluating", "Parsing results")

        # 3. Parse metrics
        metrics = self._parse_results(results, experiment_dir)

        # 4. Register model
        best_pt = experiment_dir / "train" / "weights" / "best.pt"
        if not best_pt.exists():
            # Try alternative paths
            for candidate in experiment_dir.rglob("best.pt"):
                best_pt = candidate
                break

        _progress(95, "registering", "Registering model version")

        if best_pt.exists():
            mv = registry.register(
                name=config.model_name,
                domain=ModelDomain.CV,
                artifact_path=best_pt,
                metrics=metrics,
                base_model=base_model,
                dataset_name=config.dataset_name,
            )
            result_path = mv.artifact_path
        else:
            logger.warning("best.pt not found, registering experiment dir")
            mv = registry.register(
                name=config.model_name,
                domain=ModelDomain.CV,
                artifact_path=experiment_dir,
                metrics=metrics,
                base_model=base_model,
                dataset_name=config.dataset_name,
            )
            result_path = mv.artifact_path

        _progress(100, "complete", f"Training complete — v{mv.version}")

        return {
            "metrics": metrics.to_dict(),
            "result_path": result_path,
            "version": mv.version,
            "experiment_dir": str(experiment_dir),
        }

    @staticmethod
    def _parse_results(results, experiment_dir: Path) -> TrainingMetrics:
        """Extract metrics from ultralytics training results."""
        metrics = TrainingMetrics()

        try:
            if hasattr(results, "results_dict"):
                rd = results.results_dict
                metrics.mAP50 = rd.get("metrics/mAP50(B)", 0)
                metrics.mAP50_95 = rd.get("metrics/mAP50-95(B)", 0)
                metrics.precision = rd.get("metrics/precision(B)", 0)
                metrics.recall = rd.get("metrics/recall(B)", 0)
        except Exception:
            pass

        # Try reading results.csv
        csv_path = experiment_dir / "train" / "results.csv"
        if csv_path.exists():
            try:
                lines = csv_path.read_text().strip().split("\n")
                if len(lines) > 1:
                    headers = [h.strip() for h in lines[0].split(",")]
                    last = [v.strip() for v in lines[-1].split(",")]
                    row = dict(zip(headers, last))
                    metrics.epoch = int(float(row.get("epoch", 0)))
                    metrics.train_loss = float(row.get("train/box_loss", 0)) + float(row.get("train/cls_loss", 0))
                    metrics.val_loss = float(row.get("val/box_loss", 0)) + float(row.get("val/cls_loss", 0))
                    if not metrics.mAP50:
                        metrics.mAP50 = float(row.get("metrics/mAP50(B)", 0))
                    if not metrics.mAP50_95:
                        metrics.mAP50_95 = float(row.get("metrics/mAP50-95(B)", 0))
            except Exception as e:
                logger.debug("Failed to parse results.csv: %s", e)

        return metrics


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def train_yolo(config: TrainingConfig, **kwargs) -> dict:
    """Convenience function for YOLO training."""
    trainer = YOLOTrainer()
    return trainer.train(config, **kwargs)
