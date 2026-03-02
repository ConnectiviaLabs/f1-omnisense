"""AST (Audio Spectrogram Transformer) fine-tuning pipeline.

Uses HuggingFace Trainer with the MIT/ast-finetuned-audioset model.
"""

from __future__ import annotations

import json
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


class ASTTrainer:
    """Audio Spectrogram Transformer fine-tuning trainer."""

    def __init__(self):
        self._config = get_config()

    def train(
        self,
        config: TrainingConfig,
        job_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        """Run AST fine-tuning.

        1. Export dataset to HuggingFace format
        2. Build HF Dataset from WAV + labels
        3. Train with HuggingFace Trainer
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
        _progress(5, "export", f"Exporting {config.dataset_name} to HuggingFace format")
        try:
            export_dir = dm.export_huggingface_audio(config.dataset_name)
        except Exception as e:
            raise RuntimeError(f"Dataset export failed: {e}") from e

        metadata_path = export_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {export_dir}")

        metadata = json.loads(metadata_path.read_text())
        if not metadata:
            raise ValueError("Empty metadata — no audio samples")

        _progress(10, "setup", "Building HuggingFace dataset")

        # 2. Build HF Dataset
        import numpy as np
        from datasets import Dataset, DatasetDict
        from transformers import ASTFeatureExtractor

        base_model = config.base_model or self._config.audio_base_model
        feature_extractor = ASTFeatureExtractor.from_pretrained(base_model)

        # Build label map
        labels = sorted(set(m["label"] for m in metadata if m.get("label")))
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for l, i in label2id.items()}

        # Load audio files
        import soundfile as sf

        audio_data = []
        for entry in metadata:
            audio_path = export_dir / entry["file_name"]
            if not audio_path.exists():
                continue
            try:
                waveform, sr = sf.read(str(audio_path))
                if len(waveform.shape) > 1:
                    waveform = waveform.mean(axis=1)  # mono
                audio_data.append({
                    "audio": waveform.tolist(),
                    "sampling_rate": sr,
                    "label": label2id.get(entry.get("label", ""), 0),
                })
            except Exception as e:
                logger.debug("Failed to load %s: %s", audio_path, e)

        if not audio_data:
            raise ValueError("No valid audio files loaded")

        _progress(20, "setup", f"Loaded {len(audio_data)} audio samples, {len(labels)} classes")

        # Create dataset with train/eval split
        ds = Dataset.from_list(audio_data)
        split = ds.train_test_split(test_size=0.2, seed=42)
        dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})

        # Preprocess
        def preprocess(batch):
            inputs = feature_extractor(
                batch["audio"],
                sampling_rate=batch["sampling_rate"][0] if isinstance(batch["sampling_rate"], list) else batch["sampling_rate"],
                return_tensors="np",
                padding=True,
            )
            batch["input_values"] = inputs["input_values"]
            return batch

        _progress(30, "preprocessing", "Feature extraction")
        dataset_dict = dataset_dict.map(preprocess, batched=True, batch_size=8)

        if cancel_event and cancel_event.is_set():
            return {"cancelled": True}

        # 3. Train
        _progress(35, "training", f"Starting AST training ({config.epochs} epochs)")

        from transformers import (
            ASTForAudioClassification,
            Trainer,
            TrainingArguments,
        )

        model = ASTForAudioClassification.from_pretrained(
            base_model,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )

        output_dir = self._config.experiments_dir / f"{config.model_name}_ast_{int(time.time())}"

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate or self._config.audio_learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,
            remove_unused_columns=False,
        )

        def compute_metrics(eval_pred):
            from sklearn.metrics import accuracy_score, f1_score
            logits, labels_arr = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels_arr, preds),
                "f1": f1_score(labels_arr, preds, average="weighted"),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            compute_metrics=compute_metrics,
        )

        trainer.train()

        if cancel_event and cancel_event.is_set():
            return {"cancelled": True}

        _progress(85, "evaluating", "Running evaluation")

        # 4. Evaluate
        eval_results = trainer.evaluate()

        metrics = TrainingMetrics(
            epoch=config.epochs,
            accuracy=eval_results.get("eval_accuracy", 0),
            f1_score=eval_results.get("eval_f1", 0),
            val_loss=eval_results.get("eval_loss", 0),
        )

        # Save model
        _progress(90, "saving", "Saving model")
        model_dir = output_dir / "best_model"
        trainer.save_model(str(model_dir))
        feature_extractor.save_pretrained(str(model_dir))

        # Save label map
        (model_dir / "label_map.json").write_text(json.dumps({"label2id": label2id, "id2label": id2label}, indent=2))

        # 5. Register
        _progress(95, "registering", "Registering model version")
        mv = registry.register(
            name=config.model_name,
            domain=ModelDomain.AUDIO,
            artifact_path=model_dir,
            metrics=metrics,
            base_model=base_model,
            dataset_name=config.dataset_name,
        )

        _progress(100, "complete", f"Training complete — v{mv.version}")

        return {
            "metrics": metrics.to_dict(),
            "result_path": str(model_dir),
            "version": mv.version,
            "experiment_dir": str(output_dir),
            "label_map": label2id,
        }


def train_ast(config: TrainingConfig, **kwargs) -> dict:
    """Convenience function for AST training."""
    trainer = ASTTrainer()
    return trainer.train(config, **kwargs)
