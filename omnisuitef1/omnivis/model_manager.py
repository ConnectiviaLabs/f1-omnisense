"""Singleton model registry — lazy-load, GPU-aware, auto-eviction.

Adapted from MediaSense ModelManager. Handles YOLO, RF-DETR, HuggingFace
transformers models, and SAHI sliced inference wrappers.

Usage:
    from omnisee.model_manager import get_model_manager

    manager = get_model_manager()
    model = manager.load_model("Detection")
    hf_model, proc = manager.load_hf_model("IDEA-Research/grounding-dino-base")
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import numpy as np

from omnivis._types import ModelType

logger = logging.getLogger(__name__)

# ── Lazy imports ──────────────────────────────────────────────────────────

_torch = None
_YOLO = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO
    return _YOLO


# ── Default model paths (relative to models_dir) ─────────────────────────

YOLO_MODELS: Dict[str, str] = {
    "Detection": "yolo26l.pt",
    "Segmentation": "yolo26l-seg.pt",
    "Pose": "yolo26l-pose.pt",
    "Classification": "yolo26l-cls.pt",
    "Track": "yolo26l.pt",
    "OBB": "yolo26l-obb.pt",
    "OpenVocab": "yolov8x-worldv2.pt",
    "Drone": "DroneDetection.pt",
    "DroneVision": "yolo11l-visdrone.pt",
    "Fish": "exp3-0-1_bs_best.pt",
    "Maritime": "Best_Maritime_Custom.pt",
}

TENSORRT_ENGINES: Dict[str, str] = {
    "Detection": "yolo26l.engine",
    "Segmentation": "yolo26l-seg.engine",
    "Pose": "yolo26l-pose.engine",
    "Classification": "yolo26l-cls.engine",
    "Track": "yolo26l.engine",
    "OBB": "yolo26l-obb.engine",
    "OpenVocab": "yolov8x-worldv2.engine",
    "Drone": "DetecDroneBest.engine",
    "DroneVision": "yolo11l-visdrone.engine",
}

HUGGINGFACE_FALLBACKS: Dict[str, Dict[str, str]] = {
    "Drone": {"repo_id": "MuayThaiLegz/DroneDetection", "filename": "weight/mediasense_best_model.pt"},
    "Maritime": {"repo_id": "MuayThaiLegz/Maritime_Custom", "filename": "best.pt"},
}

# RF-DETR threat config
THREAT_RESOLUTION = int(os.environ.get("OMNIDOC_THREAT_RESOLUTION", "960"))
THREAT_CONFIDENCE = float(os.environ.get("OMNIDOC_THREAT_CONF", "0.25"))

# Eviction
MODEL_TIMEOUT = int(os.environ.get("OMNIDOC_MODEL_TIMEOUT", "300"))
WARMUP_ENABLED = os.environ.get("OMNIDOC_MODEL_WARMUP", "true").lower() == "true"


class ModelManager:
    """Thread-safe singleton that lazy-loads, caches, and auto-evicts all models."""

    _instance: Optional[ModelManager] = None
    _instance_lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._models: Dict[str, Any] = {}
        self._hf_models: Dict[str, Any] = {}
        self._sahi_models: Dict[str, Any] = {}
        self._last_access: Dict[str, float] = {}
        self._model_lock = Lock()
        self._hf_lock = Lock()
        self._sahi_lock = Lock()

        # Configurable models directory
        default_dir = os.environ.get("OMNIDOC_MODELS_DIR", str(Path.home() / ".cache" / "omnidoc" / "models"))
        self.models_dir = Path(default_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Custom model registrations (user can add via register_model)
        self._custom_yolo: Dict[str, str] = {}

        self._initialized = True
        logger.info("ModelManager initialized (models_dir=%s)", self.models_dir)

    # ── Device helpers ────────────────────────────────────────────────────

    def get_device(self) -> str:
        torch = _get_torch()
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── YOLO / RF-DETR models ─────────────────────────────────────────────

    def load_model(self, model_type: str, *, force_reload: bool = False) -> Optional[Any]:
        """Load and cache a YOLO or RF-DETR model."""
        self._cleanup_expired()

        with self._model_lock:
            self._last_access[model_type] = time.time()

            if model_type in self._models and not force_reload:
                return self._models[model_type]

            try:
                t0 = time.perf_counter()

                if model_type == "Threat":
                    model = self._load_threat()
                else:
                    model = self._load_yolo(model_type)

                if model is None:
                    return None

                self._models[model_type] = model
                ms = (time.perf_counter() - t0) * 1000
                logger.info("Loaded %s in %.0fms", model_type, ms)

                if model_type != "Threat":
                    self._warmup(model)

                return model
            except Exception as e:
                logger.error("Failed to load %s: %s", model_type, e)
                return None

    def _load_yolo(self, model_type: str) -> Optional[Any]:
        YOLO = _get_yolo()
        filename = self._custom_yolo.get(model_type) or YOLO_MODELS.get(model_type)
        if not filename:
            logger.error("No model path for %s", model_type)
            return None

        full_path = self.models_dir / filename

        # TensorRT priority
        engine = TENSORRT_ENGINES.get(model_type)
        if engine:
            engine_path = self.models_dir / engine
            if engine_path.exists():
                logger.info("Loading TensorRT engine: %s", engine)
                return YOLO(str(engine_path))

        # Local weights
        if full_path.exists():
            return YOLO(str(full_path))

        # HuggingFace download
        if model_type in HUGGINGFACE_FALLBACKS:
            downloaded = self._download_hf_weights(model_type, full_path)
            if downloaded:
                return YOLO(downloaded)

        # Let ultralytics auto-download
        logger.info("Auto-downloading: %s", filename)
        return YOLO(filename)

    def _load_threat(self) -> Optional[Any]:
        try:
            from rfdetr import RFDETRNano
        except ImportError:
            logger.error("rfdetr not installed — cannot load Threat model")
            return None

        weights = self.models_dir / "checkpoint_best_total.pth"
        if not weights.exists():
            try:
                import requests
                url = "https://huggingface.co/Subh775/Threat-Detection-RFDETR/resolve/main/checkpoint_best_total.pth"
                logger.info("Downloading RF-DETR weights...")
                resp = requests.get(url, stream=True)
                resp.raise_for_status()
                with open(weights, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
            except Exception as e:
                logger.error("RF-DETR download failed: %s", e)
                return None

        model = RFDETRNano(resolution=THREAT_RESOLUTION, pretrain_weights=str(weights))
        model.optimize_for_inference()
        return model

    def _download_hf_weights(self, model_type: str, target: Path) -> Optional[str]:
        cfg = HUGGINGFACE_FALLBACKS.get(model_type)
        if not cfg or not cfg.get("repo_id"):
            return None
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"],
                                   local_dir=str(self.models_dir), local_dir_use_symlinks=False)
        except Exception as e:
            logger.warning("HF download for %s failed: %s", model_type, e)
            return None

    # ── HuggingFace transformers models ───────────────────────────────────

    def load_hf_model(self, model_id: str, *, task: str = "auto") -> Any:
        """Load and cache a HuggingFace transformers model.

        Returns (model, processor) tuple or just model depending on task.
        """
        with self._hf_lock:
            self._last_access[f"hf:{model_id}"] = time.time()
            if model_id in self._hf_models:
                return self._hf_models[model_id]

            torch = _get_torch()
            device = self.get_device()

            try:
                t0 = time.perf_counter()

                if "grounding-dino" in model_id.lower():
                    result = self._load_grounding_dino(model_id, device)
                elif "timesformer" in model_id.lower():
                    result = self._load_timesformer(model_id, device)
                elif "videomae" in model_id.lower():
                    result = self._load_videomae(model_id, device)
                elif "sam2" in model_id.lower():
                    result = self._load_sam2(model_id, device)
                elif "ast" in model_id.lower() or "drone-audio" in model_id.lower():
                    result = self._load_audio_classifier(model_id, device)
                else:
                    from transformers import AutoModel, AutoProcessor
                    processor = AutoProcessor.from_pretrained(model_id)
                    model = AutoModel.from_pretrained(model_id).to(device).eval()
                    result = (model, processor)

                self._hf_models[model_id] = result
                ms = (time.perf_counter() - t0) * 1000
                logger.info("Loaded HF model %s in %.0fms", model_id, ms)
                return result

            except Exception as e:
                logger.error("Failed to load HF model %s: %s", model_id, e)
                raise

    def _load_grounding_dino(self, model_id: str, device: str):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        return (model, processor)

    def _load_timesformer(self, model_id: str, device: str):
        from transformers import AutoImageProcessor, TimesformerForVideoClassification
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = TimesformerForVideoClassification.from_pretrained(model_id).to(device).eval()
        return (model, processor)

    def _load_videomae(self, model_id: str, device: str):
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
        processor = VideoMAEImageProcessor.from_pretrained(model_id)
        model = VideoMAEForVideoClassification.from_pretrained(model_id).to(device).eval()
        return (model, processor)

    def _load_sam2(self, model_id: str, device: str):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            weights = self.models_dir / "sam2.1_hiera_large.pt"
            config = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2 = build_sam2(config, str(weights), device=device)
            predictor = SAM2ImagePredictor(sam2)
            return predictor
        except ImportError:
            logger.warning("sam2 package not installed, trying transformers SAM")
            from transformers import SamModel, SamProcessor
            processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
            model = SamModel.from_pretrained("facebook/sam-vit-large").to(device).eval()
            return (model, processor)

    def _load_audio_classifier(self, model_id: str, device: str):
        from transformers import pipeline as hf_pipeline
        torch = _get_torch()
        device_id = 0 if "cuda" in device else -1
        classifier = hf_pipeline("audio-classification", model=model_id, device=device_id,
                                 model_kwargs={"torch_dtype": torch.float32})
        return classifier

    # ── SAHI sliced inference ─────────────────────────────────────────────

    def get_sahi_model(self, model_path: str, confidence: float = 0.25) -> Optional[Any]:
        """Get or create a SAHI AutoDetectionModel for sliced inference."""
        try:
            from sahi import AutoDetectionModel
        except ImportError:
            logger.warning("sahi not installed — sliced inference disabled")
            return None

        cache_key = f"{model_path}_{confidence}"
        if cache_key in self._sahi_models:
            return self._sahi_models[cache_key]

        with self._sahi_lock:
            if cache_key in self._sahi_models:
                return self._sahi_models[cache_key]
            try:
                sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8", model_path=model_path,
                    confidence_threshold=confidence, device=self.get_device(),
                )
                self._sahi_models[cache_key] = sahi_model
                logger.info("SAHI model loaded: %s (conf=%.2f)", model_path, confidence)
                return sahi_model
            except Exception as e:
                logger.warning("SAHI model load failed: %s", e)
                return None

    # ── Registration / lifecycle ──────────────────────────────────────────

    def register_model(self, name: str, path: str):
        """Register a custom YOLO model path."""
        self._custom_yolo[name] = path

    def unload_model(self, model_type: str):
        with self._model_lock:
            if model_type in self._models:
                del self._models[model_type]
                self._last_access.pop(model_type, None)
                self._free_gpu()
                logger.info("Unloaded %s", model_type)

    def unload_hf_model(self, model_id: str):
        with self._hf_lock:
            key = f"hf:{model_id}"
            if model_id in self._hf_models:
                del self._hf_models[model_id]
                self._last_access.pop(key, None)
                self._free_gpu()

    def loaded_models(self) -> list[str]:
        return list(self._models.keys()) + [k for k in self._hf_models]

    # ── Internal ──────────────────────────────────────────────────────────

    def _warmup(self, model, imgsz: int = 640, iterations: int = 2):
        if not WARMUP_ENABLED:
            return
        try:
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            for _ in range(iterations):
                model(dummy, verbose=False)
        except Exception:
            pass

    def _cleanup_expired(self):
        now = time.time()
        expired = [k for k, t in self._last_access.items() if now - t > MODEL_TIMEOUT]
        if not expired:
            return
        with self._model_lock:
            for k in expired:
                self._models.pop(k, None)
                self._last_access.pop(k, None)
        with self._hf_lock:
            for k in expired:
                if k.startswith("hf:"):
                    self._hf_models.pop(k[3:], None)
        if expired:
            self._free_gpu()
            logger.info("Evicted unused models: %s", expired)

    def _free_gpu(self):
        gc.collect()
        torch = _get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Module-level singleton ────────────────────────────────────────────────

_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
