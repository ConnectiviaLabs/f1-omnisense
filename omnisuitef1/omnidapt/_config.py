"""omnidapt configuration — env-var driven with sensible defaults.

All config keys use the OMNIDAPT_ prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(f"OMNIDAPT_{key}", default)


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, str(default)).lower() in ("true", "1", "yes")


@dataclass
class OmnidaptConfig:
    # Storage
    base_dir: Path = field(default_factory=lambda: Path(_env("BASE_DIR", str(Path.home() / ".cache" / "omnidapt"))))
    use_mongo: bool = field(default_factory=lambda: _env_bool("USE_MONGO", False))

    # CV training defaults
    cv_epochs: int = field(default_factory=lambda: _env_int("CV_EPOCHS", 50))
    cv_batch_size: int = field(default_factory=lambda: _env_int("CV_BATCH_SIZE", 16))
    cv_image_size: int = field(default_factory=lambda: _env_int("CV_IMAGE_SIZE", 640))
    cv_base_model: str = field(default_factory=lambda: _env("CV_BASE_MODEL", "yolo26l.pt"))
    cv_learning_rate: float = field(default_factory=lambda: _env_float("CV_LEARNING_RATE", 0.01))
    cv_patience: int = field(default_factory=lambda: _env_int("CV_PATIENCE", 10))

    # Audio training defaults
    audio_epochs: int = field(default_factory=lambda: _env_int("AUDIO_EPOCHS", 20))
    audio_batch_size: int = field(default_factory=lambda: _env_int("AUDIO_BATCH_SIZE", 16))
    audio_base_model: str = field(default_factory=lambda: _env("AUDIO_BASE_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593"))
    audio_learning_rate: float = field(default_factory=lambda: _env_float("AUDIO_LEARNING_RATE", 5e-5))

    # Scheduler
    retrain_interval_hours: int = field(default_factory=lambda: _env_int("RETRAIN_INTERVAL", 24))
    perf_threshold: float = field(default_factory=lambda: _env_float("PERF_THRESHOLD", 0.7))
    max_model_age_days: int = field(default_factory=lambda: _env_int("MAX_MODEL_AGE", 30))

    # Server
    server_port: int = field(default_factory=lambda: _env_int("PORT", 8300))

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)

    @property
    def experiments_dir(self) -> Path:
        return self.base_dir / "experiments"

    @property
    def exports_dir(self) -> Path:
        return self.base_dir / "exports"

    @property
    def yolo_export_dir(self) -> Path:
        return self.base_dir / "exports" / "yolo"

    @property
    def audio_export_dir(self) -> Path:
        return self.base_dir / "exports" / "audio"

    @property
    def cv_models_dir(self) -> Path:
        return self.base_dir / "models" / "cv"

    @property
    def audio_models_dir(self) -> Path:
        return self.base_dir / "models" / "audio"

    @property
    def registry_dir(self) -> Path:
        return self.base_dir / "registry"

    @property
    def datasets_dir(self) -> Path:
        return self.base_dir / "datasets"

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for d in [
            self.experiments_dir,
            self.yolo_export_dir,
            self.audio_export_dir,
            self.cv_models_dir,
            self.audio_models_dir,
            self.registry_dir,
            self.datasets_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# ── Singleton ────────────────────────────────────────────────────────────

_config: OmnidaptConfig | None = None


def get_config() -> OmnidaptConfig:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        _config = OmnidaptConfig()
        _config.ensure_dirs()
    return _config


def reset_config() -> None:
    """Reset config singleton (for testing)."""
    global _config
    _config = None
