"""FastAPI REST API for omnidapt — datasets, training, models, scheduler.

Run: uvicorn omnidapt.server:app --port 8300
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="omnidapt", description="Continuous Adaptation Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ─────────────────────────────────────────────────────

class DatasetCreate(BaseModel):
    name: str
    domain: str
    description: str = ""
    labels: List[str] = []


class FrameData(BaseModel):
    image_data: str
    label: str = ""
    bbox: Optional[List[float]] = None
    width: int = 0
    height: int = 0
    format: str = "jpg"


class AudioChunkData(BaseModel):
    audio_data: str
    label: str = ""
    duration: float = 0
    sample_rate: int = 16000
    format: str = "wav"


class TrainRequest(BaseModel):
    domain: str
    dataset_name: str
    model_name: str = ""
    base_model: str = ""
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0
    image_size: int = 640
    patience: int = 10
    device: str = ""


class PromoteRequest(BaseModel):
    to_status: str


# ── Lazy singletons ─────────────────────────────────────────────────────

def _dm():
    from omnidapt.dataset_manager import get_dataset_manager
    return get_dataset_manager()


def _registry():
    from omnidapt.model_registry import get_model_registry
    return get_model_registry()


def _jobs():
    from omnidapt.jobs import get_job_manager
    return get_job_manager()


def _scheduler():
    from omnidapt.scheduler import get_scheduler
    return get_scheduler()


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "omnidapt"}


# ── Datasets ─────────────────────────────────────────────────────────────

@app.get("/datasets")
async def list_datasets(domain: Optional[str] = None):
    from omnidapt._types import ModelDomain
    d = ModelDomain(domain) if domain else None
    return [ds.to_dict() for ds in _dm().list_datasets(d)]


@app.post("/datasets")
async def create_dataset(body: DatasetCreate):
    ds = _dm().create_dataset(body.name, body.domain, body.description, body.labels)
    return ds.to_dict()


@app.get("/datasets/{name}")
async def get_dataset(name: str):
    ds = _dm().get_dataset(name)
    if not ds:
        raise HTTPException(404, f"Dataset '{name}' not found")
    return ds.to_dict()


@app.delete("/datasets/{name}")
async def delete_dataset(name: str):
    _dm().delete_dataset(name)
    return {"deleted": name}


@app.post("/datasets/{name}/frames")
async def add_frames(name: str, frames: List[FrameData]):
    count = _dm().add_frames(name, [f.model_dump() for f in frames])
    return {"added": count, "dataset": name}


@app.post("/datasets/{name}/audio-chunks")
async def add_audio_chunks(name: str, chunks: List[AudioChunkData]):
    count = _dm().add_audio_chunks(name, [c.model_dump() for c in chunks])
    return {"added": count, "dataset": name}


@app.get("/datasets/{name}/samples")
async def get_samples(name: str, limit: int = 50, offset: int = 0):
    return _dm().get_samples(name, limit, offset)


@app.get("/datasets/{name}/export")
async def export_dataset(name: str, format: str = "yolo"):
    from fastapi.responses import FileResponse
    try:
        zip_path = _dm().export_zip(name)
        return FileResponse(str(zip_path), filename=f"{name}.zip", media_type="application/zip")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/datasets/{name}/stats")
async def get_stats(name: str):
    try:
        return _dm().get_stats(name).to_dict()
    except ValueError as e:
        raise HTTPException(404, str(e))


# ── Training ─────────────────────────────────────────────────────────────

@app.post("/train")
async def start_training(body: TrainRequest):
    from omnidapt._config import get_config
    from omnidapt._types import ModelDomain, TrainingConfig

    cfg = get_config()
    domain = ModelDomain(body.domain)

    config = TrainingConfig(
        domain=domain,
        dataset_name=body.dataset_name,
        model_name=body.model_name or body.dataset_name,
        base_model=body.base_model,
        epochs=body.epochs or (cfg.cv_epochs if domain == ModelDomain.CV else cfg.audio_epochs),
        batch_size=body.batch_size or (cfg.cv_batch_size if domain == ModelDomain.CV else cfg.audio_batch_size),
        learning_rate=body.learning_rate or (cfg.cv_learning_rate if domain == ModelDomain.CV else cfg.audio_learning_rate),
        image_size=body.image_size,
        patience=body.patience,
        device=body.device,
    )

    if domain == ModelDomain.CV:
        from omnidapt.yolo_trainer import train_yolo
        func = train_yolo
    else:
        from omnidapt.ast_trainer import train_ast
        func = train_ast

    job_id = _jobs().submit(func, config=config, kwargs={"config": config})
    return {"job_id": job_id, "config": config.to_dict()}


@app.get("/train/jobs")
async def list_jobs(status: Optional[str] = None):
    from omnidapt._types import TrainingStatus
    s = TrainingStatus(status) if status else None
    return [j.to_dict() for j in _jobs().list_jobs(s)]


@app.get("/train/jobs/{job_id}")
async def get_job(job_id: str):
    job = _jobs().get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return job.to_dict()


@app.post("/train/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    success = _jobs().cancel(job_id)
    return {"cancelled": success, "job_id": job_id}


# ── Models ───────────────────────────────────────────────────────────────

@app.get("/models")
async def list_models():
    docs = _registry()._storage.find_documents("model_versions")
    names = set(d["name"] for d in docs)
    result = []
    for name in sorted(names):
        prod = _registry().get_production(name)
        versions = _registry().list_versions(name)
        result.append({
            "name": name,
            "production_version": prod.version if prod else None,
            "total_versions": len(versions),
        })
    return result


@app.get("/models/{name}")
async def get_model(name: str):
    versions = _registry().list_versions(name)
    if not versions:
        raise HTTPException(404, f"Model '{name}' not found")
    return [v.to_dict() for v in versions]


@app.post("/models/{name}/{version}/promote")
async def promote_model(name: str, version: str, body: PromoteRequest):
    from omnidapt._types import ModelStatus
    success = _registry().promote(name, version, ModelStatus(body.to_status))
    if not success:
        raise HTTPException(400, "Promotion failed — invalid transition or not found")
    return {"promoted": True, "name": name, "version": version, "to": body.to_status}


@app.post("/models/{name}/rollback")
async def rollback_model(name: str):
    result = _registry().rollback(name)
    if not result:
        raise HTTPException(400, "Rollback failed — no validated version")
    return {"rolled_back": True, "new_production": result.version}


@app.get("/models/{name}/metrics")
async def get_metrics(name: str):
    return _registry().get_metrics_history(name)


@app.post("/models/{name}/register-omnivis")
async def register_omnivis(name: str):
    success = _registry().register_with_omnivis(name)
    return {"registered": success, "name": name}


# ── Scheduler ────────────────────────────────────────────────────────────

@app.get("/scheduler/status")
async def scheduler_status():
    return _scheduler().get_status()


@app.post("/scheduler/status")
async def toggle_scheduler(action: str = "start"):
    s = _scheduler()
    if action == "start":
        await s.start()
    elif action == "stop":
        await s.stop()
    return s.get_status()


@app.post("/scheduler/trigger")
async def trigger_training(body: TrainRequest):
    from omnidapt._types import ModelDomain, TrainingConfig
    config = TrainingConfig(
        domain=ModelDomain(body.domain),
        dataset_name=body.dataset_name,
        model_name=body.model_name or body.dataset_name,
    )
    _scheduler().request_training(config)
    return {"triggered": True, "dataset": body.dataset_name}


@app.get("/scheduler/health")
async def scheduler_health():
    reports = _scheduler().check_all_health()
    return [r.to_dict() for r in reports]
