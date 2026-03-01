"""3D model generation orchestrator for F1.

Coordinates: uploaded image -> Meshy / Tripo / TRELLIS / Hunyuan -> GLB + metadata.
After generation, pushes GLB files to MongoDB GridFS for Vercel serving.
Manages background jobs with threading.
"""

import os
import json
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Output directory for generated 3D models
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_3D_DIR = _PROJECT_ROOT / "output" / "3d_models"

# In-memory job store
_3d_jobs = {}


def generate_3d_hunyuan(model_name: str, image_path: Path,
                         output_dir: Path = None, **kwargs) -> dict:
    """Generate 3D model via Hunyuan3D.

    NOTE: Currently disabled in routing — requires GCP VM + bpy.
    Code kept for future re-enablement.

    Args:
        model_name: Model identifier.
        image_path: Input image path.
        output_dir: Output directory. Defaults to 3d_models/{model_name}/.
        **kwargs: Hunyuan params — textured (bool), steps, guidance_scale,
                  seed, octree_resolution, remove_bg, num_chunks.

    Returns:
        Dict with output paths and metadata.
    """
    from .hunyuan_hf_client import get_hunyuan_hf_client as get_hunyuan_client

    if output_dir is None:
        output_dir = MODELS_3D_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    textured = kwargs.pop("textured", False)
    gen_params = {
        k: v for k, v in kwargs.items()
        if k in ("steps", "guidance_scale", "seed", "octree_resolution",
                  "remove_bg", "num_chunks", "randomize_seed")
    }

    client = get_hunyuan_client()
    start_time = time.time()

    if textured:
        result = client.generate_textured(
            image_path=image_path,
            output_shape_path=output_dir / "hunyuan.glb",
            output_textured_path=output_dir / "hunyuan_textured.glb",
            **gen_params,
        )
        glb_path = result.glb_path
        textured_glb = result.textured_glb_path
    else:
        result = client.generate(
            image_path=image_path,
            output_path=output_dir / "hunyuan.glb",
            **gen_params,
        )
        glb_path = result.glb_path
        textured_glb = None

    elapsed = time.time() - start_time

    metadata = {
        "model_name": model_name,
        "provider": "hunyuan",
        "mode": "textured" if textured else "shape",
        "input_image": str(image_path.name),
        "output_glb": "hunyuan.glb",
        "output_textured_glb": "hunyuan_textured.glb" if textured_glb else None,
        "mesh_stats": result.mesh_stats,
        "seed": result.seed,
        "file_size_bytes": glb_path.stat().st_size if glb_path.exists() else 0,
        "generation_time_seconds": round(elapsed, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "parameters": gen_params,
    }
    _save_metadata(output_dir, metadata)
    return metadata


def generate_3d_meshy(model_name: str, image_path: Path,
                       output_dir: Path = None, progress_callback=None,
                       **kwargs) -> dict:
    """Generate 3D model via Meshy.ai.

    Args:
        model_name: Model identifier.
        image_path: Input image path.
        output_dir: Output directory.
        progress_callback: Optional fn(status, progress_pct).
        **kwargs: Meshy params (texture_prompt, enable_pbr, target_polycount).

    Returns:
        Dict with output paths and metadata.
    """
    from .meshy_client import get_meshy_client

    if output_dir is None:
        output_dir = MODELS_3D_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_meshy_client()
    start_time = time.time()

    task_id = client.create_task(
        image_path=image_path,
        texture_prompt=kwargs.get("texture_prompt", ""),
        enable_pbr=kwargs.get("enable_pbr", True),
        topology=kwargs.get("topology", "triangle"),
        target_polycount=kwargs.get("target_polycount", 30000),
    )

    glb_path = output_dir / "meshy.glb"
    client.wait_and_download(
        task_id=task_id,
        output_path=glb_path,
        timeout=kwargs.get("timeout", 600),
        poll_interval=kwargs.get("poll_interval", 5),
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time

    metadata = {
        "model_name": model_name,
        "provider": "meshy",
        "input_image": str(image_path.name),
        "output_glb": "meshy.glb",
        "meshy_task_id": task_id,
        "file_size_bytes": glb_path.stat().st_size if glb_path.exists() else 0,
        "generation_time_seconds": round(elapsed, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "parameters": {k: v for k, v in kwargs.items()},
    }

    # Append to meshy_regenerations list
    meta_path = output_dir / "metadata.json"
    existing = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    regens = existing.get("meshy_regenerations", [])
    regens.append(metadata)
    existing["meshy_regenerations"] = regens
    existing["latest_meshy"] = metadata
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)

    return metadata


def generate_3d_tripo(model_name: str, image_path: Path,
                       output_dir: Path = None, progress_callback=None,
                       **kwargs) -> dict:
    """Generate 3D model via Tripo AI.

    Args:
        model_name: Model identifier.
        image_path: Input image path.
        output_dir: Output directory.
        progress_callback: Optional fn(status, progress_pct).
        **kwargs: Tripo params (face_limit, texture, pbr).

    Returns:
        Dict with output paths and metadata.
    """
    from .tripo_client import get_tripo_client

    if output_dir is None:
        output_dir = MODELS_3D_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_tripo_client()
    start_time = time.time()

    task_id = client.create_task(
        image_path=image_path,
        texture=kwargs.get("texture", True),
        pbr=kwargs.get("pbr", True),
        face_limit=kwargs.get("face_limit"),
    )

    glb_path = output_dir / "tripo.glb"
    client.wait_and_download(
        task_id=task_id,
        output_path=glb_path,
        timeout=kwargs.get("timeout", 600),
        poll_interval=kwargs.get("poll_interval", 5),
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time

    metadata = {
        "model_name": model_name,
        "provider": "tripo",
        "input_image": str(image_path.name),
        "output_glb": "tripo.glb",
        "tripo_task_id": task_id,
        "file_size_bytes": glb_path.stat().st_size if glb_path.exists() else 0,
        "generation_time_seconds": round(elapsed, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "parameters": {k: v for k, v in kwargs.items()},
    }
    _save_metadata(output_dir, metadata)
    return metadata


def generate_3d_trellis(model_name: str, image_path: Path,
                         output_dir: Path = None, progress_callback=None,
                         **kwargs) -> dict:
    """Generate 3D model via TRELLIS (Microsoft) on fal.ai.

    Args:
        model_name: Model identifier.
        image_path: Input image path.
        output_dir: Output directory.
        progress_callback: Optional fn(status, progress_pct).
        **kwargs: TRELLIS params (mesh_simplify, texture_size, seed).

    Returns:
        Dict with output paths and metadata.
    """
    from .trellis_client import get_trellis_client

    if output_dir is None:
        output_dir = MODELS_3D_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_trellis_client()
    start_time = time.time()

    request_id = client.create_task(
        image_path=image_path,
        mesh_simplify=kwargs.get("mesh_simplify", 0.95),
        texture_size=kwargs.get("texture_size", 1024),
        seed=kwargs.get("seed", 42),
    )

    glb_path = output_dir / "trellis.glb"
    client.wait_and_download(
        request_id=request_id,
        output_path=glb_path,
        timeout=kwargs.get("timeout", 600),
        poll_interval=kwargs.get("poll_interval", 5),
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time

    metadata = {
        "model_name": model_name,
        "provider": "trellis",
        "input_image": str(image_path.name),
        "output_glb": "trellis.glb",
        "trellis_request_id": request_id,
        "file_size_bytes": glb_path.stat().st_size if glb_path.exists() else 0,
        "generation_time_seconds": round(elapsed, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "parameters": {k: v for k, v in kwargs.items()},
    }
    _save_metadata(output_dir, metadata)
    return metadata


def generate_3d_texture_paint(model_name: str, output_dir: Path = None,
                               **kwargs) -> dict:
    """Generate textured 3D model via TEXTure engine (local GPU painting).

    Args:
        model_name: Model identifier.
        output_dir: Output directory.
        **kwargs: TEXTure params (preset, prompt, n_views, texture_resolution, seed).

    Returns:
        Dict with output paths and metadata.
    """
    from .texture_paint import paint_texture

    if output_dir is None:
        output_dir = MODELS_3D_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    result = paint_texture(
        model_name=model_name,
        preset=kwargs.get("material_preset") or kwargs.get("preset"),
        prompt=kwargs.get("prompt"),
        n_views=kwargs.get("n_views", 8),
        texture_resolution=kwargs.get("texture_resolution", 1024),
        seed=kwargs.get("seed", 42),
    )

    elapsed = time.time() - start_time

    glb_path = Path(result["glb_path"])
    metadata = {
        "model_name": model_name,
        "provider": "texture_paint",
        "output_glb": "texture_paint.glb",
        "prompt": result.get("prompt"),
        "file_size_bytes": int(glb_path.stat().st_size) if glb_path.exists() else 0,
        "generation_time_seconds": round(elapsed, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
    }

    meta_path = output_dir / "metadata.json"
    existing = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    existing["texture_paint"] = metadata
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)

    return metadata


# --- Job Management ---

def submit_job(model_name: str, provider: str = "meshy",
               image_path: str = None, **params) -> str:
    """Submit a background 3D generation job.

    Args:
        model_name: Model identifier.
        provider: 'meshy', 'tripo', 'trellis', or 'texture_paint'.
        image_path: Path to uploaded input image.
        **params: Provider-specific parameters.

    Returns:
        Job ID string.
    """
    job_id = str(uuid.uuid4())[:8]

    _3d_jobs[job_id] = {
        "job_id": job_id,
        "model_name": model_name,
        "provider": provider,
        "status": "queued",
        "progress": 0,
        "glb_url": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }

    img_path = Path(image_path) if image_path else None

    thread = threading.Thread(
        target=_run_generation,
        args=(job_id, model_name, provider, img_path, params),
        daemon=True,
    )
    thread.start()

    return job_id


def _run_generation(job_id: str, model_name: str, provider: str,
                    image_path: Path, params: dict):
    """Background worker for 3D generation."""
    try:
        material_preset = params.pop("material_preset", None)

        _3d_jobs[job_id]["status"] = "generating"
        _3d_jobs[job_id]["progress"] = 20

        output_dir = MODELS_3D_DIR / model_name

        if image_path is None or not image_path.exists():
            # Check for existing input image
            existing_render = output_dir / "input_image.png"
            if not existing_render.exists():
                raise FileNotFoundError(
                    f"No input image found for model '{model_name}'. Upload an image first."
                )
            image_path = existing_render

        def meshy_progress(status, pct):
            _3d_jobs[job_id]["progress"] = 20 + int(pct * 0.7)

        if provider == "hunyuan":
            result = generate_3d_hunyuan(model_name, image_path, output_dir, **params)
            _3d_jobs[job_id]["progress"] = 90
        elif provider == "meshy":
            result = generate_3d_meshy(
                model_name, image_path, output_dir,
                progress_callback=meshy_progress, **params
            )
        elif provider == "tripo":
            result = generate_3d_tripo(
                model_name, image_path, output_dir,
                progress_callback=meshy_progress, **params
            )
        elif provider == "trellis":
            result = generate_3d_trellis(
                model_name, image_path, output_dir,
                progress_callback=meshy_progress, **params
            )
        elif provider == "texture_paint":
            _3d_jobs[job_id]["status"] = "painting_texture"
            _3d_jobs[job_id]["progress"] = 20
            result = generate_3d_texture_paint(model_name, output_dir, **params)
            _3d_jobs[job_id]["progress"] = 90
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Apply PBR material for Hunyuan shape-only mode
        if provider == "hunyuan" and not params.get("textured"):
            _3d_jobs[job_id]["status"] = "applying_material"
            _3d_jobs[job_id]["progress"] = 92
            try:
                from .apply_pbr import apply_texture_to_model
                pbr_result = apply_texture_to_model(
                    model_name,
                    preset=material_preset if material_preset else "carbon_fiber",
                )
                logger.info("PBR material applied: %s -> %s",
                            model_name, pbr_result.get("preset"))
            except Exception as e:
                logger.warning("PBR application skipped: %s", e)

        # Push generated GLBs to MongoDB GridFS for Vercel serving
        _3d_jobs[job_id]["progress"] = 95
        try:
            pushed = _push_job_glbs_to_gridfs(model_name, provider, output_dir)
            if pushed:
                _3d_jobs[job_id]["gridfs_files"] = pushed
        except Exception as e:
            logger.warning("GridFS push failed (non-fatal): %s", e)

        _3d_jobs[job_id]["status"] = "completed"
        _3d_jobs[job_id]["progress"] = 100
        _3d_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        _3d_jobs[job_id]["glb_url"] = f"/api/3d-gen/models/{model_name}/glb?provider={provider}"

        logger.info("3D generation completed: %s (%s)", model_name, provider)

    except Exception as e:
        logger.error("3D generation failed for %s: %s", model_name, e)
        _3d_jobs[job_id]["status"] = "failed"
        _3d_jobs[job_id]["error"] = str(e)


def get_job(job_id: str) -> dict | None:
    """Get job status."""
    return _3d_jobs.get(job_id)


def list_jobs() -> list:
    """List all jobs."""
    return list(_3d_jobs.values())


def list_generated_models() -> list:
    """List generated models from local filesystem + MongoDB."""
    seen = set()
    models = []

    # 1. Local filesystem
    if MODELS_3D_DIR.exists():
        for model_dir in sorted(MODELS_3D_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            meta_path = model_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["model_name"] = model_dir.name
                meta["directory"] = str(model_dir)
            elif list(model_dir.glob("*.glb")):
                meta = {"model_name": model_dir.name, "directory": str(model_dir)}
            else:
                continue
            meta["has_hunyuan"] = (model_dir / "hunyuan.glb").exists()
            meta["has_meshy"] = (model_dir / "meshy.glb").exists()
            meta["has_pbr"] = (model_dir / "hunyuan_pbr.glb").exists()
            meta["has_texture_paint"] = (model_dir / "texture_paint.glb").exists()
            meta["has_tripo"] = (model_dir / "tripo.glb").exists()
            meta["has_trellis"] = (model_dir / "trellis.glb").exists()
            seen.add(model_dir.name)
            models.append(meta)

    # 2. MongoDB — models that only exist in GridFS (e.g. after redeploy)
    try:
        from pymongo import MongoClient
        uri = os.environ.get("MONGODB_URI", "")
        if uri:
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            db = client[os.environ.get("MONGODB_DB", "marip_f1")]
            for doc in db["generated_models"].find({}, {"_id": 0}):
                name = doc.get("model_name")
                if name and name not in seen:
                    models.append(doc)
            client.close()
    except Exception as e:
        logger.debug("MongoDB model list unavailable: %s", e)

    return models


def get_mesh_quality(model_name: str) -> dict:
    """Get mesh quality info for generated GLBs."""
    result = {
        "model_name": model_name,
        "generated": [],
    }

    model_dir = MODELS_3D_DIR / model_name
    for glb_name in ("hunyuan.glb", "hunyuan_pbr.glb", "hunyuan_textured.glb", "meshy.glb", "tripo.glb", "trellis.glb", "texture_paint.glb"):
        glb_path = model_dir / glb_name
        if not glb_path.exists():
            continue
        try:
            import trimesh
            mesh = trimesh.load(str(glb_path), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            bbox = mesh.bounding_box.extents
            result["generated"].append({
                "file": glb_name,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "bbox": [round(float(bbox[0]), 1), round(float(bbox[1]), 1), round(float(bbox[2]), 1)],
                "file_size_bytes": int(glb_path.stat().st_size),
            })
        except Exception as e:
            logger.warning("Could not load %s: %s", glb_name, e)
            result["generated"].append({
                "file": glb_name,
                "file_size_bytes": int(glb_path.stat().st_size),
            })

    return result


def _push_glb_to_gridfs(glb_path: Path, gridfs_filename: str) -> bool:
    """Upload a GLB file to MongoDB GridFS for Vercel serving.

    Args:
        glb_path: Local path to the GLB file.
        gridfs_filename: Filename to store in GridFS (e.g. 'image3_hunyuan.glb').

    Returns:
        True if upload succeeded, False otherwise.
    """
    try:
        from pymongo import MongoClient
        import gridfs

        uri = os.environ.get("MONGODB_URI", "")
        if not uri:
            logger.warning("MONGODB_URI not set — skipping GridFS push")
            return False

        db_name = os.environ.get("MONGODB_DB", "marip_f1")
        client = MongoClient(uri)
        db = client[db_name]
        fs = gridfs.GridFS(db)

        # Delete existing file with same name to avoid duplicates
        existing = fs.find_one({"filename": gridfs_filename})
        if existing:
            fs.delete(existing._id)
            logger.info("Replaced existing GridFS file: %s", gridfs_filename)

        with open(glb_path, "rb") as f:
            fs.put(
                f,
                filename=gridfs_filename,
                content_type="model/gltf-binary",
                metadata={
                    "source": "3d-generation",
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        size_mb = glb_path.stat().st_size / (1024 * 1024)
        logger.info("Pushed to GridFS: %s (%.1f MB)", gridfs_filename, size_mb)
        client.close()
        return True

    except ImportError:
        logger.warning("pymongo/gridfs not installed — skipping GridFS push")
        return False
    except Exception as e:
        logger.warning("GridFS push failed for %s: %s", gridfs_filename, e)
        return False


def _push_job_glbs_to_gridfs(model_name: str, provider: str, output_dir: Path):
    """Push all GLB files from a completed job to GridFS.

    Naming convention: {model_name}_{variant}.glb
    e.g. image3_hunyuan.glb, image3_hunyuan_pbr.glb
    """
    glb_variants = {
        "hunyuan.glb": f"{model_name}_hunyuan.glb",
        "hunyuan_pbr.glb": f"{model_name}_hunyuan_pbr.glb",
        "hunyuan_textured.glb": f"{model_name}_hunyuan_textured.glb",
        "meshy.glb": f"{model_name}_meshy.glb",
        "tripo.glb": f"{model_name}_tripo.glb",
        "trellis.glb": f"{model_name}_trellis.glb",
        "texture_paint.glb": f"{model_name}_texture_paint.glb",
    }

    pushed = []
    for local_name, gridfs_name in glb_variants.items():
        glb_path = output_dir / local_name
        if glb_path.exists():
            if _push_glb_to_gridfs(glb_path, gridfs_name):
                pushed.append(gridfs_name)

    if pushed:
        logger.info("GridFS push complete for %s: %s", model_name, pushed)
        # Save metadata to MongoDB so model list persists across deploys
        try:
            from pymongo import MongoClient
            uri = os.environ.get("MONGODB_URI", "")
            if uri:
                client = MongoClient(uri)
                db = client[os.environ.get("MONGODB_DB", "marip_f1")]
                meta = {}
                meta_path = output_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                doc = {
                    "model_name": model_name,
                    "has_hunyuan": any("hunyuan" in f and "pbr" not in f and "textured" not in f for f in pushed),
                    "has_pbr": any("pbr" in f for f in pushed),
                    "has_hunyuan_textured": any("textured" in f for f in pushed),
                    "has_meshy": any("meshy" in f for f in pushed),
                    "has_tripo": any("tripo" in f for f in pushed),
                    "has_trellis": any("trellis" in f for f in pushed),
                    "has_texture_paint": any("texture_paint" in f for f in pushed),
                    "gridfs_files": pushed,
                    "provider": meta.get("provider", provider),
                    "created_at": meta.get("created_at", datetime.now(timezone.utc).isoformat()),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                db["generated_models"].update_one(
                    {"model_name": model_name}, {"$set": doc}, upsert=True,
                )
                logger.info("Saved model metadata to MongoDB: %s", model_name)
                client.close()
        except Exception as e:
            logger.warning("MongoDB metadata save failed: %s", e)

    return pushed


def _save_metadata(output_dir: Path, metadata: dict):
    """Merge metadata into output_dir/metadata.json."""
    meta_path = output_dir / "metadata.json"
    existing = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    existing.update(metadata)
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)
