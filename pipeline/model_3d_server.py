"""3D model generation routes for F1 OmniSense.

Provides endpoints for submitting generation jobs,
polling status, downloading GLBs, and managing PBR materials.

Can run standalone on port 8101 or be included as a router in the main server.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from texture.generate_3d import (
    submit_job as submit_3d_job,
    get_job as get_3d_job,
    list_jobs as list_3d_jobs,
    list_generated_models,
    get_mesh_quality,
    MODELS_3D_DIR,
)
from texture.apply_pbr import MATERIAL_PBR

logger = logging.getLogger(__name__)

# ── Router (importable by chat_server) ────────────────────────────────

router = APIRouter(prefix="/api/3d-gen", tags=["3D Generation"])


class Regenerate3DRequest(BaseModel):
    texture_prompt: str = ""
    enable_pbr: bool = True
    target_polycount: int = 30000


@router.post("/generate")
async def generate_3d_model(
    image: UploadFile = File(...),
    model_name: str = Form(""),
    provider: str = Form("meshy"),
    textured: bool = Form(False),
    steps: int = Form(30),
    guidance_scale: float = Form(5.0),
    seed: int = Form(1234),
    octree_resolution: int = Form(256),
    randomize_seed: bool = Form(False),
    texture_prompt: str = Form(""),
    enable_pbr: bool = Form(True),
    target_polycount: int = Form(30000),
    material_preset: str = Form(""),
    multi_view: bool = Form(False),
    n_views: int = Form(8),
    texture_resolution: int = Form(1024),
):
    """Submit a 3D model generation job with image upload."""
    if provider not in ("hunyuan", "meshy"):
        raise HTTPException(status_code=400, detail="Provider must be 'hunyuan' or 'meshy'")

    if not model_name:
        stem = Path(image.filename).stem if image.filename else "model"
        model_name = stem.replace(" ", "_").replace(".", "_")

    output_dir = MODELS_3D_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "input_image.png"

    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)

    params = {}
    if provider == "hunyuan":
        params = {
            "textured": textured, "steps": steps,
            "guidance_scale": guidance_scale, "seed": seed,
            "octree_resolution": octree_resolution,
            "randomize_seed": randomize_seed,
            "material_preset": material_preset or None,
        }
    elif provider == "meshy":
        params = {
            "texture_prompt": texture_prompt,
            "enable_pbr": enable_pbr,
            "target_polycount": target_polycount,
        }
    # elif provider == "texture_paint":
    #     params = {
    #         "material_preset": material_preset or None,
    #         "prompt": texture_prompt or None,
    #         "n_views": n_views,
    #         "texture_resolution": texture_resolution,
    #         "seed": seed,
    #     }

    job_id = submit_3d_job(
        model_name=model_name,
        provider=provider,
        image_path=str(image_path),
        **params,
    )

    return {"job_id": job_id, "status": "queued", "provider": provider, "model_name": model_name}


@router.post("/regenerate/{model_name}")
def regenerate_3d_model(model_name: str, request: Regenerate3DRequest):
    """Regenerate a 3D model with Meshy (reuses existing input image)."""
    model_dir = MODELS_3D_DIR / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"No model found for '{model_name}'")

    image_path = model_dir / "input_image.png"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"No input image found for '{model_name}'")

    job_id = submit_3d_job(
        model_name=model_name,
        provider="meshy",
        image_path=str(image_path),
        texture_prompt=request.texture_prompt,
        enable_pbr=request.enable_pbr,
        target_polycount=request.target_polycount,
    )

    return {"job_id": job_id, "status": "queued", "provider": "meshy"}


@router.get("/jobs/{job_id}")
def get_3d_job_status(job_id: str):
    """Get status of a 3D generation job."""
    job = get_3d_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.get("/jobs")
def list_3d_generation_jobs():
    """List all 3D generation jobs."""
    return {"jobs": list_3d_jobs()}


@router.get("/models")
def list_3d_models():
    """List all generated 3D models."""
    return {"models": list_generated_models()}


@router.get("/models/{model_name}/glb")
def download_3d_glb(model_name: str, provider: str = "hunyuan", textured: bool = False):
    """Download GLB file for a model (local filesystem first, GridFS fallback)."""
    model_dir = MODELS_3D_DIR / model_name

    if provider == "texture_paint":
        glb_name = "texture_paint.glb"
    elif textured and provider == "hunyuan":
        glb_name = "hunyuan_textured.glb"
    elif provider == "hunyuan" and not textured:
        pbr_path = model_dir / "hunyuan_pbr.glb"
        glb_name = "hunyuan_pbr.glb" if pbr_path.exists() else "hunyuan.glb"
    else:
        glb_name = f"{provider}.glb"

    glb_path = model_dir / glb_name

    # Serve from local filesystem if available
    if glb_path.exists():
        return FileResponse(
            glb_path,
            media_type="model/gltf-binary",
            filename=f"{model_name}_{glb_name}",
        )

    # Fallback: serve from MongoDB GridFS
    gridfs_filename = f"{model_name}_{glb_name}"
    try:
        import gridfs
        from starlette.responses import StreamingResponse
        from pymongo import MongoClient

        uri = os.environ.get("MONGODB_URI", "")
        if uri:
            client = MongoClient(uri)
            db = client[os.environ.get("MONGODB_DB", "marip_f1")]
            fs = gridfs.GridFS(db)
            grid_file = fs.find_one({"filename": gridfs_filename})
            if grid_file:
                def stream():
                    while True:
                        chunk = grid_file.read(256 * 1024)
                        if not chunk:
                            break
                        yield chunk

                return StreamingResponse(
                    stream(),
                    media_type="model/gltf-binary",
                    headers={
                        "Content-Length": str(grid_file.length),
                        "Content-Disposition": f'inline; filename="{gridfs_filename}"',
                        "Cache-Control": "public, max-age=604800",
                    },
                )
    except Exception as e:
        logger.warning("GridFS fallback failed for %s: %s", gridfs_filename, e)

    raise HTTPException(
        status_code=404,
        detail=f"GLB not found: {model_name}/{glb_name}"
    )


@router.post("/apply-texture/{model_name}")
def apply_texture_to_model(model_name: str, preset: str = None):
    """Apply PBR material to an existing Hunyuan GLB."""
    from texture.apply_pbr import apply_texture_to_model as apply_pbr

    model_dir = MODELS_3D_DIR / model_name
    if not (model_dir / "hunyuan.glb").exists():
        raise HTTPException(
            status_code=404,
            detail=f"No Hunyuan GLB found for '{model_name}'. Generate shape first.",
        )

    result = apply_pbr(model_name, preset=preset or "carbon_fiber")
    return result


@router.get("/quality/{model_name}")
def get_3d_quality(model_name: str):
    """Get mesh quality info for a model."""
    return get_mesh_quality(model_name)


@router.get("/texture-paint/status")
def get_texture_paint_status():
    """Check if TEXTure painting is available (requires GPU + kaolin)."""
    try:
        from texture.texture_paint import is_available
        return is_available()
    except ImportError:
        return {
            "available": False,
            "gpu": "",
            "vram_gb": 0.0,
            "reason": "texture_paint module not available",
        }


@router.get("/material-presets")
def list_material_presets():
    """List available F1 PBR material presets."""
    presets = []
    for key, props in MATERIAL_PBR.items():
        presets.append({
            "value": key,
            "label": props.get("display_name", key.replace("_", " ").title()),
            "baseColor": props["baseColor"],
            "metallic": props["metallic"],
            "roughness": props["roughness"],
        })
    return {"presets": presets}


# ── Static file mount helper ──────────────────────────────────────────

def mount_3d_static(app: FastAPI):
    """Mount the 3D models directory for static file access."""
    MODELS_3D_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/3d-models", StaticFiles(directory=str(MODELS_3D_DIR)), name="3d-models")


# ── Standalone mode ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    app = FastAPI(title="F1 3D Model Generator", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    mount_3d_static(app)

    @app.get("/health")
    def health_check():
        return {"status": "ok", "service": "f1-3d-gen"}

    port = int(os.getenv("MODEL_3D_PORT", "8101"))
    logger.info("Starting F1 3D Model Generator on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
