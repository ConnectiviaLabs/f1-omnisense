"""OmniBedding APIRouter — embedding visualization with dimensionality reduction and clustering.

Endpoints:
    POST /api/omni/bedding/visualize             — reduce + cluster + return JSON for frontend
    POST /api/omni/bedding/visualize/knowledge    — visualize RAG knowledge base embeddings
    POST /api/omni/bedding/visualize/driver/{code} — visualize driver telemetry embeddings
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/omni/bedding", tags=["OmniBedding"])


# ── Request/Response models ─────────────────────────────────────────────

class VisualizeRequest(BaseModel):
    """Request body for custom embedding visualization."""
    embeddings: List[List[float]]
    labels: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    n_components: int = 3
    method: str = "auto"  # "auto" | "umap" | "tsne" | "pca"
    n_clusters: int = 4
    cluster_method: str = "kmeans"  # "kmeans" | "dbscan"
    anomaly_percentile: float = 95


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/visualize")
def visualize(req: VisualizeRequest):
    """Reduce, cluster, and return embedding visualization as JSON for frontend (Three.js/D3)."""
    from omnibedding import visualize as _visualize

    embeddings = np.array(req.embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise HTTPException(400, "embeddings must be a 2D array of shape (N, D)")

    result = _visualize(
        embeddings=embeddings,
        labels=req.labels,
        categories=req.categories,
        n_components=req.n_components,
        method=req.method,
        n_clusters=req.n_clusters,
        cluster_method=req.cluster_method,
        output="json",
        anomaly_percentile=req.anomaly_percentile,
    )
    return result


@router.get("/visualize/knowledge")
def visualize_knowledge(
    n_components: int = Query(3, ge=2, le=3),
    method: str = Query("auto"),
    n_clusters: int = Query(6, ge=2, le=20),
    limit: int = Query(500, ge=10, le=5000),
):
    """Visualize RAG knowledge base embeddings from MongoDB Atlas.

    Fetches stored embeddings from f1_knowledge collection, reduces to 2D/3D,
    clusters, and returns JSON for the frontend.
    """
    from omnibedding import visualize as _visualize
    from pymongo import MongoClient

    uri = os.getenv("MONGODB_URI", "")
    db_name = os.getenv("MONGODB_DB", "marip_f1")

    client = MongoClient(uri)
    db = client[db_name]

    docs = list(db["f1_knowledge"].find(
        {"embedding": {"$exists": True}},
        {"embedding": 1, "page_content": 1, "metadata": 1, "_id": 0},
    ).limit(limit))

    if not docs:
        raise HTTPException(404, "No embeddings found in f1_knowledge collection")

    embeddings = []
    labels = []
    categories = []
    metadata_list = []

    for doc in docs:
        emb = doc.get("embedding")
        if not emb or not isinstance(emb, list):
            continue
        embeddings.append(emb)
        meta = doc.get("metadata", {})
        content_preview = (doc.get("page_content", "") or "")[:80]
        labels.append(content_preview)
        categories.append(meta.get("data_type", "unknown"))
        metadata_list.append({
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
            "category": meta.get("category", ""),
            "data_type": meta.get("data_type", ""),
        })

    if len(embeddings) < 3:
        raise HTTPException(400, f"Need at least 3 embeddings, found {len(embeddings)}")

    result = _visualize(
        embeddings=np.array(embeddings, dtype=np.float32),
        labels=labels,
        categories=categories,
        n_components=n_components,
        method=method,
        n_clusters=n_clusters,
        output="json",
        metadata=metadata_list,
    )
    return result


@router.get("/visualize/driver/{driver_code}")
def visualize_driver(
    driver_code: str,
    n_components: int = Query(3, ge=2, le=3),
    method: str = Query("auto"),
    n_clusters: int = Query(4, ge=2, le=10),
):
    """Visualize driver telemetry as an embedding space.

    Treats each race's aggregated telemetry row as a high-dimensional point,
    reduces to 2D/3D, clusters, and returns JSON.
    """
    from omnibedding import visualize as _visualize
    from pipeline.anomaly.run_f1_anomaly import (
        load_car_race_data,
        load_bio_race_data,
        merge_telemetry,
    )

    drivers = {"NOR": "Lando Norris", "PIA": "Oscar Piastri"}
    if driver_code not in drivers:
        raise HTTPException(404, f"Unknown driver: {driver_code}")

    df = merge_telemetry(load_car_race_data(driver_code), load_bio_race_data(driver_code))

    # Use numeric columns as embedding dimensions
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "race" in df.columns:
        labels = df["race"].astype(str).tolist()
    else:
        labels = [f"row_{i}" for i in range(len(df))]

    embeddings = df[numeric_cols].fillna(0).values.astype(np.float32)

    if len(embeddings) < 3:
        raise HTTPException(400, f"Not enough data points for {driver_code}")

    # Force PCA for small datasets (t-SNE perplexity requires n_samples > 30)
    effective_method = method if len(embeddings) > 30 else "pca"

    result = _visualize(
        embeddings=embeddings,
        labels=labels,
        categories=[driver_code] * len(labels),
        n_components=n_components,
        method=effective_method,
        n_clusters=min(n_clusters, len(embeddings) // 2),
        output="json",
    )
    result["driver"] = drivers[driver_code]
    result["driver_code"] = driver_code
    result["feature_columns"] = numeric_cols
    return result
