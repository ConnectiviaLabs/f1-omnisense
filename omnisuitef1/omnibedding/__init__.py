"""omnibedding — Embedding Visualization Module.

Thin visualization layer: numpy arrays in → 2D/3D scatter plots, cluster analysis,
anomaly detection, JSON export for frontends.

Part of omnisuite. Consumes embeddings from omnivis, omnirag, or any source.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from omnibedding._types import (
    CLUSTER_COLORS,
    ClusterInfo,
    ClusterMethod,
    EmbeddingPoint,
    EmbeddingViz,
    ReductionMethod,
)
from omnibedding.reducer import reduce
from omnibedding.cluster import cluster_and_detect
from omnibedding.plotter import (
    plot_2d,
    plot_3d,
    plot_interactive_2d,
    plot_interactive_3d,
    plot_similarity_network,
    to_json,
)


def visualize(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    n_components: int = 3,
    method: str = "auto",
    n_clusters: int = 4,
    cluster_method: str = "kmeans",
    output: str = "viz",
    scale: float = 1.0,
    anomaly_percentile: float = 95,
    metadata: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> Union[EmbeddingViz, str, Dict]:
    """One-call pipeline: reduce → cluster → detect anomalies → output.

    Parameters
    ----------
    embeddings : (N, D) array of embedding vectors.
    labels : Optional label per point.
    categories : Optional category per point.
    n_components : 2 or 3.
    method : "auto" | "umap" | "tsne" | "pca".
    n_clusters : Number of clusters for KMeans.
    cluster_method : "kmeans" | "dbscan".
    output : "viz" (EmbeddingViz), "json" (dict), "png" (file path), "html" (HTML string).
    scale : Coordinate scale (1.0 default, 10000 for Three.js JSON).
    anomaly_percentile : 95 = top 5% are anomalies.
    metadata : Optional dict per point (included in EmbeddingPoint.metadata).
    **kwargs : Passed to reducer / plotter.

    Returns
    -------
    EmbeddingViz if output="viz", Dict if output="json",
    str (file path) if output="png", str (HTML) if output="html".
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    N = len(embeddings)

    if labels is None:
        labels = [f"point_{i}" for i in range(N)]
    if categories is None:
        categories = ["default"] * N
    if metadata is None:
        metadata = [{}] * N

    # Step 1: Reduce
    json_scale = 10000.0 if output == "json" else scale
    coords, method_used, explained_var = reduce(
        embeddings,
        n_components=n_components,
        method=method,
        scale=json_scale if output == "json" else scale,
        **{k: v for k, v in kwargs.items() if k in (
            "normalize", "n_neighbors", "min_dist", "metric",
            "perplexity", "learning_rate", "random_state",
        )},
    )

    # Step 2: Cluster + anomaly detect
    cluster_labels, is_anomaly, anomaly_scores, distances, cluster_infos = (
        cluster_and_detect(
            coords,
            n_clusters=n_clusters,
            cluster_method=cluster_method,
            anomaly_percentile=anomaly_percentile,
            categories=categories,
            **{k: v for k, v in kwargs.items() if k in (
                "knn_neighbors", "dbscan_eps", "dbscan_min_samples",
            )},
        )
    )

    # Step 3: Build EmbeddingViz
    points = []
    for i in range(N):
        z_val = float(coords[i, 2]) if n_components == 3 else 0.0
        points.append(EmbeddingPoint(
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            z=z_val,
            cluster=int(cluster_labels[i]),
            is_anomaly=bool(is_anomaly[i]),
            anomaly_score=float(anomaly_scores[i]),
            distance=float(distances[i]),
            label=labels[i] if i < len(labels) else "",
            category=categories[i] if i < len(categories) else "",
            metadata=metadata[i] if i < len(metadata) else {},
        ))

    cm = ClusterMethod.DBSCAN if cluster_method.lower() == "dbscan" else ClusterMethod.KMEANS

    viz = EmbeddingViz(
        points=points,
        clusters=cluster_infos,
        reduction_method=method_used,
        cluster_method=cm,
        n_dimensions=n_components,
        explained_variance=explained_var,
        total_points=N,
        total_anomalies=int(is_anomaly.sum()),
    )

    # Step 4: Output
    output = output.lower()
    if output == "viz":
        return viz
    elif output == "json":
        return to_json(viz, scale=10000)
    elif output == "png":
        if n_components == 2:
            return plot_2d(viz, **{k: v for k, v in kwargs.items() if k in (
                "title", "figsize", "save_path", "dpi",
            )})
        else:
            return plot_3d(viz, **{k: v for k, v in kwargs.items() if k in (
                "title", "figsize", "save_path", "elev", "azim", "dpi",
            )})
    elif output == "html":
        if n_components == 2:
            return plot_interactive_2d(viz, **{k: v for k, v in kwargs.items() if k in (
                "title", "width", "height", "save_path",
            )})
        else:
            return plot_interactive_3d(viz, **{k: v for k, v in kwargs.items() if k in (
                "title", "width", "height", "save_path",
            )})
    else:
        raise ValueError(f"Unknown output '{output}'. Use viz/json/png/html.")


__all__ = [
    # Types
    "EmbeddingPoint",
    "ClusterInfo",
    "EmbeddingViz",
    "ReductionMethod",
    "ClusterMethod",
    "CLUSTER_COLORS",
    # Core functions
    "reduce",
    "cluster_and_detect",
    # Plotting
    "plot_2d",
    "plot_3d",
    "plot_interactive_2d",
    "plot_interactive_3d",
    "plot_similarity_network",
    "to_json",
    # Convenience
    "visualize",
]
