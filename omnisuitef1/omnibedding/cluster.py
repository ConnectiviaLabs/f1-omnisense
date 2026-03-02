"""Clustering + anomaly detection — KMeans/DBSCAN + dual anomaly scoring.

Ports DataSense embeddingsAnalysisAndClustering() and k-NN anomaly detection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from omnibedding._types import ClusterInfo, ClusterMethod

logger = logging.getLogger(__name__)


def _kmeans_cluster(
    coords: np.ndarray, n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """KMeans clustering. Returns (labels, centroids)."""
    k = min(n_clusters, len(coords))
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(coords)
    return labels, model.cluster_centers_


def _dbscan_cluster(
    coords: np.ndarray, eps: float, min_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """DBSCAN clustering. Returns (labels, centroids).
    Noise points get label -1. Centroids computed as cluster means.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(coords)
    unique = set(labels)
    unique.discard(-1)
    centroids = []
    for cid in sorted(unique):
        mask = labels == cid
        centroids.append(coords[mask].mean(axis=0))
    if not centroids:
        centroids = [coords.mean(axis=0)]
    return labels, np.array(centroids)


def _distance_anomalies(
    coords: np.ndarray, centroids: np.ndarray, labels: np.ndarray, percentile: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance-based anomaly detection (DataSense method).
    Points in the top `percentile`% of distance from their cluster centroid are anomalies.
    Returns (is_anomaly, distances).
    """
    distances = np.zeros(len(coords))
    for i, (pt, cid) in enumerate(zip(coords, labels)):
        if 0 <= cid < len(centroids):
            distances[i] = np.linalg.norm(pt - centroids[cid])
        else:
            # noise point (DBSCAN -1): distance from global centroid
            distances[i] = np.linalg.norm(pt - coords.mean(axis=0))

    threshold = np.percentile(distances, percentile)
    is_anomaly = distances > threshold
    return is_anomaly, distances


def _knn_anomalies(
    coords: np.ndarray, k: int, percentile: float
) -> Tuple[np.ndarray, np.ndarray]:
    """k-NN anomaly detection (DataSense method).
    Average distance to k nearest neighbors; top percentile% are anomalies.
    Returns (is_anomaly, anomaly_scores).
    """
    k_actual = min(k, len(coords) - 1)
    if k_actual < 1:
        return np.zeros(len(coords), dtype=bool), np.zeros(len(coords))

    nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    # Skip first column (distance to self = 0)
    scores = distances[:, 1:].mean(axis=1)

    threshold = np.percentile(scores, percentile)
    is_anomaly = scores > threshold
    return is_anomaly, scores


def _build_cluster_infos(
    coords: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    is_anomaly: np.ndarray,
    distances: np.ndarray,
    categories: Optional[List[str]],
) -> List[ClusterInfo]:
    """Build ClusterInfo objects for each cluster."""
    unique_labels = sorted(set(labels))
    infos = []

    for cid in unique_labels:
        mask = labels == cid
        cluster_distances = distances[mask]
        cluster_anomalies = is_anomaly[mask]

        avg_dist = float(cluster_distances.mean()) if len(cluster_distances) > 0 else 0.0
        std_dist = float(cluster_distances.std()) if len(cluster_distances) > 0 else 0.0
        density = 1.0 / (avg_dist + 1e-8)

        cat_counts: Dict[str, int] = {}
        if categories:
            for i, m in enumerate(mask):
                if m and i < len(categories):
                    cat = categories[i]
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1

        centroid = centroids[cid].tolist() if 0 <= cid < len(centroids) else []

        infos.append(
            ClusterInfo(
                cluster_id=int(cid),
                centroid=centroid,
                size=int(mask.sum()),
                anomaly_count=int(cluster_anomalies.sum()),
                avg_distance=avg_dist,
                std_distance=std_dist,
                density=density,
                categories=cat_counts,
            )
        )

    return infos


def cluster_and_detect(
    coords: np.ndarray,
    n_clusters: int = 4,
    cluster_method: str = "kmeans",
    anomaly_percentile: float = 95,
    knn_neighbors: int = 4,
    categories: Optional[List[str]] = None,
    dbscan_eps: float = 0.3,
    dbscan_min_samples: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[ClusterInfo]]:
    """Cluster reduced coordinates and detect anomalies.

    Parameters
    ----------
    coords : (N, 2|3) reduced coordinates.
    n_clusters : Number of clusters for KMeans.
    cluster_method : "kmeans" or "dbscan".
    anomaly_percentile : Percentile threshold (95 = top 5% are anomalies).
    knn_neighbors : k for k-NN anomaly scoring.
    categories : Optional category label per point (for cluster breakdowns).
    dbscan_eps, dbscan_min_samples : DBSCAN parameters.

    Returns
    -------
    (cluster_labels, is_anomaly, anomaly_scores, distances, cluster_infos)
    """
    coords = np.asarray(coords, dtype=np.float64)

    # Cluster
    method_str = cluster_method.lower()
    if method_str == "kmeans":
        labels, centroids = _kmeans_cluster(coords, n_clusters)
    elif method_str == "dbscan":
        labels, centroids = _dbscan_cluster(coords, dbscan_eps, dbscan_min_samples)
    else:
        raise ValueError(f"Unknown cluster_method '{cluster_method}'. Use kmeans/dbscan.")

    # Dual anomaly detection
    dist_anomaly, distances = _distance_anomalies(
        coords, centroids, labels, anomaly_percentile
    )
    knn_anomaly, anomaly_scores = _knn_anomalies(
        coords, knn_neighbors, anomaly_percentile
    )

    # Union: anomaly if either method flags it
    is_anomaly = dist_anomaly | knn_anomaly

    logger.info(
        "Clustered %d points into %d clusters, %d anomalies (%.1f%%)",
        len(coords),
        len(set(labels)),
        is_anomaly.sum(),
        is_anomaly.mean() * 100,
    )

    # Build cluster infos
    cluster_infos = _build_cluster_infos(
        coords, labels, centroids, is_anomaly, distances, categories
    )

    return labels, is_anomaly, anomaly_scores, distances, cluster_infos
