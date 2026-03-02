"""Dimensionality reduction — UMAP → t-SNE → PCA cascade.

Ports best of cadAI (UMAP cosine params) and DataSense (PCA + normalization).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from omnibedding._types import ReductionMethod

logger = logging.getLogger(__name__)


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def _center_and_scale(coords: np.ndarray, scale: float) -> np.ndarray:
    """Center at origin, scale to ±scale."""
    coords = coords - coords.mean(axis=0)
    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords = coords * (scale / max_abs)
    return coords


def _reduce_umap(embeddings: np.ndarray, n_components: int, **kwargs) -> np.ndarray:
    """UMAP reduction with cadAI defaults."""
    import umap  # optional dep

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=kwargs.get("n_neighbors", 15),
        min_dist=kwargs.get("min_dist", 0.1),
        metric=kwargs.get("metric", "cosine"),
        random_state=kwargs.get("random_state", 42),
    )
    return reducer.fit_transform(embeddings)


def _reduce_tsne(embeddings: np.ndarray, n_components: int, **kwargs) -> np.ndarray:
    """t-SNE reduction via sklearn."""
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=n_components,
        perplexity=kwargs.get("perplexity", 30),
        learning_rate=kwargs.get("learning_rate", "auto"),
        init=kwargs.get("init", "pca"),
        random_state=kwargs.get("random_state", 42),
    )
    return tsne.fit_transform(embeddings)


def _reduce_pca(
    embeddings: np.ndarray, n_components: int, **kwargs
) -> Tuple[np.ndarray, float]:
    """PCA reduction with explained variance. Returns (coords, explained_variance)."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=kwargs.get("random_state", 10))
    coords = pca.fit_transform(embeddings)
    explained = float(pca.explained_variance_ratio_.sum())
    return coords, explained


def reduce(
    embeddings: np.ndarray,
    n_components: int = 3,
    method: str = "auto",
    normalize: bool = True,
    scale: float = 1.0,
    **kwargs,
) -> Tuple[np.ndarray, ReductionMethod, float]:
    """Reduce high-dimensional embeddings to 2D or 3D.

    Parameters
    ----------
    embeddings : (N, D) array of embedding vectors.
    n_components : 2 or 3.
    method : "auto" (cascade UMAP→t-SNE→PCA), "umap", "tsne", or "pca".
    normalize : L2-normalize input embeddings.
    scale : Scale output coordinates to ±scale (1.0 default, 10000 for Three.js).
    **kwargs : Passed to the underlying reducer (n_neighbors, perplexity, etc.).

    Returns
    -------
    (coords, method_used, explained_variance)
        coords: (N, n_components) array.
        method_used: which ReductionMethod was actually applied.
        explained_variance: only meaningful for PCA, 0.0 otherwise.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
    if n_components not in (2, 3):
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")

    N = embeddings.shape[0]
    if N < 2:
        raise ValueError(f"Need at least 2 points, got {N}")

    if normalize:
        embeddings = _l2_normalize(embeddings)

    explained_variance = 0.0
    method = method.lower()

    if method == "auto":
        # Cascade: UMAP → t-SNE (if small) → PCA
        try:
            coords = _reduce_umap(embeddings, n_components, **kwargs)
            method_used = ReductionMethod.UMAP
            logger.info("Using UMAP (%d points → %dD)", N, n_components)
        except ImportError:
            if N <= 5000:
                coords = _reduce_tsne(embeddings, n_components, **kwargs)
                method_used = ReductionMethod.TSNE
                logger.info("Using t-SNE (%d points → %dD)", N, n_components)
            else:
                coords, explained_variance = _reduce_pca(
                    embeddings, n_components, **kwargs
                )
                method_used = ReductionMethod.PCA
                logger.info(
                    "Using PCA (%d points → %dD, %.1f%% variance)",
                    N, n_components, explained_variance * 100,
                )
    elif method == "umap":
        coords = _reduce_umap(embeddings, n_components, **kwargs)
        method_used = ReductionMethod.UMAP
    elif method == "tsne":
        coords = _reduce_tsne(embeddings, n_components, **kwargs)
        method_used = ReductionMethod.TSNE
    elif method == "pca":
        coords, explained_variance = _reduce_pca(embeddings, n_components, **kwargs)
        method_used = ReductionMethod.PCA
    else:
        raise ValueError(f"Unknown method '{method}'. Use auto/umap/tsne/pca.")

    coords = _center_and_scale(coords, scale)
    return coords, method_used, explained_variance
