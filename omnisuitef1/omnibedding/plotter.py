"""Visualization output — 2D/3D scatter, network graph, JSON export.

Combines DataSense visuals (cluster colors, anomaly markers, centroids)
with cadAI JSON export format (for Three.js/D3.js frontends).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from omnibedding._types import CLUSTER_COLORS, EmbeddingViz

logger = logging.getLogger(__name__)


def _color_for_cluster(cid: int) -> str:
    """Get color for a cluster ID, cycling through palette."""
    if cid < 0:
        return "#666666"  # noise / unassigned
    return CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]


def _extract_arrays(viz: EmbeddingViz) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Extract parallel arrays from viz points."""
    xs = np.array([p.x for p in viz.points])
    ys = np.array([p.y for p in viz.points])
    zs = np.array([p.z for p in viz.points])
    clusters = np.array([p.cluster for p in viz.points])
    labels = [p.label for p in viz.points]
    categories = [p.category for p in viz.points]
    return xs, ys, zs, clusters, labels, categories


# ── Matplotlib static plots ─────────────────────────────────────────────


def plot_2d(
    viz: EmbeddingViz,
    title: str = "Embedding Visualization (2D)",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show_centroids: bool = True,
    show_legend: bool = True,
    dpi: int = 150,
) -> str:
    """Matplotlib 2D scatter plot. Returns path to saved PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, _, clusters, _, _ = _extract_arrays(viz)
    unique_clusters = sorted(set(clusters))

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    # Plot each cluster
    for cid in unique_clusters:
        mask = clusters == cid
        color = _color_for_cluster(cid)
        label_str = f"Cluster {cid}" if cid >= 0 else "Noise"
        ax.scatter(
            xs[mask], ys[mask],
            c=color, s=20, alpha=0.7, label=f"{label_str} ({mask.sum()})",
            edgecolors="none",
        )

    # Anomaly highlights
    anomaly_mask = np.array([p.is_anomaly for p in viz.points])
    if anomaly_mask.any():
        ax.scatter(
            xs[anomaly_mask], ys[anomaly_mask],
            facecolors="none", edgecolors="#FF0000", s=60, linewidths=1.5,
            alpha=0.8, label=f"Anomalies ({anomaly_mask.sum()})",
        )

    # Centroids
    if show_centroids:
        for ci in viz.clusters:
            if len(ci.centroid) >= 2:
                ax.scatter(
                    ci.centroid[0], ci.centroid[1],
                    c=_color_for_cluster(ci.cluster_id),
                    s=200, marker="X", edgecolors="white", linewidths=2, zorder=5,
                )

    ax.set_title(title, color="white", fontsize=14)
    ax.set_xlabel("Dimension 1", color="white")
    ax.set_ylabel("Dimension 2", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=0.15, color="white")

    if show_legend:
        legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        legend.get_frame().set_facecolor("#1a1a2e")
        for text in legend.get_texts():
            text.set_color("white")

    plt.tight_layout()

    if not save_path:
        save_path = str(Path(tempfile.gettempdir()) / "omnibedding_2d.png")
    fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved 2D plot: %s", save_path)
    return save_path


def plot_3d(
    viz: EmbeddingViz,
    title: str = "Embedding Visualization (3D)",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    elev: float = 25,
    azim: float = 45,
    show_centroids: bool = True,
    show_legend: bool = True,
    dpi: int = 150,
) -> str:
    """Matplotlib 3D scatter plot. Returns path to saved PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, zs, clusters, _, _ = _extract_arrays(viz)
    unique_clusters = sorted(set(clusters))

    fig = plt.figure(figsize=figsize, facecolor="#0a0a1a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a1a")

    for cid in unique_clusters:
        mask = clusters == cid
        color = _color_for_cluster(cid)
        label_str = f"Cluster {cid}" if cid >= 0 else "Noise"
        ax.scatter(
            xs[mask], ys[mask], zs[mask],
            c=color, s=15, alpha=0.7, label=f"{label_str} ({mask.sum()})",
            depthshade=True,
        )

    # Anomaly highlights
    anomaly_mask = np.array([p.is_anomaly for p in viz.points])
    if anomaly_mask.any():
        ax.scatter(
            xs[anomaly_mask], ys[anomaly_mask], zs[anomaly_mask],
            facecolors="none", edgecolors="#FF0000", s=50, linewidths=1.2,
            alpha=0.8, label=f"Anomalies ({anomaly_mask.sum()})",
        )

    if show_centroids:
        for ci in viz.clusters:
            if len(ci.centroid) >= 3:
                ax.scatter(
                    *ci.centroid[:3],
                    c=_color_for_cluster(ci.cluster_id),
                    s=200, marker="X", edgecolors="white", linewidths=2, zorder=5,
                )

    ax.set_title(title, color="white", fontsize=14)
    ax.set_xlabel("Dim 1", color="white")
    ax.set_ylabel("Dim 2", color="white")
    ax.set_zlabel("Dim 3", color="white")
    ax.tick_params(colors="white")
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    if show_legend:
        legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        legend.get_frame().set_facecolor("#1a1a2e")
        for text in legend.get_texts():
            text.set_color("white")

    plt.tight_layout()

    if not save_path:
        save_path = str(Path(tempfile.gettempdir()) / "omnibedding_3d.png")
    fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved 3D plot: %s", save_path)
    return save_path


# ── Plotly interactive (optional) ────────────────────────────────────────


def plot_interactive_2d(
    viz: EmbeddingViz,
    title: str = "Embedding Visualization (2D Interactive)",
    width: int = 1000,
    height: int = 700,
    save_path: Optional[str] = None,
) -> str:
    """Plotly 2D scatter → HTML string. Requires plotly."""
    import plotly.graph_objects as go

    fig = go.Figure()

    unique_clusters = sorted(set(p.cluster for p in viz.points))
    for cid in unique_clusters:
        pts = [p for p in viz.points if p.cluster == cid]
        color = _color_for_cluster(cid)
        name = f"Cluster {cid}" if cid >= 0 else "Noise"

        fig.add_trace(go.Scatter(
            x=[p.x for p in pts],
            y=[p.y for p in pts],
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.7),
            name=f"{name} ({len(pts)})",
            text=[
                f"Label: {p.label}<br>Cat: {p.category}<br>"
                f"Cluster: {p.cluster}<br>Anomaly: {p.is_anomaly}<br>"
                f"Score: {p.anomaly_score:.3f}"
                for p in pts
            ],
            hoverinfo="text",
        ))

    # Anomaly ring overlay
    anomalies = [p for p in viz.points if p.is_anomaly]
    if anomalies:
        fig.add_trace(go.Scatter(
            x=[p.x for p in anomalies],
            y=[p.y for p in anomalies],
            mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(color="red", width=2)),
            name=f"Anomalies ({len(anomalies)})",
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=title, width=width, height=height,
        template="plotly_dark",
        xaxis_title="Dimension 1", yaxis_title="Dimension 2",
    )

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    if save_path:
        Path(save_path).write_text(html)
        logger.info("Saved interactive 2D: %s", save_path)
    return html


def plot_interactive_3d(
    viz: EmbeddingViz,
    title: str = "Embedding Visualization (3D Interactive)",
    width: int = 1000,
    height: int = 800,
    save_path: Optional[str] = None,
) -> str:
    """Plotly 3D scatter → HTML string. Requires plotly."""
    import plotly.graph_objects as go

    fig = go.Figure()

    unique_clusters = sorted(set(p.cluster for p in viz.points))
    for cid in unique_clusters:
        pts = [p for p in viz.points if p.cluster == cid]
        color = _color_for_cluster(cid)
        name = f"Cluster {cid}" if cid >= 0 else "Noise"

        fig.add_trace(go.Scatter3d(
            x=[p.x for p in pts],
            y=[p.y for p in pts],
            z=[p.z for p in pts],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.7),
            name=f"{name} ({len(pts)})",
            text=[
                f"Label: {p.label}<br>Cat: {p.category}<br>"
                f"Cluster: {p.cluster}<br>Anomaly: {p.is_anomaly}<br>"
                f"Score: {p.anomaly_score:.3f}"
                for p in pts
            ],
            hoverinfo="text",
        ))

    # Anomaly overlay
    anomalies = [p for p in viz.points if p.is_anomaly]
    if anomalies:
        fig.add_trace(go.Scatter3d(
            x=[p.x for p in anomalies],
            y=[p.y for p in anomalies],
            z=[p.z for p in anomalies],
            mode="markers",
            marker=dict(size=6, color="rgba(0,0,0,0)", line=dict(color="red", width=2)),
            name=f"Anomalies ({len(anomalies)})",
            hoverinfo="skip",
        ))

    # Centroids
    for ci in viz.clusters:
        if len(ci.centroid) >= 3:
            fig.add_trace(go.Scatter3d(
                x=[ci.centroid[0]], y=[ci.centroid[1]], z=[ci.centroid[2]],
                mode="markers",
                marker=dict(
                    size=10, color=_color_for_cluster(ci.cluster_id),
                    symbol="x", line=dict(color="white", width=2),
                ),
                name=f"Centroid {ci.cluster_id}",
                showlegend=False,
            ))

    fig.update_layout(
        title=title, width=width, height=height,
        template="plotly_dark",
        scene=dict(
            xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3",
        ),
    )

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    if save_path:
        Path(save_path).write_text(html)
        logger.info("Saved interactive 3D: %s", save_path)
    return html


# ── JSON export for frontends ────────────────────────────────────────────


def to_json(viz: EmbeddingViz, scale: float = 10000) -> Dict[str, Any]:
    """Export visualization as JSON for Three.js/D3.js frontends (cadAI format).

    Returns dict with points, categories, and metadata.
    Scale adjusts coordinates (default 10000 for Three.js).
    """
    # Find current range to rescale
    if not viz.points:
        return {"points": [], "categories": {}, "total_count": 0, "method": viz.reduction_method.value}

    xs = [p.x for p in viz.points]
    ys = [p.y for p in viz.points]
    zs = [p.z for p in viz.points]
    max_abs = max(
        max(abs(v) for v in xs) if xs else 1,
        max(abs(v) for v in ys) if ys else 1,
        max(abs(v) for v in zs) if zs else 1,
    )
    factor = scale / max_abs if max_abs > 0 else 1.0

    # Build category color map
    all_cats = sorted(set(p.category for p in viz.points if p.category))
    cat_colors = {cat: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cat in enumerate(all_cats)}

    # Build points
    points = []
    cat_counts: Dict[str, int] = {}
    for i, p in enumerate(viz.points):
        cat_counts[p.category] = cat_counts.get(p.category, 0) + 1
        points.append({
            "id": p.label or f"point_{i}",
            "x": round(p.x * factor, 2),
            "y": round(p.y * factor, 2),
            "z": round(p.z * factor, 2),
            "category": p.category,
            "color": cat_colors.get(p.category, "#666666"),
            "cluster": p.cluster,
            "is_anomaly": p.is_anomaly,
            "anomaly_score": round(p.anomaly_score, 4),
            "preview": str(p.metadata.get("preview", p.label))[:200],
        })

    categories = {
        cat: {"count": cat_counts.get(cat, 0), "color": cat_colors.get(cat, "#666666")}
        for cat in all_cats
    }

    return {
        "points": points,
        "categories": categories,
        "total_count": len(points),
        "method": viz.reduction_method.value,
        "n_dimensions": viz.n_dimensions,
        "total_anomalies": viz.total_anomalies,
        "clusters": [c.to_dict() for c in viz.clusters],
    }


# ── Similarity network graph ────────────────────────────────────────────


def plot_similarity_network(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    threshold: float = 0.65,
    top_k: int = 3,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 150,
    max_nodes: int = 500,
) -> str:
    """Cosine similarity network graph (ported from cadAI vectorstore_graph.py).

    Builds a graph where nodes are documents and edges connect similar pairs.
    Returns path to saved PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    embeddings = np.asarray(embeddings, dtype=np.float64)
    N = min(len(embeddings), max_nodes)
    embeddings = embeddings[:N]

    if labels is None:
        labels = [f"doc_{i}" for i in range(N)]
    else:
        labels = labels[:N]

    if categories is None:
        categories = ["default"] * N
    else:
        categories = categories[:N]

    # L2-normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms

    # Cosine similarity matrix
    sim = normed @ normed.T
    np.fill_diagonal(sim, 0)
    sim[sim < threshold] = 0

    # Build adjacency: top-k edges per node
    edges = []
    edge_weights = []
    for i in range(N):
        row = sim[i]
        top_idx = np.argsort(row)[-top_k:]
        for j in top_idx:
            if row[j] > 0 and i < j:
                edges.append((i, j))
                edge_weights.append(float(row[j]))

    # Layout: spring (force-directed)
    # Simple Fruchterman-Reingold via sklearn-like approach
    # Use networkx if available, else basic force layout
    try:
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(N))
        for (i, j), w in zip(edges, edge_weights):
            G.add_edge(i, j, weight=w)
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    except ImportError:
        # Fallback: random layout
        rng = np.random.RandomState(42)
        pos = {i: rng.randn(2) for i in range(N)}

    # Unique categories for coloring
    unique_cats = sorted(set(categories))
    cat_color_map = {cat: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cat in enumerate(unique_cats)}

    fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Draw edges
    for (i, j), w in zip(edges, edge_weights):
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        ax.plot([x0, x1], [y0, y1], color="white", alpha=w * 0.5, linewidth=0.5)

    # Draw nodes
    for cat in unique_cats:
        idxs = [i for i in range(N) if categories[i] == cat]
        if not idxs:
            continue
        ax.scatter(
            [pos[i][0] for i in idxs],
            [pos[i][1] for i in idxs],
            c=cat_color_map[cat], s=30, alpha=0.8,
            label=f"{cat} ({len(idxs)})", edgecolors="none",
        )

    ax.set_title("Similarity Network", color="white", fontsize=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    legend = ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    legend.get_frame().set_facecolor("#1a1a2e")
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()

    if not save_path:
        save_path = str(Path(tempfile.gettempdir()) / "omnibedding_network.png")
    fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved network graph: %s (%d nodes, %d edges)", save_path, N, len(edges))
    return save_path
