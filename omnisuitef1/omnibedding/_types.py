"""Type definitions for omnibedding — embedding visualization module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ReductionMethod(str, Enum):
    UMAP = "umap"
    TSNE = "tsne"
    PCA = "pca"


class ClusterMethod(str, Enum):
    KMEANS = "kmeans"
    DBSCAN = "dbscan"


# 10-color palette from DataSense EmbeddingsScatterPlot
CLUSTER_COLORS = [
    "#00BCD4",  # cyan
    "#4CAF50",  # green
    "#9C27B0",  # purple
    "#FF9800",  # orange
    "#E91E63",  # pink
    "#FFEB3B",  # yellow
    "#009688",  # teal
    "#F44336",  # red
    "#CDDC39",  # lime
    "#3F51B5",  # indigo
]


@dataclass
class EmbeddingPoint:
    """A single point in reduced embedding space."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    cluster: int = -1
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    distance: float = 0.0
    label: str = ""
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "cluster": self.cluster,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 4),
            "distance": round(self.distance, 4),
            "label": self.label,
            "category": self.category,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EmbeddingPoint:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ClusterInfo:
    """Statistics for a single cluster."""

    cluster_id: int = -1
    centroid: List[float] = field(default_factory=list)
    size: int = 0
    anomaly_count: int = 0
    avg_distance: float = 0.0
    std_distance: float = 0.0
    density: float = 0.0
    categories: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "centroid": [round(c, 4) for c in self.centroid],
            "size": self.size,
            "anomaly_count": self.anomaly_count,
            "avg_distance": round(self.avg_distance, 4),
            "std_distance": round(self.std_distance, 4),
            "density": round(self.density, 4),
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ClusterInfo:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EmbeddingViz:
    """Complete visualization result from the reduce → cluster → detect pipeline."""

    points: List[EmbeddingPoint] = field(default_factory=list)
    clusters: List[ClusterInfo] = field(default_factory=list)
    reduction_method: ReductionMethod = ReductionMethod.PCA
    cluster_method: ClusterMethod = ClusterMethod.KMEANS
    n_dimensions: int = 3
    explained_variance: float = 0.0
    total_points: int = 0
    total_anomalies: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "clusters": [c.to_dict() for c in self.clusters],
            "reduction_method": self.reduction_method.value,
            "cluster_method": self.cluster_method.value,
            "n_dimensions": self.n_dimensions,
            "explained_variance": round(self.explained_variance, 4),
            "total_points": self.total_points,
            "total_anomalies": self.total_anomalies,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EmbeddingViz:
        points = [EmbeddingPoint.from_dict(p) for p in d.get("points", [])]
        clusters = [ClusterInfo.from_dict(c) for c in d.get("clusters", [])]
        return cls(
            points=points,
            clusters=clusters,
            reduction_method=ReductionMethod(d.get("reduction_method", "pca")),
            cluster_method=ClusterMethod(d.get("cluster_method", "kmeans")),
            n_dimensions=d.get("n_dimensions", 3),
            explained_variance=d.get("explained_variance", 0.0),
            total_points=d.get("total_points", len(points)),
            total_anomalies=d.get("total_anomalies", 0),
        )
