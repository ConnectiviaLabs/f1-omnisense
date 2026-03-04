"""CLIP ViT-B/32 embeddings — image + text in same 512-dim space.

Adapted from F1 pipeline/embeddings.py CLIPEmbedder using open_clip
with laion2b_s34b_b79k weights (better than base OpenAI).

Usage:
    from omnisee.clip_embeddings import embed_clip, embed_text, auto_tag, search_by_text

    vec = embed_clip(frame)
    tags = auto_tag(frame, ["a photo of a car", "a photo of a person"])
    results = search_by_text("red car", image_embeddings)
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np

from omnivis._types import CLIPEmbedding

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """Singleton CLIP embedder using open_clip ViT-B/32."""

    _instance: Optional[CLIPEmbedder] = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError("CLIP dependencies missing. Run: pip install open-clip-torch pillow")

        self._torch = torch
        self._Image = Image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(self.device).eval()
        self._initialized = True
        logger.info("CLIP ViT-B/32 loaded on %s (512-dim)", self.device)

    def embed_image(self, image: Union[np.ndarray, Path, str]) -> List[float]:
        """Embed a single image (BGR ndarray or file path) into 512-dim."""
        return self.embed_images([image])[0]

    def embed_images(self, images: List[Union[np.ndarray, Path, str]]) -> List[List[float]]:
        """Embed multiple images into 512-dim vectors."""
        import torch
        vectors = []
        for img in images:
            pil_img = self._to_pil(img)
            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                vectors.append(features[0].cpu().tolist())
        return vectors

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into 512-dim (same space as images)."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts into 512-dim vectors."""
        import torch
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().tolist()

    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _to_pil(self, image):
        if isinstance(image, (str, Path)):
            return self._Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            import cv2
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return self._Image.fromarray(rgb)
        else:
            return image.convert("RGB")


def _get_embedder() -> CLIPEmbedder:
    return CLIPEmbedder()


# ── Convenience functions ─────────────────────────────────────────────────

def embed_clip(image: Union[np.ndarray, Path, str]) -> List[float]:
    """Embed image into 512-dim CLIP vector."""
    return _get_embedder().embed_image(image)


def embed_text(text: str) -> List[float]:
    """Embed text into 512-dim CLIP vector (same space as images)."""
    return _get_embedder().embed_text(text)


def auto_tag(
    image: Union[np.ndarray, Path, str],
    categories: List[str],
    top_k: int = 5,
) -> List[dict]:
    """Tag an image against category descriptions.

    Returns list of {"label": str, "score": float} sorted by score descending.
    """
    embedder = _get_embedder()
    img_vec = embedder.embed_image(image)
    cat_vecs = embedder.embed_texts(categories)

    scored = []
    for cat, vec in zip(categories, cat_vecs):
        score = embedder.similarity(img_vec, vec)
        scored.append({"label": cat, "score": round(score, 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def search_by_text(
    query: str,
    image_embeddings: List[List[float]],
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """Search image embeddings by text query.

    Returns list of (index, similarity_score) sorted by score descending.
    """
    embedder = _get_embedder()
    q_vec = np.array(embedder.embed_text(query))

    scores = []
    for i, emb in enumerate(image_embeddings):
        e = np.array(emb)
        sim = float(np.dot(q_vec, e) / (np.linalg.norm(q_vec) * np.linalg.norm(e) + 1e-9))
        scores.append((i, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
