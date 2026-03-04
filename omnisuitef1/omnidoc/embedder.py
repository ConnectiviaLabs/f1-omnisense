"""Dual embedding engine: BGE-large (text, 1024-dim) + CLIP ViT-B/32 (image+text, 512-dim).

BGE: High-quality text embeddings for document chunk search.
CLIP: Cross-modal embeddings enabling text-to-image search.

Usage:
    from omnidoc.embedder import get_embedder

    emb = get_embedder()

    # Text embeddings (BGE, 1024-dim)
    vecs = emb.embed_texts(["piping design criteria", "ASME B31.1"])

    # Image embeddings (CLIP, 512-dim)
    vecs = emb.embed_images([Path("page1.png")])

    # Cross-modal: text query → image search (CLIP, 512-dim)
    vec = emb.embed_text_for_image_search("pump discharge nozzle diagram")
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── BGE text embeddings (1024-dim) ────────────────────────────────────────

class BGEEmbedder:
    """Text embeddings using BAAI/bge-large-en-v1.5 (1024-dim).

    Downloads model on first use (~1.3GB), cached in ~/.cache/huggingface/.
    """

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    DIMENSION = 1024

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run:\n"
                "  pip install sentence-transformers"
            )
        self._model = SentenceTransformer(self.MODEL_NAME)
        logger.info("BGE %s: loaded (%s)", self.MODEL_NAME, self._model.device)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts into 1024-dim vectors."""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query."""
        embeddings = self._model.encode(
            [f"Represent this sentence for searching relevant passages: {text}"],
            normalize_embeddings=True,
        )
        return embeddings[0].tolist()

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]


# ── CLIP image+text embeddings (512-dim) ─────────────────────────────────

class CLIPEmbedder:
    """Cross-modal embeddings using CLIP ViT-B/32 (512-dim).

    Images and text are embedded into the same vector space,
    enabling: "find diagrams showing pump connections" → matching images.
    """

    DIMENSION = 512

    def __init__(self):
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "CLIP dependencies not installed. Run:\n"
                "  pip install open-clip-torch pillow"
            )

        self._torch = torch
        self._Image = Image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(self.device).eval()
        logger.info("CLIP ViT-B/32: loaded (%s)", self.device)

    def embed_images(self, image_paths: List[Path]) -> List[List[float]]:
        """Embed images into 512-dim CLIP vectors."""
        import torch

        vectors = []
        for path in image_paths:
            img = self._Image.open(path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(img_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                vectors.append(features[0].cpu().tolist())
        return vectors

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts into 512-dim CLIP vectors (same space as images)."""
        import torch

        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().tolist()

    def embed_image(self, path: Path) -> List[float]:
        return self.embed_images([path])[0]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


# ── Unified engine ────────────────────────────────────────────────────────

class UnifiedEmbedder:
    """Dual engine: BGE for text search + CLIP for image search.

    Lazy-initializes each model on first use.
    """

    def __init__(self, enable_bge: bool = True, enable_clip: bool = True):
        self._enable_bge = enable_bge
        self._enable_clip = enable_clip
        self._bge: Optional[BGEEmbedder] = None
        self._clip: Optional[CLIPEmbedder] = None

    @property
    def bge(self) -> BGEEmbedder:
        if self._bge is None:
            if not self._enable_bge:
                raise RuntimeError("BGE embedder is disabled")
            self._bge = BGEEmbedder()
        return self._bge

    @property
    def clip(self) -> CLIPEmbedder:
        if self._clip is None:
            if not self._enable_clip:
                raise RuntimeError("CLIP embedder is disabled")
            self._clip = CLIPEmbedder()
        return self._clip

    # ── Sync text (BGE 1024-dim) ──

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.bge.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.bge.embed_query(text)

    # ── Sync images (CLIP 512-dim) ──

    def embed_images(self, paths: List[Path]) -> List[List[float]]:
        return self.clip.embed_images(paths)

    def embed_text_for_image_search(self, text: str) -> List[float]:
        return self.clip.embed_text(text)

    # ── Async wrappers ──

    async def async_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_texts, texts)

    async def async_embed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def async_embed_images(self, paths: List[Path]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_images, paths)

    async def async_embed_text_for_image_search(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_text_for_image_search, text)


# ── Singleton ─────────────────────────────────────────────────────────────

_instance: Optional[UnifiedEmbedder] = None


def get_embedder(enable_bge: bool = True, enable_clip: bool = True) -> UnifiedEmbedder:
    """Get or create the singleton UnifiedEmbedder."""
    global _instance
    if _instance is None:
        _instance = UnifiedEmbedder(enable_bge=enable_bge, enable_clip=enable_clip)
    return _instance
