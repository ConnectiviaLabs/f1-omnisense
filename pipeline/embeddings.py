"""Dual embedding pipeline: CLIP ViT-B/32 (image+text) + nomic-embed-text (text).

CLIP: Encodes both images and text into the same 512-dim vector space,
      enabling cross-modal search (find diagrams by text description).

Nomic: 768-dim text embeddings via Nomic API,
       optimized for document search.

Usage:
    from pipeline.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()

    # Text embeddings (nomic)
    vecs = engine.embed_texts(["pipe routing algorithm", "ASME B31.1"])

    # Image embeddings (CLIP)
    vecs = engine.embed_images([Path("page1.png"), Path("diagram.png")])

    # Cross-modal: text query against image index
    query_vec = engine.embed_text_for_image_search("pump discharge nozzle")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests


# ── Nomic text embeddings (via Nomic API) ─────────────────────────────

class NomicEmbedder:
    """Text embeddings using nomic-embed-text-v1.5 via the Nomic API.

    768-dim vectors, optimized for document/technical search.
    Requires NOMIC_API_KEY environment variable.
    """

    API_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
    MODEL = "nomic-embed-text-v1.5"

    def __init__(self):
        self._api_key = os.getenv("NOMIC_API_KEY")
        if not self._api_key:
            raise ValueError("NOMIC_API_KEY environment variable is required")
        print(f"  Nomic {self.MODEL}: API mode")

    def _call_api(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Call the Nomic embedding API."""
        resp = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.MODEL,
                "texts": texts,
                "task_type": task_type,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings into 768-dim vectors."""
        return self._call_api(texts, task_type="search_document")

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (uses 'search_query' task for better retrieval)."""
        return self._call_api([text], task_type="search_query")[0]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]


# ── CLIP image+text embeddings ──────────────────────────────────────────

class CLIPEmbedder:
    """Cross-modal embeddings using CLIP ViT-B/32.

    512-dim vectors for both images and text in the same space.
    Enables: "find me a diagram showing pump connections" → matching images.
    """

    def __init__(self):
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError:
            raise ValueError(
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

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
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

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text into 512-dim CLIP vectors (same space as images)."""
        import torch

        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().tolist()

    def embed_image(self, path: Path) -> list[float]:
        """Embed a single image."""
        return self.embed_images([path])[0]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string (for image search)."""
        return self.embed_texts([text])[0]


# ── Unified engine ──────────────────────────────────────────────────────

class EmbeddingEngine:
    """Dual embedding engine combining nomic (text) + CLIP (image+text).

    Usage:
        engine = EmbeddingEngine()

        # Pure text search (nomic, 768-dim)
        vecs = engine.embed_texts(["equipment list", "piping class A1A"])

        # Image indexing (CLIP, 512-dim)
        vecs = engine.embed_images([Path("page1.png")])

        # Cross-modal text→image query (CLIP, 512-dim)
        query = engine.embed_text_for_image_search("pump nozzle diagram")
    """

    def __init__(self, enable_clip: bool = True, enable_nomic: bool = True):
        self.nomic = None
        self.clip = None

        if enable_nomic:
            try:
                self.nomic = NomicEmbedder()
                print("  Nomic-embed-text: ready (768-dim, text)")
            except (ValueError, Exception) as e:
                print(f"  Nomic: skipped ({e})")

        if enable_clip:
            try:
                self.clip = CLIPEmbedder()
                print(f"  CLIP ViT-B/32: ready (512-dim, image+text, {self.clip.device})")
            except (ValueError, Exception) as e:
                print(f"  CLIP: skipped ({e})")

    # ── Text embeddings (nomic) ─────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using nomic (768-dim). For document chunk search."""
        if not self.nomic:
            raise RuntimeError("Nomic embedder not available")
        return self.nomic.embed(texts)

    def embed_text(self, text: str) -> list[float]:
        """Embed single text using nomic."""
        return self.embed_texts([text])[0]

    # ── Image embeddings (CLIP) ─────────────────────────────────────────

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        """Embed images using CLIP (512-dim). For visual indexing."""
        if not self.clip:
            raise RuntimeError("CLIP embedder not available")
        return self.clip.embed_images(paths)

    def embed_image(self, path: Path) -> list[float]:
        """Embed single image using CLIP."""
        return self.embed_images([path])[0]

    # ── Cross-modal (CLIP text for image search) ────────────────────────

    def embed_text_for_image_search(self, text: str) -> list[float]:
        """Embed text into CLIP space (512-dim) for image retrieval."""
        if not self.clip:
            raise RuntimeError("CLIP embedder not available")
        return self.clip.embed_text(text)

    def embed_texts_for_image_search(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into CLIP space for image retrieval."""
        if not self.clip:
            raise RuntimeError("CLIP embedder not available")
        return self.clip.embed_texts(texts)
