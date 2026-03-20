"""Composable context pipeline — ordered context source assembly for RAG prompts.

Inspired by DataSense kex_chat.py multi-layer retrieval (4 layers: vector search,
KEx embeddings, cross-domain context, conversation history). omniRAG generalizes
this: each "layer" is a ContextSource callable. Users register sources; the
pipeline calls them in order and assembles the final context string.

Usage:
    pipeline = ContextPipeline()
    pipeline.register("vector_search", retriever_as_source(retriever), priority=0)
    pipeline.register("kex_insights", my_kex_fn, priority=10)

    blocks = pipeline.gather("What caused the anomaly?")
    context = pipeline.format(blocks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from omnirag._types import SearchResult

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass
class ContextBlock:
    """A single block of context from one source."""
    label: str
    content: str
    source_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "content": self.content,
            "source_type": self.source_type,
            "metadata": self.metadata,
        }


# Signature: (query: str, **kwargs) -> List[ContextBlock]
ContextSourceFn = Callable[..., List[ContextBlock]]


# ── Pipeline ─────────────────────────────────────────────────────────────────

class ContextPipeline:
    """Ordered list of context sources assembled into a prompt context string.

    Each source is a callable returning List[ContextBlock]. Sources run in
    priority order (lower = first). A failing source is skipped with a warning.
    """

    def __init__(self):
        self._sources: Dict[str, Tuple[int, ContextSourceFn]] = {}

    def register(
        self,
        name: str,
        source_fn: ContextSourceFn,
        priority: int = 50,
    ):
        """Register a named context source. Duplicate names replace."""
        self._sources[name] = (priority, source_fn)

    def unregister(self, name: str):
        """Remove a context source by name."""
        self._sources.pop(name, None)

    def gather(self, query: str, **kwargs) -> List[ContextBlock]:
        """Run all sources in priority order and collect blocks.

        Each source is called with (query, **kwargs). Failures are logged
        and skipped — one broken source does not break the pipeline.
        """
        ordered = sorted(self._sources.items(), key=lambda x: x[1][0])
        all_blocks: List[ContextBlock] = []

        for name, (_, source_fn) in ordered:
            try:
                blocks = source_fn(query, **kwargs)
                all_blocks.extend(blocks)
            except Exception as exc:
                logger.warning("Context source '%s' failed: %s", name, exc)

        return all_blocks

    def format(
        self,
        blocks: List[ContextBlock],
        separator: str = "\n---\n",
    ) -> str:
        """Assemble blocks into a single context string with headers."""
        if not blocks:
            return "No relevant context found."

        parts = []
        for block in blocks:
            header = f"[{block.source_type}]" if block.source_type else ""
            if block.label:
                header = f"{header} {block.label}".strip()
            parts.append(f"{header}\n{block.content}" if header else block.content)

        return separator.join(parts)

    def list_sources(self) -> List[str]:
        """Return registered source names in priority order."""
        ordered = sorted(self._sources.items(), key=lambda x: x[1][0])
        return [name for name, _ in ordered]


# ── Built-in helper ──────────────────────────────────────────────────────────

def retriever_as_source(
    retriever,
    k: int = 5,
    min_score: float = 0.3,
) -> ContextSourceFn:
    """Wrap an existing RAGRetriever as a ContextPipeline source.

    Args:
        retriever: RAGRetriever instance.
        k: Number of results to retrieve.
        min_score: Minimum similarity score.

    Returns:
        A callable compatible with ContextPipeline.register().
    """
    def _source(query: str, **kwargs) -> List[ContextBlock]:
        category = kwargs.get("category")
        results = retriever.search_enhanced(
            query, k=k, min_score=min_score, category=category,
        )
        blocks = []
        for r in results:
            meta = r.document.metadata
            source_file = meta.get("source", meta.get("source_file", "unknown"))
            category_val = meta.get("category", "")
            page = meta.get("page", "")

            label = f"[{r.rank}] {source_file}"
            if category_val:
                label += f" | {category_val}"
            if page:
                label += f" | Page {page}"
            label += f" | Relevance: {r.score:.2f}"

            blocks.append(ContextBlock(
                label=label,
                content=r.document.content,
                source_type="vector_search",
                metadata={"score": r.score, "rank": r.rank, **meta},
            ))
        return blocks

    return _source
