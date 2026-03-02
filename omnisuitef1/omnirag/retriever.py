"""Enhanced semantic retrieval with cliff detection, query expansion, and deduplication.

Ported from cadAI EngineeringRetriever with domain-agnostic query expansion.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from omnirag._types import RAGDocument, SearchResult
from omnirag.vectorstore import VectorStoreProtocol

logger = logging.getLogger(__name__)


class RAGRetriever:
    """High-level retriever wrapping any VectorStoreProtocol backend.

    Features:
      - Enhanced search with cliff detection (stops when relevance drops sharply)
      - Query expansion via configurable regex patterns
      - Content-hash deduplication
      - Min-score filtering

    Usage:
        retriever = RAGRetriever(vectorstore, embed_fn)
        results = retriever.search_enhanced("what is the max pressure?")
    """

    def __init__(
        self,
        vectorstore: VectorStoreProtocol,
        embed_fn: Callable[[str], List[float]],
        expand_patterns: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Args:
            vectorstore: Backend implementing VectorStoreProtocol.
            embed_fn: Function that embeds a query string → vector.
            expand_patterns: List of (regex, template) for query expansion.
                Template uses {match} placeholder. Example:
                [(r'\\d+-[A-Z]-\\d+', '{match} specifications details')]
        """
        self._store = vectorstore
        self._embed = embed_fn
        self._expand_patterns = expand_patterns or []

    def search(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
    ) -> List[SearchResult]:
        """Basic similarity search."""
        query_vec = self._embed(query)
        filt = {"category": category} if category else None
        raw = self._store.similarity_search(query_vec, k=k, filter=filt)

        return [
            SearchResult(document=doc, score=score, rank=i + 1)
            for i, (doc, score) in enumerate(raw)
        ]

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search returning scored results."""
        return self.search(query, k=k, category=category)

    def search_enhanced(
        self,
        query: str,
        k: int = 8,
        min_score: float = 0.3,
        cliff_threshold: float = 0.15,
        category: Optional[str] = None,
        data_types: Optional[List[str]] = None,
        expand_query: bool = True,
    ) -> List[SearchResult]:
        """Production search with cliff detection, dedup, and query expansion.

        Algorithm:
          1. Primary vector search (k=10)
          2. Query expansion: generate variant queries, merge results
          3. Deduplicate by content hash (first 200 chars)
          4. Sort by score descending
          5. Filter by min_score
          6. Cliff detection: stop when score drops > cliff_threshold
          7. Return top k results
        """
        # 1. Primary search
        query_vec = self._embed(query)
        filt: Optional[Dict[str, Any]] = {}
        if category:
            filt["category"] = category
        if data_types:
            filt["data_type"] = {"$in": data_types}
        if not filt:
            filt = None
        primary = self._store.similarity_search(query_vec, k=10, filter=filt)

        all_results: List[Tuple[RAGDocument, float]] = list(primary)

        # 2. Query expansion
        if expand_query:
            expanded = self._expand_query(query)
            for eq in expanded:
                eq_vec = self._embed(eq)
                extra = self._store.similarity_search(eq_vec, k=5, filter=filt)
                all_results.extend(extra)

        # 3. Deduplicate by content hash (first 200 chars)
        seen_hashes = set()
        deduped = []
        for doc, score in all_results:
            h = hash(doc.content[:200])
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append((doc, score))

        # 4. Sort by score descending
        deduped.sort(key=lambda x: x[1], reverse=True)

        # 5. Min-score filter
        filtered = [(doc, s) for doc, s in deduped if s >= min_score]

        # 6. Cliff detection
        if len(filtered) > 1:
            clipped = [filtered[0]]
            for i in range(1, len(filtered)):
                score_drop = filtered[i - 1][1] - filtered[i][1]
                if score_drop > cliff_threshold:
                    break
                clipped.append(filtered[i])
            filtered = clipped

        # 7. Return top k
        filtered = filtered[:k]

        return [
            SearchResult(document=doc, score=score, rank=i + 1)
            for i, (doc, score) in enumerate(filtered)
        ]

    def get_relevant_context(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.3,
    ) -> str:
        """Retrieve and format context string for LLM prompts.

        Returns markdown-formatted context with source attribution.
        """
        results = self.search_enhanced(query, k=k, min_score=min_score)
        if not results:
            return "No relevant context found."

        blocks = []
        for r in results:
            meta = r.document.metadata
            source = meta.get("source", meta.get("source_file", "unknown"))
            category = meta.get("category", "")
            page = meta.get("page", "")

            header = f"[Source: {source}"
            if category:
                header += f" | Category: {category}"
            if page:
                header += f" | Page: {page}"
            header += f" | Relevance: {r.score:.2f}]"

            blocks.append(f"{header}\n{r.document.content}")

        return "\n---\n".join(blocks)

    def _expand_query(self, query: str) -> List[str]:
        """Generate expanded queries from configurable regex patterns.

        Also applies generic transformations:
          - "what is X?" → "X"
          - "how does X work?" → "X mechanism operation"
        """
        expanded = []

        # Generic transformations
        lower = query.lower().strip()
        if lower.startswith("what is ") or lower.startswith("what are "):
            noun = re.sub(r"^what (?:is|are) ", "", lower).rstrip("?").strip()
            if noun:
                expanded.append(noun)

        if lower.startswith("how does ") or lower.startswith("how do "):
            subject = re.sub(r"^how (?:does|do) ", "", lower).rstrip("?").strip()
            if subject:
                expanded.append(f"{subject} mechanism operation")

        # Configurable domain-specific patterns
        for pattern, template in self._expand_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                expanded.append(template.format(match=match))

        return expanded[:3]  # Max 3 expansions
