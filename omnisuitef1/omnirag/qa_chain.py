"""RAG Q&A chain: retrieve → build prompt → generate → cite sources.

Combines cadAI EngineeringQA prompt building with F1 chat_server citation format.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from omnirag._types import ChatMessage, ChatResponse, SearchResult
from omnirag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions using the provided context.

Instructions:
1. Use ONLY the provided context to answer. If the context doesn't contain enough information, say so clearly.
2. Cite source documents and page numbers when available.
3. Be precise with numerical values, units, and specifications.
4. Keep answers concise but thorough. Use bullet points for lists.
5. If a question is ambiguous, briefly clarify what you're answering."""


class RAGChain:
    """Full RAG pipeline: retrieve context → build prompt → generate answer.

    Usage:
        chain = RAGChain(retriever)
        response = chain.ask("What is the maximum operating pressure?")
        print(response.answer, response.sources)
    """

    def __init__(
        self,
        retriever: RAGRetriever,
        llm_provider: str = "auto",
        system_prompt: Optional[str] = None,
    ):
        self._retriever = retriever
        self._llm_provider = llm_provider
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def ask(
        self,
        question: str,
        k: int = 5,
        category: Optional[str] = None,
        include_sources: bool = True,
    ) -> ChatResponse:
        """Ask a question with RAG context.

        1. Retrieve top-k relevant documents
        2. Build context string with numbered citations
        3. Generate answer via LLM
        4. Return answer + sources
        """
        # 1. Retrieve
        results = self._retriever.search_enhanced(
            question, k=k, min_score=0.3, category=category,
        )

        # 2. Build context
        context_blocks = []
        sources = []
        for r in results:
            meta = r.document.metadata
            source_file = meta.get("source", meta.get("source_file", "unknown"))
            category_val = meta.get("category", "")
            page = meta.get("page", "")
            data_type = meta.get("data_type", "")

            label = data_type or category_val or "document"
            header = f"[{r.rank}] ({label})"
            if page:
                header += f" Page {page}"
            header += f" — {source_file}"

            context_blocks.append(f"{header}\n{r.document.content}")

            if include_sources:
                sources.append({
                    "file": source_file,
                    "category": category_val,
                    "data_type": data_type,
                    "page": page,
                    "score": round(r.score, 4),
                    "preview": r.document.content[:200],
                })

        context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        # 3. Build messages
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._build_user_message(question, context)},
        ]

        # 4. Generate
        from omnirag.llm import generate
        answer, model_name = generate(messages, provider=self._llm_provider)

        return ChatResponse(
            answer=answer,
            sources=sources,
            model_used=model_name,
        )

    def ask_with_history(
        self,
        question: str,
        history: List[ChatMessage],
        k: int = 5,
        category: Optional[str] = None,
        data_types: Optional[List[str]] = None,
    ) -> ChatResponse:
        """Ask with conversation history included in the prompt."""
        # 1. Retrieve
        results = self._retriever.search_enhanced(
            question, k=k, min_score=0.3, category=category, data_types=data_types,
        )

        # 2. Build context
        context_blocks = []
        sources = []
        for r in results:
            meta = r.document.metadata
            source_file = meta.get("source", meta.get("source_file", "unknown"))
            category_val = meta.get("category", "")
            page = meta.get("page", "")

            context_blocks.append(
                f"[{r.rank}] (Page {page}) — {source_file}\n{r.document.content}"
            )
            sources.append({
                "file": source_file,
                "category": category_val,
                "page": page,
                "score": round(r.score, 4),
                "preview": r.document.content[:200],
            })

        context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        # 3. Build messages with history
        messages = [{"role": "system", "content": self._system_prompt}]

        # Add conversation history (last 10 turns)
        for msg in history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({
            "role": "user",
            "content": self._build_user_message(question, context),
        })

        # 4. Generate
        from omnirag.llm import generate
        answer, model_name = generate(messages, provider=self._llm_provider)

        return ChatResponse(
            answer=answer,
            sources=sources,
            model_used=model_name,
        )

    def _build_user_message(self, question: str, context: str) -> str:
        """Format user message with context block."""
        return (
            f"CONTEXT FROM KNOWLEDGE BASE:\n{context}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            f"Answer based on the context above. Cite sources where applicable."
        )
