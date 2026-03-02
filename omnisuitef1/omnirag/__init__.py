"""OmniRAG — unified retrieval-augmented generation service.

Combines best patterns from cadAI (enhanced retrieval, multi-LLM),
F1 (MongoDB vector search, CLIP visual search), and omnidoc (embeddings, loaders).

Quick start:
    from omnirag import get_vectorstore, RAGRetriever, RAGChain, ingest_file

    # Ingest documents
    store = get_vectorstore()
    ingest_file("document.pdf", store)

    # Search
    from omnidoc.embedder import get_embedder
    retriever = RAGRetriever(store, get_embedder().embed_query)
    results = retriever.search_enhanced("what is the max pressure?")

    # Q&A with citations
    chain = RAGChain(retriever)
    response = chain.ask("what is the max pressure?")
    print(response.answer, response.sources)
"""

from omnirag.vectorstore import (
    get_vectorstore, AtlasStore, ChromaStore, VectorStoreProtocol,
)
from omnirag.retriever import RAGRetriever
from omnirag.qa_chain import RAGChain
from omnirag.conversation import ConversationManager
from omnirag.llm import get_llm, generate, LLMConfig
from omnirag.ingest import ingest_file, ingest_texts, ingest_directory

from omnirag._types import (
    RAGDocument, SearchResult, ChatMessage, ChatResponse, IngestResult,
)

__all__ = [
    "get_vectorstore", "AtlasStore", "ChromaStore", "VectorStoreProtocol",
    "RAGRetriever",
    "RAGChain",
    "ConversationManager",
    "get_llm", "generate", "LLMConfig",
    "ingest_file", "ingest_texts", "ingest_directory",
    "RAGDocument", "SearchResult", "ChatMessage", "ChatResponse", "IngestResult",
]
