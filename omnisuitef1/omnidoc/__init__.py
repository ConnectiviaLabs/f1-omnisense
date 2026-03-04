"""OmniDoc — Unified document ingestion combining best of OmniSense, cadAI, and F1.

Features:
- 12+ format support (PDF, HTML, DOCX, CSV, JSON, XML, RTF, ODT, Excel, PPT, images, MD)
- BGE-large-en-v1.5 text embeddings (1024-dim)
- CLIP ViT-B/32 cross-modal image embeddings (512-dim)
- pdfplumber + Camelot table extraction (lattice → stream fallback)
- 5-pass Groq Vision deep extraction for engineering PDFs
- Multi-view PDF rendering (3x zoom, quadrants, zoom regions)
"""

from omnidoc.ingest import process_document, ProcessedDocument

__all__ = ["process_document", "ProcessedDocument"]
