"""
Batch PDF Ingestion into f1_knowledge Vectorstore
──────────────────────────────────────────────────
Uses OmniDoc for text extraction + chunking, BGE-large for 1024-dim
embeddings, and AtlasVectorStore for storage.

Auto-categorizes PDFs by filename pattern:
  - FIA_*_Technical*  → "fia_technical_regulation"
  - FIA_*_Sporting*   → "fia_sporting_regulation"
  - FIA_*_Financial*  → "fia_financial_regulation"
  - FIA_*_Operational* → "fia_operational_regulation"
  - FIA_*_General*    → "fia_general_provisions"
  - McLaren_*         → "mclaren_corporate"

Usage:
    python3 -m pipeline.enrichment.ingest_pdfs
    python3 -m pipeline.enrichment.ingest_pdfs --rebuild  # drop collection first
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "omnisuitef1"))

PDF_DIR = Path(__file__).resolve().parents[2] / "data" / "mclaren_pdfs"


def categorize_pdf(filename: str) -> tuple[str, str]:
    """Return (category, data_type) based on filename pattern."""
    fn = filename.lower()
    if "technical_regulation" in fn:
        return "fia_technical_regulation", "regulation"
    elif "sporting_regulation" in fn:
        return "fia_sporting_regulation", "regulation"
    elif "financial_regulation" in fn:
        return "fia_financial_regulation", "regulation"
    elif "operational_regulation" in fn:
        return "fia_operational_regulation", "regulation"
    elif "general_provision" in fn:
        return "fia_general_provisions", "regulation"
    elif "mclaren" in fn:
        return "mclaren_corporate", "corporate_document"
    else:
        return "f1_document", "document"


def extract_year(filename: str) -> str | None:
    """Extract year from filename like FIA_2025_..."""
    import re
    m = re.search(r"(\d{4})", filename)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Drop collection first")
    args = parser.parse_args()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"  No PDFs found in {PDF_DIR}")
        return

    print(f"  Found {len(pdfs)} PDFs in {PDF_DIR.name}/")
    for p in pdfs:
        cat, _ = categorize_pdf(p.name)
        print(f"    {p.name} → {cat}")

    # Import heavy deps after listing files
    from omnidoc.ingest import process_document
    from omnidoc.embedder import get_embedder
    from langchain_core.documents import Document
    from pipeline.vectorstore import AtlasVectorStore

    # Init embedder (BGE 1024-dim, no CLIP needed)
    print("\n  Loading BGE embedder...")
    embedder = get_embedder(enable_clip=False)

    # Init vectorstore
    vs = AtlasVectorStore()
    if args.rebuild:
        print("  Dropping existing collection (--rebuild)...")
        vs.delete_collection()

    total_chunks = 0
    total_tables = 0

    for pi, pdf_path in enumerate(pdfs):
        print(f"\n{'='*60}")
        print(f"  [{pi+1}/{len(pdfs)}] {pdf_path.name}")
        print(f"{'='*60}")

        category, data_type = categorize_pdf(pdf_path.name)
        year = extract_year(pdf_path.name)

        # Process with OmniDoc (skip embeddings — we do BGE separately)
        t0 = time.time()
        try:
            result = process_document(
                pdf_path,
                embed=False,
                extract_tables=True,
                extract_images=False,  # skip images to save space
                deep_extract=False,
                save_json=False,
                chunk_size=1000,
                chunk_overlap=200,
            )
        except Exception as e:
            print(f"  ERROR processing {pdf_path.name}: {e}")
            continue

        if not result.chunks:
            print(f"  WARNING: No text extracted from {pdf_path.name}")
            continue

        proc_time = time.time() - t0
        print(f"  Extracted: {len(result.chunks)} chunks, {len(result.tables)} tables ({proc_time:.1f}s)")

        # Build LangChain Documents with metadata
        docs = []
        for i, chunk_text in enumerate(result.chunks):
            meta = result.chunk_metadata[i] if i < len(result.chunk_metadata) else {}
            doc_meta = {
                "category": category,
                "data_type": data_type,
                "source": pdf_path.name,
                "page": meta.get("page", 0),
                "chunk": i + 1,
                "total_chunks": len(result.chunks),
            }
            if year:
                doc_meta["year"] = year

            # Carry over any enriched metadata from OmniDoc
            for key in ("equipment_tags", "line_refs", "standards", "section"):
                if meta.get(key):
                    doc_meta[key] = meta[key]

            docs.append(Document(page_content=chunk_text, metadata=doc_meta))

        # Embed with BGE (1024-dim)
        t0 = time.time()
        texts = [doc.page_content for doc in docs]
        embeddings = []
        BATCH = 32
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            embeddings.extend(embedder.embed_texts(batch))
        embed_time = time.time() - t0
        print(f"  Embedded: {len(embeddings)} vectors ({embed_time:.1f}s)")

        # Upsert to Atlas
        count = vs.upsert_documents(docs, embeddings)
        total_chunks += count
        total_tables += len(result.tables)
        print(f"  Inserted: {count} docs into f1_knowledge")

    # Summary
    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  PDFs processed: {len(pdfs)}")
    print(f"  Total chunks:   {total_chunks}")
    print(f"  Total tables:   {total_tables}")
    print(f"  Collection:     f1_knowledge ({vs.count()} total docs)")

    # Try to ensure vector index
    print("\n  Ensuring vector search index (1024-dim, cosine)...")
    vs.ensure_vector_index()


if __name__ == "__main__":
    main()
