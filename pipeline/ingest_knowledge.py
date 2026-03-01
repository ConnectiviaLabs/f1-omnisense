"""Ingest extracted F1 intelligence into MongoDB Atlas vector store.

Reads pipeline/output/intelligence.json, creates LangChain Documents,
embeds them with nomic-embed-text (768-dim), and upserts into Atlas.

Usage:
    python pipeline/ingest_knowledge.py
    python pipeline/ingest_knowledge.py --rebuild   # Drop collection first
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_core.documents import Document

INTELLIGENCE_PATH = Path(__file__).parent / "output" / "intelligence.json"


# ── Document Builders ────────────────────────────────────────────────────

def _build_rule_docs(rules: list[dict]) -> list[Document]:
    """One document per regulation rule."""
    docs = []
    for r in rules:
        desc = r.get("description") or ""
        category = r.get("category") or "General"
        rule_id = r.get("id") or "unknown"
        severity = r.get("severity") or "info"
        value = r.get("value") or ""
        unit = r.get("unit") or ""
        condition = r.get("condition") or ""
        reference = r.get("reference") or ""
        standard = r.get("source_standard") or ""

        text_parts = [f"[Regulation {rule_id}] {category}"]
        if desc:
            text_parts.append(desc)
        if value:
            text_parts.append(f"Value: {value} {unit}".strip())
        if condition:
            text_parts.append(f"Condition: {condition}")
        if standard:
            text_parts.append(f"Standard: {standard}")
        text_parts.append(f"Severity: {severity}")

        text = "\n".join(text_parts)
        if len(text) < 20:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "data_type": "regulation",
                "category": category,
                "rule_id": rule_id,
                "severity": severity,
                "source": r.get("_source", ""),
                "page": r.get("_page", 0),
            },
        ))
    return docs


def _build_equipment_docs(equipment: list[dict]) -> list[Document]:
    """One document per equipment item."""
    docs = []
    for e in equipment:
        tag = e.get("tag") or "unknown"
        etype = e.get("type") or ""
        desc = e.get("description") or ""
        location = e.get("location_description") or ""

        text_parts = [f"[Equipment] {tag}"]
        if etype:
            text_parts.append(f"Type: {etype}")
        if desc:
            text_parts.append(desc)
        if location:
            text_parts.append(f"Location: {location}")

        specs = e.get("specs", {})
        if specs:
            spec_str = ", ".join(f"{k}={v}" for k, v in specs.items() if v)
            if spec_str:
                text_parts.append(f"Specs: {spec_str}")

        text = "\n".join(text_parts)
        if len(text) < 20:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "data_type": "equipment",
                "category": etype or "equipment",
                "tag": tag,
                "source": e.get("_source", ""),
                "page": e.get("_page", 0),
            },
        ))
    return docs


def _build_dimension_docs(dims: list[dict]) -> list[Document]:
    """One document per dimensional specification."""
    docs = []
    for d in dims:
        component = d.get("component") or "unknown"
        dimension = d.get("dimension") or ""
        value = d.get("value")
        unit = d.get("unit") or ""

        text_parts = [f"[Dimension] {component}"]
        if dimension:
            text_parts.append(dimension)
        if value is not None:
            text_parts.append(f"Value: {value} {unit}".strip())

        text = "\n".join(text_parts)
        if len(text) < 15:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "data_type": "dimension",
                "category": "dimensional_spec",
                "component": component,
                "source": d.get("_source", ""),
                "page": d.get("_page", 0),
            },
        ))
    return docs


def _build_material_docs(mats: list[dict]) -> list[Document]:
    """One document per material specification."""
    docs = []
    for m in mats:
        material = m.get("material") or "unknown"
        application = m.get("application") or ""
        props = m.get("properties", {})

        text_parts = [f"[Material] {material}"]
        if application:
            text_parts.append(f"Application: {application}")
        if props:
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items() if v)
            if prop_str:
                text_parts.append(f"Properties: {prop_str}")

        text = "\n".join(text_parts)
        if len(text) < 15:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "data_type": "material",
                "category": "material_spec",
                "material": material,
                "source": m.get("_source", ""),
                "page": m.get("_page", 0),
            },
        ))
    return docs


def _build_document_meta_docs(documents: list[dict]) -> list[Document]:
    """One document per PDF source with its metadata."""
    docs = []
    for doc in documents:
        title = doc.get("title") or doc.get("name", "unknown")
        doc_type = doc.get("document_type") or ""
        revision = doc.get("revision") or ""
        date = doc.get("date") or ""
        sections = doc.get("sections", [])
        standards = doc.get("standards", [])
        topics = doc.get("topics", [])

        text_parts = [f"[Document] {title}"]
        if doc_type:
            text_parts.append(f"Type: {doc_type}")
        if revision:
            text_parts.append(f"Revision: {revision}")
        if date:
            text_parts.append(f"Date: {date}")
        if sections:
            section_list = "; ".join(
                f"{s.get('number', '')}: {s.get('title', '')}" for s in sections[:20]
            )
            text_parts.append(f"Sections: {section_list}")
        if standards:
            text_parts.append(f"Standards: {', '.join(standards[:15])}")
        if topics:
            text_parts.append(f"Topics: {', '.join(topics[:15])}")

        text = "\n".join(text_parts)

        docs.append(Document(
            page_content=text,
            metadata={
                "data_type": "document_metadata",
                "category": "document",
                "title": title,
                "source": doc.get("name", ""),
            },
        ))
    return docs


# ── Batch Embedding ──────────────────────────────────────────────────────

def embed_documents(docs: list[Document], batch_size: int = 32) -> list[list[float]]:
    """Embed all documents using BGE-large-en-v1.5 (1024-dim)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from omnidoc.embedder import get_embedder

    embedder = get_embedder(enable_clip=False)
    texts = [doc.page_content for doc in docs]
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embedder.embed_texts(batch)
        all_vecs.extend(vecs)

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(texts):
            print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return all_vecs


# ── Main ─────────────────────────────────────────────────────────────────

def ingest(rebuild: bool = False):
    """Main ingestion pipeline."""
    print("=" * 60)
    print("  F1 Knowledge Base Ingestion")
    print("=" * 60)

    # Load intelligence
    if not INTELLIGENCE_PATH.exists():
        print(f"\n  ERROR: {INTELLIGENCE_PATH} not found.")
        print("  Run: python pipeline/export_for_ui.py")
        return

    with open(INTELLIGENCE_PATH) as f:
        data = json.load(f)

    print(f"\n  Source: {INTELLIGENCE_PATH.name}")
    print(f"  Rules: {len(data.get('rules', []))}")
    print(f"  Equipment: {len(data.get('equipment', []))}")
    print(f"  Dimensions: {len(data.get('dimensional_data', []))}")
    print(f"  Materials: {len(data.get('material_specs', []))}")
    print(f"  Documents: {len(data.get('documents', []))}")

    # Build documents
    print("\n[1/3] Building documents...")
    all_docs: list[Document] = []

    rule_docs = _build_rule_docs(data.get("rules", []))
    print(f"  Rules: {len(rule_docs)} documents")
    all_docs.extend(rule_docs)

    equip_docs = _build_equipment_docs(data.get("equipment", []))
    print(f"  Equipment: {len(equip_docs)} documents")
    all_docs.extend(equip_docs)

    dim_docs = _build_dimension_docs(data.get("dimensional_data", []))
    print(f"  Dimensions: {len(dim_docs)} documents")
    all_docs.extend(dim_docs)

    mat_docs = _build_material_docs(data.get("material_specs", []))
    print(f"  Materials: {len(mat_docs)} documents")
    all_docs.extend(mat_docs)

    meta_docs = _build_document_meta_docs(data.get("documents", []))
    print(f"  Document metadata: {len(meta_docs)} documents")
    all_docs.extend(meta_docs)

    print(f"\n  Total documents: {len(all_docs)}")

    # Embed
    print("\n[2/3] Embedding with BGE-large-en-v1.5 (1024-dim)...")
    t0 = time.time()
    embeddings = embed_documents(all_docs)
    embed_time = time.time() - t0
    print(f"  Embedded {len(embeddings)} documents in {embed_time:.1f}s")
    print(f"  Rate: {len(embeddings) / embed_time:.1f} docs/sec")

    # Upsert to Atlas
    print("\n[3/3] Upserting to MongoDB Atlas...")
    from pipeline.vectorstore import AtlasVectorStore

    vs = AtlasVectorStore()

    if rebuild:
        print("  Rebuilding collection (--rebuild flag)...")
        vs.delete_collection()

    t0 = time.time()
    count = vs.upsert_documents(all_docs, embeddings)
    upsert_time = time.time() - t0
    print(f"  Upserted {count} documents in {upsert_time:.1f}s")

    # Create vector index
    print("\n  Ensuring vector search index...")
    vs.ensure_vector_index()

    total_in_db = vs.count()
    print(f"\n  Total documents in collection: {total_in_db}")
    print(f"\n  Done! Knowledge base ready for RAG queries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest F1 intelligence into Atlas vector store")
    parser.add_argument("--rebuild", action="store_true", help="Drop collection and rebuild from scratch")
    args = parser.parse_args()

    ingest(rebuild=args.rebuild)
