"""
graphrag/vector/vector_index.py
=================================
FIX SUMMARY:
  [1] DROP TABLE now only runs when force_rebuild=True.
      Previously it ran unconditionally, so the incremental filter below it
      always saw an empty table and re-embedded every node on every run.
  [2] Node type filter changed from 'REQ' → 'CRU'.
      REQ nodes do not exist in the active schema; CRU is the canonical
      requirement anchor type built by cru_builder.py.
  [3] For CRU nodes, acceptance_criteria text (stored in extra_json) is
      appended to the embedding text so semantic search covers both the
      requirement statement and its pass/fail criteria.
  [4] Removed ensure_embeddings_table() call – GraphStore.__init__ already
      runs _init_schema() which issues CREATE TABLE IF NOT EXISTS for all
      tables. After a force-rebuild DROP, re-init via graph_store._init_schema().
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from graphrag.storage.graph_store import GraphStore

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _enrich_text(node: dict) -> str:
    """
    For CRU nodes: append acceptance_criteria from extra_json if present,
    so the embedding covers both the statement and its testable criteria.
    For all other types: use text as-is.
    """
    base = node.get("text") or ""
    if node.get("node_type") == "CRU":
        try:
            extra = json.loads(node.get("extra_json") or "{}")
            ac = extra.get("acceptance_criteria")
            if ac and isinstance(ac, str) and ac.strip():
                base = f"{base} {ac}".strip()
        except Exception:
            pass
    return base


def build_embeddings(graph_store: GraphStore, force_rebuild: bool = False):
    # FIX [1]: only destroy existing embeddings when explicitly requested
    if force_rebuild:
        graph_store.execute("DROP TABLE IF EXISTS node_embeddings;")
        # Re-create the table — _init_schema() is idempotent (CREATE IF NOT EXISTS)
        graph_store._init_schema()
    model = SentenceTransformer(MODEL_NAME)

    # FIX [2]: CRU replaces REQ; extra_json fetched for CRU text enrichment
    nodes = graph_store.query("""
    SELECT
        node_id,
        node_type,
        module,
        version,
        doc_id,
        section_path,
        COALESCE(text, '') AS text,
        COALESCE(extra_json, '{}') AS extra_json
    FROM nodes
    WHERE node_type IN ('CRU', 'CHUNK', 'DEFECT', 'FAILURE')
      AND text IS NOT NULL
      AND LENGTH(TRIM(text)) > 0
    """)

    # Incremental: skip nodes already embedded (only meaningful when not force_rebuild)
    if not force_rebuild:
        existing = {
            r["node_id"] for r in graph_store.query("SELECT node_id FROM node_embeddings")
        }
        nodes = [n for n in nodes if n["node_id"] not in existing]

    # FIX [3]: richer text for CRU nodes
    texts = [_enrich_text(n) for n in nodes]

    if not texts:
        print("No new nodes to embed.")
        return

    vectors = model.encode(texts, normalize_embeddings=True)

    for node, vec in zip(nodes, vectors):
        graph_store.upsert_embedding(
            node_id=node["node_id"],
            node_type=node["node_type"],
            module=node.get("module"),
            version=node.get("version"),
            doctype=node.get("doc_id"),
            section_path=node.get("section_path"),
            embedding_bytes=_to_bytes(vec),
        )

    print(f"✅ Embedded {len(nodes)} nodes")