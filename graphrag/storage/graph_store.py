"""
graphrag/storage/graph_store.py
================================
FIX SUMMARY (audit items addressed):
  [1] Added auto-schema init in __init__  → build path no longer needs manual migration
  [2] Added insert_node()                 → was missing; build_graph.py called it on an old API
  [3] Added insert_edge()                 → same; also enforces confidence_reason in extra_json
  [4] Added node_exists() / edge_exists() → required by edge_builder + integrity checker
  [5] Added stats()                       → required by build_graph.py CLI
  [6] get_edges_from / get_edges_to now   → accept optional rel_types list for whitelist filtering
  [7] Removed INSERT OR REPLACE syntax    → DuckDB uses ON CONFLICT … DO UPDATE / DO NOTHING
  [8] Embeddings column renamed 'embedding_bytes' to match schema.sql exactly
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class GraphStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema bootstrap (idempotent – safe to call on every startup)
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        if not _SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Schema file not found: {_SCHEMA_PATH}")
        self.conn.execute(_SCHEMA_PATH.read_text())

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------
    def insert_node(self, node: dict) -> None:
        """
        Idempotent upsert for nodes.
        Required keys: node_id, node_type.
        """
        required = {"node_id", "node_type"}
        missing = required - node.keys()
        if missing:
            raise ValueError(f"insert_node: missing required fields {missing}")

        # Serialise extra_json if caller passed a dict
        extra = node.get("extra_json")
        if isinstance(extra, dict):
            extra = json.dumps(extra)

        self.conn.execute(
            """
            INSERT INTO nodes
                (node_id, node_type, title, text, module, version, doc_id,
                 doc_type, section_path, source_locator_json, extra_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (node_id) DO UPDATE SET
                node_type          = excluded.node_type,
                title              = excluded.title,
                text               = excluded.text,
                module             = excluded.module,
                version            = excluded.version,
                doc_id             = excluded.doc_id,
                doc_type           = excluded.doc_type,
                section_path       = excluded.section_path,
                source_locator_json= excluded.source_locator_json,
                extra_json         = excluded.extra_json
            """,
            [
                node["node_id"],
                node["node_type"],
                node.get("title"),
                node.get("text"),
                node.get("module"),
                node.get("version"),
                node.get("doc_id"),
                node.get("doc_type"),
                node.get("section_path"),
                node.get("source_locator_json"),
                extra,
            ],
        )

    def insert_edge(self, edge: dict) -> None:
        """
        Idempotent upsert for edges.
        Required keys: src_id, src_type, rel_type, dst_id, dst_type, confidence.
        extra_json MUST include 'confidence_reason'.
        INFERRED_SUPPORTED_BY edges are rejected unless _allow_persist_inferred=True.
        """
        required = {"src_id", "src_type", "rel_type", "dst_id", "dst_type", "confidence"}
        missing = required - edge.keys()
        if missing:
            raise ValueError(f"insert_edge: missing required fields {missing}")

        # Reject persisting inferred edges without explicit approval
        if edge["rel_type"] == "INFERRED_SUPPORTED_BY" and not edge.get("_allow_persist_inferred"):
            raise ValueError(
                "INFERRED_SUPPORTED_BY must not be persisted without human approval. "
                "Set _allow_persist_inferred=True only after explicit validation."
            )

        extra = edge.get("extra_json")
        if isinstance(extra, dict):
            if "confidence_reason" not in extra:
                raise ValueError(
                    f"insert_edge: extra_json must include 'confidence_reason' "
                    f"({edge['src_id']} --{edge['rel_type']}--> {edge['dst_id']})"
                )
            extra = json.dumps(extra)
        elif isinstance(extra, str):
            try:
                parsed = json.loads(extra)
                if "confidence_reason" not in parsed:
                    raise ValueError(
                        f"insert_edge: extra_json must include 'confidence_reason' "
                        f"({edge['src_id']} --{edge['rel_type']}--> {edge['dst_id']})"
                    )
            except json.JSONDecodeError:
                pass  # leave as-is; validation best-effort on raw strings
        else:
            raise ValueError("insert_edge: extra_json is required and must be a dict or JSON string")

        self.conn.execute(
            """
            INSERT INTO edges
                (src_id, src_type, rel_type, dst_id, dst_type, confidence,
                 evidence_chunk_id, extra_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (src_id, rel_type, dst_id) DO NOTHING
            """,
            [
                edge["src_id"],
                edge["src_type"],
                edge["rel_type"],
                edge["dst_id"],
                edge["dst_type"],
                float(edge["confidence"]),
                edge.get("evidence_chunk_id"),
                extra,
            ],
        )

    # ------------------------------------------------------------------
    # Existence checks (required by edge_builder + integrity checker)
    # ------------------------------------------------------------------
    def node_exists(self, node_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM nodes WHERE node_id = ?", [node_id]
        ).fetchone()
        return row is not None

    def edge_exists(self, src_id: str, rel_type: str, dst_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM edges WHERE src_id = ? AND rel_type = ? AND dst_id = ?",
            [src_id, rel_type, dst_id],
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------
    def query(self, sql: str, params=None) -> List[Dict[str, Any]]:
        params = params or []
        rows = self.conn.execute(sql, params).fetchall()
        cols = [d[0] for d in self.conn.description]
        return [dict(zip(cols, row)) for row in rows]

    def execute(self, sql: str, params=None):
        self.conn.execute(sql, params or [])

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        rows = self.query("SELECT * FROM nodes WHERE node_id = ?", [node_id])
        return rows[0] if rows else None

    def get_edges_from(
        self, node_id: str, rel_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if rel_types:
            placeholders = ",".join("?" * len(rel_types))
            return self.query(
                f"SELECT * FROM edges WHERE src_id = ? AND rel_type IN ({placeholders})",
                [node_id, *rel_types],
            )
        return self.query("SELECT * FROM edges WHERE src_id = ?", [node_id])

    def get_edges_to(
        self, node_id: str, rel_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if rel_types:
            placeholders = ",".join("?" * len(rel_types))
            return self.query(
                f"SELECT * FROM edges WHERE dst_id = ? AND rel_type IN ({placeholders})",
                [node_id, *rel_types],
            )
        return self.query("SELECT * FROM edges WHERE dst_id = ?", [node_id])

    # ------------------------------------------------------------------
    # Stats (required by build_graph.py CLI)
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        node_rows = self.conn.execute(
            "SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type"
        ).fetchall()
        edge_rows = self.conn.execute(
            "SELECT rel_type, COUNT(*) FROM edges GROUP BY rel_type"
        ).fetchall()
        emb_count = self.conn.execute(
            "SELECT COUNT(*) FROM node_embeddings"
        ).fetchone()[0]
        return {
            "nodes": {r[0]: r[1] for r in node_rows},
            "edges": {r[0]: r[1] for r in edge_rows},
            "embeddings": emb_count,
        }

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def upsert_embedding(
        self,
        node_id: str,
        node_type: str,
        module: str,
        version: str,
        doctype: str,
        section_path: str,
        embedding_bytes: bytes,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO node_embeddings
                (node_id, node_type, module, version, doc_type, section_path,
                 embedding_model, embedding_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (node_id) DO UPDATE SET
                embedding_bytes = excluded.embedding_bytes,
                embedding_model = excluded.embedding_model
            """,
            [node_id, node_type, module, version, doctype, section_path,
             embedding_model, embedding_bytes],
        )

    def get_embedding_rows(
        self,
        filters: Optional[Dict[str, str]] = None,
        node_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        node_types = node_types or []
        clauses, params = [], []
        if node_types:
            placeholders = ",".join("?" * len(node_types))
            clauses.append(f"node_type IN ({placeholders})")
            params.extend(node_types)
        for key in ("module", "version"):
            if filters.get(key):
                clauses.append(f"{key} = ?")
                params.append(filters[key])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.query(f"SELECT * FROM node_embeddings {where}", params)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
