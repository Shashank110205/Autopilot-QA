import duckdb
from typing import Any, Dict, List, Optional


class GraphStore:
    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path)

    def close(self):
        self.conn.close()

    def query(self, sql: str, params=None) -> List[Dict[str, Any]]:
        params = params or []
        rows = self.conn.execute(sql, params).fetchall()
        cols = [d[0] for d in self.conn.description]
        return [dict(zip(cols, row)) for row in rows]

    def execute(self, sql: str, params=None):
        params = params or []
        self.conn.execute(sql, params)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        rows = self.query("SELECT * FROM nodes WHERE node_id = ?", [node_id])
        return rows[0] if rows else None

    def get_edges_from(self, node_id: str) -> List[Dict[str, Any]]:
        return self.query("SELECT * FROM edges WHERE src_id = ?", [node_id])

    def get_edges_to(self, node_id: str) -> List[Dict[str, Any]]:
        return self.query("SELECT * FROM edges WHERE dst_id = ?", [node_id])

    def ensure_embeddings_table(self):
        self.execute("""
        CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            module TEXT,
            version TEXT,
            doctype TEXT,
            section_path TEXT,
            embedding BLOB
        )
        """)

    def upsert_embedding(
        self,
        node_id: str,
        node_type: str,
        module: str,
        version: str,
        doctype: str,
        section_path: str,
        embedding_bytes: bytes
    ):
        self.execute("""
        INSERT OR REPLACE INTO node_embeddings
        (node_id, node_type, module, version, doctype, section_path, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [node_id, node_type, module, version, doctype, section_path, embedding_bytes])

    def get_embedding_rows(self, filters: Dict[str, str] = None, node_types: List[str] = None) -> List[Dict[str, Any]]:
        filters = filters or {}
        node_types = node_types or []

        clauses = []
        params = []

        if node_types:
            placeholders = ",".join(["?"] * len(node_types))
            clauses.append(f"node_type IN ({placeholders})")
            params.extend(node_types)

        for key in ["module", "version", "doctype"]:
            if filters.get(key):
                clauses.append(f"{key} = ?")
                params.append(filters[key])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        return self.query(f"""
        SELECT * FROM node_embeddings
        {where}
        """, params)