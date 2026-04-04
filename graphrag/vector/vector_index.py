import numpy as np
from sentence_transformers import SentenceTransformer
from graphrag.storage.graph_store import GraphStore

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def build_embeddings(graph_store: GraphStore, force_rebuild: bool = False):
    graph_store.execute("""
    DROP TABLE IF EXISTS node_embeddings;
    """)
    graph_store.ensure_embeddings_table()
    model = SentenceTransformer(MODEL_NAME)

    nodes = graph_store.query("""
    SELECT
        node_id,
        node_type,
        module,
        version,
        doc_id,
        section_path,
        COALESCE(text, '') AS text
    FROM nodes
    WHERE node_type IN ('REQ', 'CHUNK', 'DEFECT', 'FAILURE')
      AND text IS NOT NULL
      AND LENGTH(TRIM(text)) > 0
    """)

    if not force_rebuild:
        existing = {
            r["node_id"] for r in graph_store.query("SELECT node_id FROM node_embeddings")
        }
        nodes = [n for n in nodes if n["node_id"] not in existing]

    texts = [n["text"] for n in nodes]
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