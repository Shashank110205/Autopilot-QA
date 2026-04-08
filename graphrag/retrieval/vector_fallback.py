import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from graphrag.storage.graph_store import GraphStore

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def vector_search(
    graph_store: GraphStore,
    query_text: str,
    filters: Dict[str, str],
    node_types: List[str],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    model = _get_model()
    q = model.encode([query_text], normalize_embeddings=True)[0]

    rows = graph_store.get_embedding_rows(filters=filters, node_types=node_types)

    scored = []
    for row in rows:
        emb = _from_bytes(row["embedding_bytes"])
        score = _cosine(q, emb)
        scored.append({
            "node_id": row["node_id"],
            "node_type": row["node_type"],
            "score": score,
            "provenance": "vector"
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]