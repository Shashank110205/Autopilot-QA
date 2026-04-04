from typing import List
from graphrag.storage.graph_store import GraphStore
from graphrag.models.contracts import EvidenceChunk


def vector_fallback(
    graph_store: GraphStore,
    query_text: str,
    k: int = 10
) -> List[EvidenceChunk]:
    """Vector search when graph fails."""
    # TODO: Implement FAISS/Chroma over nodeembeddings table
    # For now: keyword fallback
    candidates = graph_store.query("""
        SELECT n.*, similarity(query_embedding, n.embedding) as score
        FROM nodes n 
        JOIN nodeembeddings e ON n.node_id = e.node_id
        WHERE n.node_type IN ('CHUNK', 'REQ')
        ORDER BY score DESC LIMIT ?
    """, params=[k])
    
    return []  # Placeholder