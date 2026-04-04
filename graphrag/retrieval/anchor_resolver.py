from typing import List
from graphrag.storage.graph_store import GraphStore
from graphrag.models.contracts import Anchor, QueryInput
from graphrag.retrieval.vector_fallback import vector_search


def resolve_anchors(graph_store: GraphStore, query: QueryInput) -> List[Anchor]:
    anchors = []

    if query.req_id:
        node = graph_store.get_node(query.req_id)
        if node:
            anchors.append(Anchor(
                node_id=node["node_id"],
                node_type=node["node_type"],
                score=1.0,
                provenance="graph"
            ))
            return anchors
        raise ValueError(f"Anchor node not found: {query.req_id}")

    if query.query_text:
        candidates = vector_search(
            graph_store=graph_store,
            query_text=query.query_text,
            filters=query.filters,
            node_types=["REQ", "CHUNK"],
            top_k=5
        )

        req_first = sorted(candidates, key=lambda x: 0 if x["node_type"] == "REQ" else 1)

        for c in req_first:
            anchors.append(Anchor(
                node_id=c["node_id"],
                node_type=c["node_type"],
                score=c["score"],
                provenance="vector"
            ))

    if not anchors:
        raise ValueError("No anchors resolved from req_id or query_text")

    return anchors[:5]