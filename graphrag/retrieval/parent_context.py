import json
from typing import List, Dict, Any
from graphrag.models.contracts import EvidenceChunk
from graphrag.storage.graph_store import GraphStore


def _parse_source_locator(loc_json: str) -> Dict[str, Any]:
    if not loc_json:
        return {}
    try:
        return json.loads(loc_json)
    except Exception:
        return {"raw": loc_json}


def attach_parent_context(
    graph_store: GraphStore,
    child_chunks: List[EvidenceChunk],
    k_parent: int = 3
) -> List[EvidenceChunk]:
    """
    Attach parent CHUNK nodes for the selected child evidence chunks.

    Rule:
    - Traverse reverse PARENT_OF edges to find parent chunks.
    - Keep unique parents only.
    - Parent chunks must not outnumber child chunks.
    """
    parent_chunks: List[EvidenceChunk] = []
    seen_parent_ids = set()

    for child in child_chunks:
        parent_edges = graph_store.get_edges_to(child.chunk_id)

        ranked_parent_edges = [
            e for e in parent_edges
            if e.get("rel_type") == "PARENT_OF"
        ]
        ranked_parent_edges.sort(key=lambda e: e.get("confidence", 0.0), reverse=True)

        for edge in ranked_parent_edges[:k_parent]:
            parent_id = edge["src_id"]
            if parent_id in seen_parent_ids:
                continue

            parent_node = graph_store.get_node(parent_id)
            if not parent_node:
                continue
            if parent_node.get("node_type") != "CHUNK":
                continue

            extra_json = parent_node.get("extra_json") or "{}"
            try:
                extra = json.loads(extra_json)
            except Exception:
                extra = {}

            parent_chunks.append(
                EvidenceChunk(
                    chunk_id=parent_node["node_id"],
                    chunk_type=extra.get("chunk_type", "parent"),
                    text=parent_node.get("text", ""),
                    doc_id=parent_node.get("doc_id", ""),
                    section_path=parent_node.get("section_path", ""),
                    source_locator=_parse_source_locator(parent_node.get("source_locator_json")),
                    score=0.8,
                    confidence=edge.get("confidence", 0.8),
                    provenance="graph"
                )
            )
            seen_parent_ids.add(parent_id)

    max_parents = len(child_chunks)
    return parent_chunks[:max_parents]