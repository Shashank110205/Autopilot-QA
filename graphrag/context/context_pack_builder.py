import json
from typing import List, Dict, Any
from graphrag.models.contracts import (
    ContextPack,
    Anchor,
    EvidenceChunk,
    TracePath,
    RelatedNode,
)
from graphrag.storage.graph_store import GraphStore
from graphrag.retrieval.parent_context import attach_parent_context


def _parse_source_locator(loc_json: str) -> Dict[str, Any]:
    """Parse source_locator_json to dict."""
    if not loc_json:
        return {}
    try:
        return json.loads(loc_json)
    except Exception:
        return {"raw": loc_json}


def build_context_pack(
    graph_store: GraphStore,
    anchors: List[Anchor],
    graph_result: Dict[str, Any],
    k_evidence: int = 8,
    k_parent: int = 3,
) -> ContextPack:
    """Build final Context Pack from graph retrieval output."""

    evidence_chunks: List[EvidenceChunk] = []

    for node in graph_result.get("evidence_nodes", []):
        if node.get("node_type") != "CHUNK":
            continue

        extra_json = node.get("extra_json") or "{}"
        try:
            extra = json.loads(extra_json)
        except Exception:
            extra = {}

        evidence_chunks.append(
            EvidenceChunk(
                chunk_id=node["node_id"],
                chunk_type=extra.get("chunk_type", "child"),
                text=node.get("text", ""),
                doc_id=node.get("doc_id", ""),
                section_path=node.get("section_path", ""),
                source_locator=_parse_source_locator(node.get("source_locator_json")),
                score=1.0,
                confidence=1.0,
                provenance="graph",
            )
        )

    evidence_chunks = evidence_chunks[:k_evidence]

    parent_context = attach_parent_context(
        graph_store=graph_store,
        child_chunks=evidence_chunks,
        k_parent=k_parent,
    )

    warnings = []
    if len(evidence_chunks) < 3:
        warnings.append({
            "type": "LOW_EVIDENCE_COUNT",
            "message": f"Only {len(evidence_chunks)} chunks found. Expected more."
        })

    if not parent_context and evidence_chunks:
        warnings.append({
            "type": "MISSING_PARENT_CONTEXT",
            "message": "No parent context could be attached for the selected child chunks."
        })

    trace_paths = []
    for item in graph_result.get("trace_paths", []):
        if isinstance(item, dict):
            why = item.get("why", "Graph traversal")
            path = item.get("path", [])
        else:
            why = "Graph traversal"
            path = item

        trace_paths.append(TracePath(why=why, path=path))

    related_nodes = []
    for r in graph_result.get("related_nodes", []):
        related_nodes.append(
            RelatedNode(
                node_type=r["node_type"],
                node_id=r["node_id"],
                relation=r["relation"],
            )
        )

    return ContextPack(
        anchors=anchors,
        evidence_chunks=evidence_chunks,
        parent_context=parent_context,
        trace_paths=trace_paths,
        related_nodes=related_nodes,
        warnings=warnings,
    )