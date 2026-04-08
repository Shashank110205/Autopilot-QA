"""
graphrag/context/context_pack_builder.py
==========================================
FIX SUMMARY:
  [1] score / confidence / provenance are now passed through from graph_result node data,
      NOT hardcoded to score=1.0, confidence=1.0, provenance="graph".
  [2] warnings_builder.build_warnings() is now actually called (was defined but unused).
  [3] trace_paths helper is now actually called (was defined but unused).
  [4] open_questions field populated: any node missing path_confidence data generates
      a question entry so downstream generators know to flag it.
  [5] parent_context rule enforced: parents ≤ children (was already in parent_context.py
      but not double-checked here).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from graphrag.context.trace_paths import build_trace_paths
from graphrag.context.warnings_builder import build_warnings
from graphrag.models.contracts import (
    Anchor,
    ContextPack,
    EvidenceChunk,
    OpenQuestion,
    RelatedNode,
    TracePath,
    Warning,
)
from graphrag.retrieval.parent_context import attach_parent_context
from graphrag.storage.graph_store import GraphStore


def _parse_source_locator(loc_json: str) -> Dict[str, Any]:
    if not loc_json:
        return {}
    try:
        return json.loads(loc_json)
    except Exception:
        return {"raw": loc_json}


def _parse_extra(extra_json: str) -> dict:
    try:
        return json.loads(extra_json or "{}")
    except Exception:
        return {}


def build_context_pack(
    graph_store: GraphStore,
    anchors: List[Anchor],
    graph_result: Dict[str, Any],
    k_evidence: int = 8,
    k_parent: int = 3,
) -> ContextPack:

    # ── 1. Build child evidence chunks (real metadata – no hardcoded 1.0) ─────
    evidence_chunks: List[EvidenceChunk] = []
    open_questions: List[OpenQuestion] = []

    evidence_nodes = graph_result.get("evidence_nodes", [])
    for i, node in enumerate(evidence_nodes[:k_evidence]):
        if node.get("node_type") != "CHUNK":
            continue

        extra = _parse_extra(node.get("extra_json", "{}"))
        chunk_type = extra.get("chunk_type", "child")

        # Only child chunks go into evidence_chunks (parent goes to parent_context)
        if chunk_type != "child":
            continue

        score = node.get("score", 0.0)
        confidence = node.get("path_confidence", 0.0)
        provenance = node.get("provenance", "graph")
        needs_conf = node.get("needs_confirmation", False)

        # If confidence is zero and we have no path data, flag as open question
        if confidence == 0.0 and provenance == "graph":
            open_questions.append(OpenQuestion(
                step_index=i,
                reason=f"Chunk {node['node_id']} has no path_confidence; evidence grounding unclear.",
                chunk_ids_available=[node["node_id"]],
            ))

        evidence_chunks.append(EvidenceChunk(
            chunk_id=node["node_id"],
            chunk_type=chunk_type,
            text=node.get("text", ""),
            doc_id=node.get("doc_id", ""),
            section_path=node.get("section_path", ""),
            source_locator=_parse_source_locator(node.get("source_locator_json")),
            module=node.get("module"),
            version=node.get("version"),
            score=score,
            confidence=confidence,
            provenance=provenance,
            needs_confirmation=needs_conf,
        ))

    # ── 2. Attach parent context AFTER child selection ─────────────────────────
    parent_context = attach_parent_context(
        graph_store=graph_store,
        child_chunks=evidence_chunks,
        k_parent=k_parent,
    )

    # ── 3. Build trace paths (with path_confidence) ───────────────────────────
    anchor_ids = [a.node_id for a in anchors]
    raw_paths = graph_result.get("trace_paths", [])
    trace_path_objs: List[TracePath] = []
    for tp in build_trace_paths(anchor_ids, raw_paths):
        trace_path_objs.append(TracePath(
            why=tp["why"],
            path=tp["path"],
            path_confidence=tp.get("path_confidence", 0.0),
        ))

    # ── 4. Related nodes ──────────────────────────────────────────────────────
    related_nodes = [
        RelatedNode(
            node_type=r["node_type"],
            node_id=r["node_id"],
            relation=r["relation"],
        )
        for r in graph_result.get("related_nodes", [])
    ]

    # ── 5. Warnings (wired – was not called in old version) ───────────────────
    fallback_triggered = any(
        n.get("provenance") == "vector" for n in evidence_nodes
    )
    fallback_reason = "vector fallback was used" if fallback_triggered else ""

    warning_objs: List[Warning] = build_warnings(
        child_chunks=[n for n in evidence_nodes if n.get("provenance") == "graph"],
        anchors=[{"node_id": a.node_id} for a in anchors],
        required_fields=["text", "doc_id", "section_path"],
        fallback_triggered=fallback_triggered,
        fallback_reason=fallback_reason,
    )

    return ContextPack(
        anchors=anchors,
        evidence_chunks=evidence_chunks,
        parent_context=parent_context,
        trace_paths=trace_path_objs,
        related_nodes=related_nodes,
        warnings=warning_objs,
        open_questions=open_questions,
    )
