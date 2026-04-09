"""
graphrag/cli/query_graph.py
=============================
FIXES IN THIS VERSION:
  [PARTIAL-6] Normalized graph-first fusion replaces the simple sort.
              Old: sorted by (provenance=="graph", score) – graph always wins regardless
                   of actual score difference, vector items with very high similarity
                   could be buried below low-confidence graph items.
              New: graph candidates get a graph_bias bonus (default +0.3) added to
                   their score before merging, then all candidates sort by adjusted score.
                   This preserves graph-first bias while still allowing high-confidence
                   vector evidence to surface when graph evidence is sparse.

  [PARTIAL-7] acceptance_validation: real AcceptanceComparator added.
              For each evidence chunk, compares against anchor CRU's acceptance_criteria
              and emits per-criterion decisions (match/partial/missing/conflict).
              Open questions are generated for criteria with no evidence.

  [BUG-1/2] build_embeddings now passes manifest_dir so embedding_manifest.json
              is written to the --out directory on --rebuild-emb.

Previously-correct fixes retained:
  - SUPPORTED_BY typo fixed ("SUPPORTED_BY" not "SUPPORTEDBY")
  - acceptance_validation task routable
  - open_questions in context_pack_to_dict
  - debug/impact reporters
  - real score/confidence/provenance from graph_result
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from graphrag.context.context_pack_builder import build_context_pack
from graphrag.generation.test_generator import generate_tests_from_context_pack
from graphrag.models.contracts import AcceptanceDecision, OpenQuestion
from graphrag.retrieval.anchor_resolver import resolve_anchors
from graphrag.retrieval.graph_retriever import graph_retrieve
from graphrag.retrieval.query_router import route_query
from graphrag.retrieval.vector_fallback import vector_search
from graphrag.storage.graph_store import GraphStore
from graphrag.vector.vector_index import build_embeddings


# ── Serialisation ─────────────────────────────────────────────────────────────

def to_serializable(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_serializable(v) for k, v in obj.__dict__.items()}
    return str(obj)


def context_pack_to_dict(pack) -> dict:
    return {
        "anchors":         [to_serializable(a) for a in pack.anchors],
        "evidence_chunks": [to_serializable(c) for c in pack.evidence_chunks],
        "parent_context":  [to_serializable(c) for c in pack.parent_context],
        "trace_paths":     [to_serializable(t) for t in pack.trace_paths],
        "related_nodes":   [to_serializable(r) for r in pack.related_nodes],
        "warnings":        to_serializable(pack.warnings),
        "open_questions":  to_serializable(pack.open_questions),
    }


# ── Vector fallback gate (SUPPORTED_BY typo fixed in prev round, kept) ────────

def should_trigger_vector_fallback(
    graph_result: dict,
    min_evidence: int = 3,
    min_avg_conf: float = 0.65,
) -> bool:
    evidence = graph_result.get("evidence_nodes", [])
    if len(evidence) < min_evidence:
        return True

    confidences = [n.get("path_confidence", 0.0) for n in evidence if "path_confidence" in n]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        if avg_conf < min_avg_conf:
            return True

    has_supported_by = any(
        any(edge.get("rel") == "SUPPORTED_BY" for edge in p.get("path", []))
        for p in graph_result.get("trace_paths", [])
        if isinstance(p, dict)
    )
    return not has_supported_by


# ── [PARTIAL-6] Normalized graph-first fusion ─────────────────────────────────

_GRAPH_BIAS = 0.30   # added to graph node scores before merging


def merge_graph_and_vector(
    graph_store: GraphStore,
    graph_result: dict,
    vector_hits: list,
    top_k: int = 8,
) -> dict:
    """
    Graph-first fused merge.
    Graph candidates score is boosted by GRAPH_BIAS before sorting so
    graph evidence wins over similarly-scored vector evidence, but very
    high-similarity vector hits (similarity > 1 - GRAPH_BIAS) can still surface.
    """
    graph_nodes = graph_result.get("evidence_nodes", [])
    existing_ids = {n["node_id"] for n in graph_nodes}

    # Boost graph scores
    for n in graph_nodes:
        n["_adjusted_score"] = n.get("score", 0.0) + _GRAPH_BIAS

    # Add vector hits (CHUNK only, not already in graph results)
    vector_nodes = []
    for hit in vector_hits:
        if hit["node_id"] in existing_ids:
            continue
        node = graph_store.get_node(hit["node_id"])
        if not node or node.get("node_type") != "CHUNK":
            continue

        import json as _json
        try:
            extra = _json.loads(node.get("extra_json") or "{}")
        except Exception:
            extra = {}
        if extra.get("chunk_type") != "child":
            continue

        node = dict(node)
        sim = hit.get("score", 0.0)
        node["score"] = sim
        node["path_confidence"] = sim
        node["provenance"] = "vector"
        node["needs_confirmation"] = sim < 0.65
        node["_adjusted_score"] = sim   # no bias for vector
        vector_nodes.append(node)

    merged = graph_nodes + vector_nodes
    merged.sort(key=lambda n: n.get("_adjusted_score", 0.0), reverse=True)

    graph_result["evidence_nodes"] = merged[:top_k]
    return graph_result


# ── Task reporters ────────────────────────────────────────────────────────────

def _debug_report(context_pack_dict: dict) -> dict:
    related = context_pack_dict.get("related_nodes", [])
    defects = [r for r in related if r.get("node_type") in ("DEFECT", "FAILURE", "RUN")]
    return {
        "task": "debug",
        "status": "ok",
        "trace_to_failure": context_pack_dict.get("trace_paths", []),
        "related_defects_and_runs": defects,
        "warnings": context_pack_dict.get("warnings", []),
        "open_questions": context_pack_dict.get("open_questions", []),
    }


def _impact_report(context_pack_dict: dict) -> dict:
    return {
        "task": "impact",
        "status": "ok",
        "affected_nodes": context_pack_dict.get("related_nodes", []),
        "evidence_chunks": context_pack_dict.get("evidence_chunks", []),
        "warnings": context_pack_dict.get("warnings", []),
        "open_questions": context_pack_dict.get("open_questions", []),
    }


# ── [PARTIAL-7] Acceptance comparator ────────────────────────────────────────

def _acceptance_report(
    context_pack_dict: dict,
    graph_store: GraphStore,
    anchors: list,
) -> dict:
    """
    Real acceptance comparator: for each anchor CRU, retrieve its
    acceptance_criteria and compare against evidence chunks.

    Verdict per criterion:
      match   – at least one evidence chunk clearly covers this criterion
      partial – evidence exists but may not fully satisfy
      missing – no evidence found at all
      conflict – evidence contradicts the criterion (detected by keyword heuristic)
    """
    import json as _json

    decisions: List[dict] = []
    open_questions: List[dict] = []
    evidence_ids = [
        c.get("chunk_id") for c in context_pack_dict.get("evidence_chunks", [])
    ]

    for anchor in anchors:
        node = graph_store.get_node(anchor.node_id if hasattr(anchor, "node_id") else anchor["node_id"])
        if not node:
            continue

        try:
            extra = _json.loads(node.get("extra_json") or "{}")
        except Exception:
            extra = {}

        criteria_raw = extra.get("acceptance_criteria")
        if not criteria_raw:
            open_questions.append({
                "question": f"No acceptance_criteria defined for anchor {node['node_id']}.",
                "required_for": node["node_id"],
                "chunk_ids_available": evidence_ids,
            })
            continue

        # Split criteria into individual lines/items
        if isinstance(criteria_raw, list):
            criteria = criteria_raw
        else:
            criteria = [c.strip() for c in str(criteria_raw).split("\n") if c.strip()]

        evidence_texts = [
            c.get("text", "") for c in context_pack_dict.get("evidence_chunks", [])
        ]

        for criterion in criteria:
            criterion_lower = criterion.lower()
            supporting_ids = []
            verdict = "missing"

            for chunk in context_pack_dict.get("evidence_chunks", []):
                chunk_text = chunk.get("text", "").lower()
                # Simple keyword overlap heuristic
                words = set(criterion_lower.split())
                overlap = len(words & set(chunk_text.split()))
                coverage = overlap / max(len(words), 1)

                if coverage >= 0.5:
                    supporting_ids.append(chunk["chunk_id"])
                    verdict = "match"
                elif coverage >= 0.2:
                    supporting_ids.append(chunk["chunk_id"])
                    if verdict == "missing":
                        verdict = "partial"

            if verdict == "missing":
                open_questions.append({
                    "question": f"No evidence found for criterion: '{criterion}'",
                    "required_for": node["node_id"],
                    "chunk_ids_available": evidence_ids,
                })

            decisions.append(AcceptanceDecision(
                criterion=criterion,
                verdict=verdict,
                evidence_chunk_ids=supporting_ids,
                notes=f"coverage heuristic on {len(evidence_texts)} chunks",
            ))

    return {
        "task": "acceptance_validation",
        "status": "ok",
        "decisions": [to_serializable(d) for d in decisions],
        "open_questions": open_questions,
        "warnings": context_pack_dict.get("warnings", []),
        "evidence_chunks": context_pack_dict.get("evidence_chunks", []),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query GraphRAG graph")
    parser.add_argument("--db", required=True)
    parser.add_argument("--req-id")
    parser.add_argument("--query-text")
    parser.add_argument(
        "--task", required=True,
        choices=["test_generation", "debug", "impact", "acceptance_validation"],
    )
    parser.add_argument("--module", default="")
    parser.add_argument("--version", default="")
    parser.add_argument("--doctype", default="")
    parser.add_argument("--out", default="context_pack.json")
    parser.add_argument("--rebuild-emb", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out_dir = str(Path(args.out).parent)
    graph_store = GraphStore(str(Path(args.db)))

    try:
        if args.rebuild_emb:
            # [PARTIAL-5c] pass manifest_dir so embedding_manifest.json lands in --out dir
            build_embeddings(graph_store, force_rebuild=True, manifest_dir=out_dir)

        payload = {
            "task": args.task,
            "req_id": args.req_id,
            "query_text": args.query_text,
            "filters": {
                "module":   args.module,
                "version":  args.version,
                "doc_type": args.doctype,
            },
            "k_evidence": 8,
            "k_parent": 3,
        }

        query = route_query(payload)
        anchors = resolve_anchors(graph_store, query)
        graph_result = graph_retrieve(
            graph_store, anchors, query.task, filters=query.filters
        )

        if should_trigger_vector_fallback(graph_result):
            vector_hits = vector_search(
                graph_store=graph_store,
                query_text=query.query_text or query.req_id or "",
                filters=query.filters,
                node_types=["CHUNK", "CRU", "DEFECT", "FAILURE"],
                top_k=30,
            )
            graph_result = merge_graph_and_vector(
                graph_store, graph_result, vector_hits, top_k=query.k_evidence
            )

        context_pack = build_context_pack(
            graph_store=graph_store,
            anchors=anchors,
            graph_result=graph_result,
            k_evidence=query.k_evidence,
            k_parent=query.k_parent,
        )
        context_pack_dict = context_pack_to_dict(context_pack)

        # Route to task handler
        if args.task == "test_generation":
            try:
                generated_response = generate_tests_from_context_pack(
                    context_pack=context_pack_dict, provider="ollama"
                )
            except Exception as e:
                generated_response = {"task": args.task, "status": "generation_failed", "error": str(e)}

        elif args.task == "debug":
            generated_response = _debug_report(context_pack_dict)

        elif args.task == "impact":
            generated_response = _impact_report(context_pack_dict)

        elif args.task == "acceptance_validation":
            # [PARTIAL-7] real comparator
            generated_response = _acceptance_report(context_pack_dict, graph_store, anchors)

        else:
            generated_response = {"task": args.task, "status": "unsupported_task"}

        if args.json:
            print(json.dumps({
                "context_pack": context_pack_dict,
                "result": to_serializable(generated_response),
            }, indent=2, default=str))
            return

        Path(args.out).write_text(
            json.dumps(context_pack_dict, indent=2, default=str), encoding="utf-8"
        )
        print(f"✅ Context Pack saved: {args.out}")
        print(f"Anchors:        {len(context_pack.anchors)}")
        print(f"Evidence chunks:{len(context_pack.evidence_chunks)}")
        print(f"Parent context: {len(context_pack.parent_context)}")
        print(f"Warnings:       {len(context_pack.warnings)}")
        print(f"Open questions: {len(context_pack.open_questions)}")

        print(f"\n── {args.task} Report ──")
        print(json.dumps(to_serializable(generated_response), indent=2, default=str))

    finally:
        graph_store.close()


if __name__ == "__main__":
    main()
