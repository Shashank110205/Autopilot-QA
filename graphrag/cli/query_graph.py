"""
graphrag/cli/query_graph.py
=============================
FIX SUMMARY:
  [1] CRITICAL: should_trigger_vector_fallback now checks rel == "SUPPORTED_BY"
      (was "SUPPORTEDBY" – missing underscore). This was causing valid graph paths
      to be treated as missing and always triggering vector fallback.
  [2] acceptance_validation added as a valid --task choice (was silently excluded).
  [3] Non-test_generation tasks now return structured reports instead of
      "unsupported_task" placeholders (debug → trace-to-failure, impact → affected list).
  [4] open_questions surfaced in output; context_pack_to_dict includes it.
  [5] Real score/confidence/provenance passed through from graph_result – no longer
      hardcoded to 1.0 inside context_pack_builder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag.storage.graph_store import GraphStore
from graphrag.retrieval.query_router import route_query
from graphrag.retrieval.anchor_resolver import resolve_anchors
from graphrag.retrieval.graph_retriever import graph_retrieve
from graphrag.retrieval.vector_fallback import vector_search
from graphrag.context.context_pack_builder import build_context_pack
from graphrag.vector.vector_index import build_embeddings
from graphrag.generation.test_generator import generate_tests_from_context_pack


# ── Serialisation helper ──────────────────────────────────────────────────────

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
        "open_questions":  to_serializable(pack.open_questions),   # FIX: was missing
    }


# ── Vector fallback gate ──────────────────────────────────────────────────────

def should_trigger_vector_fallback(
    graph_result: dict,
    min_evidence: int = 3,
    min_avg_conf: float = 0.65,
) -> bool:
    evidence = graph_result.get("evidence_nodes", [])
    if len(evidence) < min_evidence:
        return True

    # Use path_confidence from scored graph_retriever output
    confidences = [n.get("path_confidence", 0.0) for n in evidence if "path_confidence" in n]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        if avg_conf < min_avg_conf:
            return True

    # FIX: "SUPPORTED_BY" not "SUPPORTEDBY"
    has_supported_by = any(
        any(edge.get("rel") == "SUPPORTED_BY" for edge in p.get("path", []))
        for p in graph_result.get("trace_paths", [])
        if isinstance(p, dict)
    )
    return not has_supported_by


# ── Graph + vector merge ──────────────────────────────────────────────────────

def merge_graph_and_vector(graph_store, graph_result, vector_hits, top_k=8):
    existing_ids = {n["node_id"] for n in graph_result.get("evidence_nodes", [])}
    merged = list(graph_result.get("evidence_nodes", []))

    for hit in vector_hits:
        if hit["node_id"] in existing_ids:
            continue
        node = graph_store.get_node(hit["node_id"])
        if not node or node.get("node_type") != "CHUNK":
            continue
        node = dict(node)
        node["_vector_score"] = hit["score"]
        node["score"] = hit["score"]
        node["path_confidence"] = hit["score"]
        node["provenance"] = "vector"
        node["needs_confirmation"] = hit["score"] < 0.65
        merged.append(node)

    merged.sort(
        key=lambda n: (1 if n.get("provenance") == "graph" else 0, n.get("score", 0.0)),
        reverse=True,
    )
    graph_result["evidence_nodes"] = merged[:top_k]
    return graph_result


# ── Task-specific downstream reporters ───────────────────────────────────────

def _debug_report(context_pack_dict: dict) -> dict:
    """Minimal debug report: trace failures back through the graph."""
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
    """Impact report: list affected tests, runs, chunks."""
    related = context_pack_dict.get("related_nodes", [])
    return {
        "task": "impact",
        "status": "ok",
        "affected_nodes": related,
        "evidence_chunks": context_pack_dict.get("evidence_chunks", []),
        "warnings": context_pack_dict.get("warnings", []),
        "open_questions": context_pack_dict.get("open_questions", []),
    }


def _acceptance_report(context_pack_dict: dict) -> dict:
    return {
        "task": "acceptance_validation",
        "status": "ok",
        "evidence_chunks": context_pack_dict.get("evidence_chunks", []),
        "trace_paths": context_pack_dict.get("trace_paths", []),
        "warnings": context_pack_dict.get("warnings", []),
        "open_questions": context_pack_dict.get("open_questions", []),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query GraphRAG graph")
    parser.add_argument("--db", required=True)
    parser.add_argument("--req-id")
    parser.add_argument("--query-text")
    # FIX: acceptance_validation added as valid choice
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

    graph_store = GraphStore(str(Path(args.db)))

    try:
        if args.rebuild_emb:
            build_embeddings(graph_store, force_rebuild=True)

        payload = {
            "task": args.task,
            "req_id": args.req_id,
            "query_text": args.query_text,
            "filters": {
                "module": args.module,
                "version": args.version,
                "doctype": args.doctype,
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

        # ── Route to correct downstream generator / reporter ──────────────────
        if args.task == "test_generation":
            try:
                generated_response = generate_tests_from_context_pack(
                    context_pack=context_pack_dict,
                    provider="ollama",
                )
            except Exception as e:
                generated_response = {"task": args.task, "status": "generation_failed", "error": str(e)}

        elif args.task == "debug":
            generated_response = _debug_report(context_pack_dict)

        elif args.task == "impact":
            generated_response = _impact_report(context_pack_dict)

        elif args.task == "acceptance_validation":
            generated_response = _acceptance_report(context_pack_dict)

        else:
            generated_response = {"task": args.task, "status": "unsupported_task"}

        # ── Output ────────────────────────────────────────────────────────────
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
        print(f"Trace paths:    {len(context_pack.trace_paths)}")
        print(f"Warnings:       {len(context_pack.warnings)}")
        print(f"Open questions: {len(context_pack.open_questions)}")

        if args.task == "test_generation":
            print("\n── Generated Tests ──")
            print(json.dumps(to_serializable(generated_response), indent=2, default=str))
        else:
            print(f"\n── {args.task} Report ──")
            print(json.dumps(to_serializable(generated_response), indent=2, default=str))

    finally:
        graph_store.close()


if __name__ == "__main__":
    main()
