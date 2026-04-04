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


def to_serializable(obj):
    """Recursively convert objects to JSON-serializable Python types."""
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


def context_pack_to_dict(pack):
    """Convert ContextPack object to serializable dict."""
    return {
        "anchors": [to_serializable(a) for a in pack.anchors],
        "evidence_chunks": [to_serializable(c) for c in pack.evidence_chunks],
        "parent_context": [to_serializable(c) for c in pack.parent_context],
        "trace_paths": [to_serializable(t) for t in pack.trace_paths],
        "related_nodes": [to_serializable(r) for r in pack.related_nodes],
        "warnings": to_serializable(pack.warnings),
    }


def should_trigger_vector_fallback(graph_result: dict, min_evidence: int = 3, min_avg_conf: float = 0.65) -> bool:
    evidence = graph_result.get("evidence_nodes", [])
    if len(evidence) < min_evidence:
        return True

    confidences = []
    for p in graph_result.get("trace_paths", []):
        path = p.get("path", []) if isinstance(p, dict) else []
        for edge in path:
            if isinstance(edge, dict) and "conf" in edge:
                confidences.append(edge["conf"])

    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        if avg_conf < min_avg_conf:
            return True

    has_supported_by = any(
        any(edge.get("rel") == "SUPPORTEDBY" for edge in p.get("path", []))
        for p in graph_result.get("trace_paths", [])
        if isinstance(p, dict)
    )
    return not has_supported_by


def merge_graph_and_vector(graph_store, graph_result, vector_hits, top_k=8):
    existing_ids = {n["node_id"] for n in graph_result.get("evidence_nodes", [])}
    merged = list(graph_result.get("evidence_nodes", []))

    for hit in vector_hits:
        if hit["node_id"] in existing_ids:
            continue
        node = graph_store.get_node(hit["node_id"])
        if not node or node["node_type"] != "CHUNK":
            continue
        node = dict(node)
        node["_vector_score"] = hit["score"]
        node["_provenance"] = "vector"
        merged.append(node)

    merged.sort(
        key=lambda n: (
            1 if n.get("_provenance") == "graph" else 0,
            n.get("_vector_score", 0.0)
        ),
        reverse=True,
    )
    graph_result["evidence_nodes"] = merged[:top_k]
    return graph_result


def main():
    parser = argparse.ArgumentParser(description="Query GraphRAG graph")
    parser.add_argument("--db", required=True, help="Path to DuckDB")
    parser.add_argument("--req-id", help="CRU/REQ ID to anchor")
    parser.add_argument("--query-text", help="Natural language query")
    parser.add_argument("--task", required=True, choices=["test_generation", "debug", "impact"])
    parser.add_argument("--module", default="")
    parser.add_argument("--version", default="")
    parser.add_argument("--doctype", default="")
    parser.add_argument("--out", default="context_pack.json")
    parser.add_argument("--rebuild-emb", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
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
        graph_result = graph_retrieve(graph_store, anchors, query.task)

        if should_trigger_vector_fallback(graph_result, min_evidence=3, min_avg_conf=0.65):
            vector_hits = vector_search(
                graph_store=graph_store,
                query_text=query.query_text or query.req_id or "",
                filters=query.filters,
                node_types=["CHUNK", "REQ", "DEFECT", "FAILURE"],
                top_k=30,
            )
            graph_result = merge_graph_and_vector(
                graph_store,
                graph_result,
                vector_hits,
                top_k=query.k_evidence,
            )

        context_pack = build_context_pack(
            graph_store=graph_store,
            anchors=anchors,
            graph_result=graph_result,
            k_evidence=query.k_evidence,
            k_parent=query.k_parent,
        )

        context_pack_dict = context_pack_to_dict(context_pack)

        if args.task == "test_generation":
            try:
                generated_response = generate_tests_from_context_pack(
                    context_pack=context_pack_dict,
                    provider="ollama",
                )
            except Exception as e:
                generated_response = {
                    "task": args.task,
                    "status": "generation_failed",
                    "message": str(e),
                }
        else:
            generated_response = {
                "task": args.task,
                "status": "unsupported_task",
                "message": f"No generator is wired for task={args.task!r}",
            }

        if args.json:
            print(json.dumps({
                "context_pack": context_pack_dict,
                "generated_tests": to_serializable(generated_response),
            }, indent=2, default=str))
            return

        Path(args.out).write_text(
            json.dumps(context_pack_dict, indent=2, default=str),
            encoding="utf-8"
        )

        print(f"✅ Context Pack saved: {args.out}")
        print(f"Anchors: {len(context_pack.anchors)}")
        print(f"Chunks: {len(context_pack.evidence_chunks)}")
        print(f"Parents: {len(context_pack.parent_context)}")
        print(f"Paths: {len(context_pack.trace_paths)}")
        print(f"Related: {len(context_pack.related_nodes)}")
        print(f"Warnings: {len(context_pack.warnings)}")

        if args.task == "test_generation":
            print("\nGenerated Tests:")
            print(json.dumps(to_serializable(generated_response), indent=2, default=str))

    finally:
        graph_store.close()


if __name__ == "__main__":
    main()