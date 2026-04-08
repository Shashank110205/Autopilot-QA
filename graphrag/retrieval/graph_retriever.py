"""
graphrag/retrieval/graph_retriever.py
=======================================
FIX SUMMARY:
  [1] Added path_confidence scoring: PRODUCT of edge.confidence along each path.
      Old version collected chunks by BFS order with no scoring – traversal-order
      dependent, not confidence-driven.
  [2] Added hop_penalty (0.9 ** hops) per architecture §5.4.
  [3] Added filter boost: +20% for module match, +10% for version match.
  [4] Child chunks are ranked by final_score before return.
  [5] path_confidence is preserved on each candidate so context_pack_builder and
      warnings_builder can use it downstream.
  [6] query_router.TASK_RELATIONS imported for relation whitelist (unchanged structure).
  [7] import json moved to module level (was inside the while loop body —
      re-imported on every BFS iteration).
"""
from __future__ import annotations

import json
from collections import deque
from typing import Dict, List

from graphrag.models.contracts import Anchor
from graphrag.retrieval.query_router import TASK_RELATIONS
from graphrag.storage.graph_store import GraphStore

HOP_PENALTY = 0.9   # multiplied per hop beyond the anchor node


def graph_retrieve(
    graph_store: GraphStore,
    anchors: List[Anchor],
    task: str,
    max_hops: int = 2,
    filters: dict | None = None,
) -> Dict[str, List]:
    """
    Task-specific k-hop BFS with path-confidence scoring.

    Returns:
      {
        "evidence_nodes": [... with score, path_confidence, path_edges],
        "trace_paths":    [...],
        "related_nodes":  [...],
      }
    """
    filters = filters or {}
    policy = TASK_RELATIONS.get(task, {})
    forward_rels = set(policy.get("forward", []))
    reverse_rels = set(policy.get("reverse", []))

    # visited key: node_id (not (node_id, depth) – avoids duplicate scoring at
    # different depths; we keep the highest-scoring path per chunk)
    best_score: Dict[str, float] = {}
    best_candidate: Dict[str, dict] = {}

    related: List[dict] = []
    visited_edges: set = set()

    # queue: (node_id, depth, path_confidence, path_edges)
    queue: deque = deque()
    for anchor in anchors:
        queue.append((anchor.node_id, 0, 1.0, []))

    queued: set = {a.node_id for a in anchors}

    while queue:
        node_id, depth, path_conf, path_edges = queue.popleft()

        if depth > max_hops:
            continue

        node = graph_store.get_node(node_id)
        if node is None:
            continue

        # ── Score and collect child CHUNK evidence ────────────────────────────
        if node.get("node_type") == "CHUNK" and depth > 0:
            extra_raw = node.get("extra_json") or "{}"
            try:
                extra = json.loads(extra_raw)
            except Exception:
                extra = {}

            if extra.get("chunk_type") == "child":
                hop_penalty = HOP_PENALTY ** depth

                # Filter boost
                boost = 1.0
                if filters.get("module") and node.get("module") == filters["module"]:
                    boost *= 1.2
                if filters.get("version") and node.get("version") == filters["version"]:
                    boost *= 1.1

                final_score = round(path_conf * hop_penalty * boost, 6)

                if final_score > best_score.get(node_id, -1):
                    best_score[node_id] = final_score
                    best_candidate[node_id] = {
                        **node,
                        "score": final_score,
                        "path_confidence": round(path_conf, 6),
                        "hops": depth,
                        "path_edges": path_edges,
                    }

        if depth >= max_hops:
            continue

        # ── Forward traversal ─────────────────────────────────────────────────
        for edge in graph_store.get_edges_from(node_id, rel_types=list(forward_rels)):
            dst_id = edge["dst_id"]
            ekey = (node_id, edge["rel_type"], dst_id)
            if ekey in visited_edges:
                continue
            visited_edges.add(ekey)

            new_conf = path_conf * edge["confidence"]
            new_path = path_edges + [{
                "src": node_id, "rel": edge["rel_type"],
                "dst": dst_id, "conf": edge["confidence"],
            }]
            if dst_id not in queued or new_conf > best_score.get(dst_id, -1):
                queued.add(dst_id)
                queue.append((dst_id, depth + 1, new_conf, new_path))

            related.append({
                "node_type": edge["dst_type"],
                "node_id": dst_id,
                "relation": edge["rel_type"],
            })

        # ── Reverse traversal ─────────────────────────────────────────────────
        for edge in graph_store.get_edges_to(node_id, rel_types=list(reverse_rels)):
            src_id = edge["src_id"]
            ekey = (src_id, edge["rel_type"], node_id, "rev")
            if ekey in visited_edges:
                continue
            visited_edges.add(ekey)

            new_conf = path_conf * edge["confidence"]
            new_path = path_edges + [{
                "src": src_id, "rel": edge["rel_type"],
                "dst": node_id, "conf": edge["confidence"], "direction": "reverse",
            }]
            if src_id not in queued:
                queued.add(src_id)
                queue.append((src_id, depth + 1, new_conf, new_path))

            related.append({
                "node_type": edge["src_type"],
                "node_id": src_id,
                "relation": edge["rel_type"],
            })

    # ── Rank child evidence by score DESC ─────────────────────────────────────
    ranked = sorted(best_candidate.values(), key=lambda x: x["score"], reverse=True)

    # Build trace paths
    trace_paths = [
        {
            "why": f"evidence via task={task}",
            "path": c["path_edges"],
            "path_confidence": c["path_confidence"],
        }
        for c in ranked
    ]

    return {
        "evidence_nodes": ranked,
        "trace_paths": trace_paths,
        "related_nodes": related[:20],
    }