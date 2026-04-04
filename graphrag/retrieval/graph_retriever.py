from collections import deque
from typing import Dict, List, Tuple
from graphrag.storage.graph_store import GraphStore
from graphrag.models.contracts import Anchor
from graphrag.retrieval.query_router import TASK_RELATIONS


def graph_retrieve(
    graph_store: GraphStore, 
    anchors: List[Anchor], 
    task: str, 
    max_hops: int = 2
) -> Dict[str, List]:
    """Task-specific graph traversal."""
    policy = TASK_RELATIONS[task]
    forward_rels = set(policy["forward"])
    reverse_rels = set(policy["reverse"])
    
    evidence = []
    paths = []
    related = []
    visited = set()
    
    queue = deque([
        (anchor.node_id, 0, [anchor.node_id], []) 
        for anchor in anchors
    ])
    
    while queue:
        node_id, depth, path_ids, path_edges = queue.popleft()
        
        if (node_id, depth) in visited or depth > max_hops:
            continue
        visited.add((node_id, depth))
        
        node = graph_store.get_node(node_id)
        if not node:
            continue
        
        # Collect evidence (prioritize CHUNK)
        if node["node_type"] == "CHUNK":
            evidence.append(node)
            paths.append({
                "target": node_id,
                "path": path_edges
            })
        
        # Forward traversal
        for edge in graph_store.get_edges_from(node_id):
            if edge["rel_type"] in forward_rels:
                next_path_edges = path_edges + [{
                    "src": node_id,
                    "rel": edge["rel_type"],
                    "dst": edge["dst_id"],
                    "conf": edge["confidence"]
                }]
                queue.append((edge["dst_id"], depth + 1, path_ids + [edge["dst_id"]], next_path_edges))
                related.append({
                    "node_type": edge["dst_type"],
                    "node_id": edge["dst_id"],
                    "relation": edge["rel_type"]
                })
        
        # Reverse traversal
        for edge in graph_store.get_edges_to(node_id):
            if edge["rel_type"] in reverse_rels:
                next_path_edges = path_edges + [{
                    "src": edge["src_id"],
                    "rel": edge["rel_type"],
                    "dst": node_id,
                    "conf": edge["confidence"]
                }]
                queue.append((edge["src_id"], depth + 1, [edge["src_id"]] + path_ids, next_path_edges))
                related.append({
                    "node_type": edge["src_type"],
                    "node_id": edge["src_id"],
                    "relation": edge["rel_type"]
                })
    
    return {
        "evidence_nodes": evidence,
        "trace_paths": paths,
        "related_nodes": related[:20]  # Limit
    }