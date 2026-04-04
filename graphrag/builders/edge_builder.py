import json
import hashlib
from datetime import datetime, timezone


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _make_parent_node_id(doc_id: str, section_path: str) -> str:
    return f"P_{_sha256(doc_id + section_path)}"


def _make_child_node_id(doc_id: str, section_path: str, cru_id: str, clause_text: str) -> str:
    normalized = clause_text.lower().strip()
    return f"CH_{_sha256(doc_id + section_path + cru_id + normalized)}"


def _derive_clause_text(cru: dict) -> str:
    actor = cru.get("actor")
    action = cru.get("action")
    constraint = cru.get("constraint")

    if action is None or str(action).strip() == "":
        if constraint is not None and str(constraint).strip() != "":
            action = f"must satisfy constraint: {constraint}"
        else:
            action = "unspecified behavior"

    clause_text = f"{actor} {action}"
    if constraint and str(constraint).strip() and str(constraint) not in clause_text:
        clause_text = f"{clause_text} {constraint}"
    return clause_text


def build_supported_by_edges(graph_store, cru_units_path: str, chunked_crus_path: str):
    with open(chunked_crus_path, "r", encoding="utf-8") as fh:
        chunk_data = json.load(fh)

    chunks = chunk_data.get("chunks")
    if chunks is None:
        raise ValueError("build_supported_by_edges: top-level 'chunks' array is missing")

    build_ts = datetime.now(timezone.utc)
    edges_written = 0

    for chunk_entry in chunks:
        traceability = chunk_entry.get("traceability") or {}
        sections = traceability.get("sections") or []
        doc_ids = traceability.get("doc_ids") or []
        source_locators = traceability.get("source_locators") or []

        cru_ids = chunk_entry.get("cru_ids", [])
        crus = chunk_entry.get("crus", [])

        if not cru_ids or not crus:
            continue

        first_cru_id = cru_ids[0]
        first_cru = graph_store.get_node(first_cru_id)
        if first_cru is None:
            continue

        section_path = sections[0] if sections else first_cru.get("section_path", "UNKNOWN_SECTION")
        doc_id = doc_ids[0] if doc_ids else first_cru.get("doc_id", "UNKNOWN_DOC")

        parent_node_id = _make_parent_node_id(doc_id, section_path)

        for cru in crus:
            cru_id = cru.get("cru_id")
            actor = cru.get("actor")
            if not cru_id or not actor:
                continue

            clause_text = _derive_clause_text(cru)
            child_node_id = _make_child_node_id(doc_id, section_path, cru_id, clause_text)

            if graph_store.get_node(child_node_id) is not None:
                graph_store.insert_edge({
                    "src_id": cru_id,
                    "src_type": "CRU",
                    "rel_type": "SUPPORTED_BY",
                    "dst_id": child_node_id,
                    "dst_type": "CHUNK",
                    "confidence": 1.0,
                    "evidence_chunk_id": child_node_id,
                    "extra_json": json.dumps({"kind": "child_support"}),
                    "created_at": build_ts,
                })
                edges_written += 1

            if graph_store.get_node(parent_node_id) is not None:
                graph_store.insert_edge({
                    "src_id": cru_id,
                    "src_type": "CRU",
                    "rel_type": "SUPPORTED_BY",
                    "dst_id": parent_node_id,
                    "dst_type": "CHUNK",
                    "confidence": 1.0,
                    "evidence_chunk_id": child_node_id,
                    "extra_json": json.dumps({"kind": "parent_context"}),
                    "created_at": build_ts,
                })
                edges_written += 1

    return {"supported_by_edges_written": edges_written}


def build_parent_of_edges(graph_store, chunked_crus_path: str):
    with open(chunked_crus_path, "r", encoding="utf-8") as fh:
        chunk_data = json.load(fh)

    chunks = chunk_data.get("chunks")
    if chunks is None:
        raise ValueError("build_parent_of_edges: top-level 'chunks' array is missing")

    build_ts = datetime.now(timezone.utc)
    edges_written = 0

    for chunk_entry in chunks:
        traceability = chunk_entry.get("traceability") or {}
        sections = traceability.get("sections") or []
        doc_ids = traceability.get("doc_ids") or []

        cru_ids = chunk_entry.get("cru_ids", [])
        crus = chunk_entry.get("crus", [])
        if not cru_ids or not crus:
            continue

        first_cru_id = cru_ids[0]
        first_cru = graph_store.get_node(first_cru_id)
        if first_cru is None:
            continue

        section_path = sections[0] if sections else first_cru.get("section_path", "UNKNOWN_SECTION")
        doc_id = doc_ids[0] if doc_ids else first_cru.get("doc_id", "UNKNOWN_DOC")
        parent_node_id = _make_parent_node_id(doc_id, section_path)

        for cru in crus:
            cru_id = cru.get("cru_id")
            actor = cru.get("actor")
            if not cru_id or not actor:
                continue

            clause_text = _derive_clause_text(cru)
            child_node_id = _make_child_node_id(doc_id, section_path, cru_id, clause_text)

            if graph_store.get_node(parent_node_id) is None or graph_store.get_node(child_node_id) is None:
                continue

            graph_store.insert_edge({
                "src_id": parent_node_id,
                "src_type": "CHUNK",
                "rel_type": "PARENT_OF",
                "dst_id": child_node_id,
                "dst_type": "CHUNK",
                "confidence": 1.0,
                "evidence_chunk_id": child_node_id,
                "extra_json": json.dumps({"kind": "hierarchy"}),
                "created_at": build_ts,
            })
            edges_written += 1

    return {"parent_of_edges_written": edges_written}

print("[TEST EDGES] build_test_edges() called")
def build_test_edges(graph_store, test_file_path: str) -> dict:
    with open(test_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_tests = data.get("phase1_test_cases", []) + data.get("phase2_test_cases", [])
    edges_created = 0
    skipped = 0

    for test in all_tests:
        test_id = test.get("test_id")
        req_id = test.get("requirement_id")

        if not test_id or not req_id:
            skipped += 1
            continue

        if not graph_store.node_exists(test_id):
            skipped += 1
            continue

        if not graph_store.node_exists(req_id):
            skipped += 1
            continue

        edge = {
            "src_id": test_id,
            "src_type": "TEST",
            "rel_type": "TESTS",
            "dst_id": req_id,
            "dst_type": "CRU",
            "confidence": 1.0,
            "evidence_chunk_id": None,
            "extra_json": json.dumps({
                "priority": test.get("priority"),
                "test_type": test.get("test_type"),
                "generation_phase": test.get("generation_phase"),
            }),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        graph_store.insert_edge(edge)
        edges_created += 1

    print(f"[TEST EDGES] Processed={len(all_tests)}, Created={edges_created}, Skipped={skipped}")
    return {
        "edge_type": "TESTS",
        "tests_processed": len(all_tests),
        "edges_created": edges_created,
        "skipped": skipped,
    }