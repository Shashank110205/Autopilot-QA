def merge_and_rerank(graph_chunks: list[dict], vector_chunks: list[dict] | None = None) -> list[dict]:
    merged = {}

    for item in graph_chunks or []:
        row = dict(item)
        row["score"] = max(float(row.get("score", 0.0)), 1.0)
        row["provenance"] = row.get("provenance", "graph")
        merged[row["chunk_id"]] = row

    for item in vector_chunks or []:
        row = dict(item)
        chunk_id = row.get("chunk_id") or row.get("node_id")
        if not chunk_id:
            continue
        if chunk_id in merged:
            merged[chunk_id]["score"] = max(merged[chunk_id]["score"], float(row.get("score", row.get("similarity_score", 0.0))))
            merged[chunk_id]["provenance"] = "graph|vector"
        else:
            row["chunk_id"] = chunk_id
            row["score"] = float(row.get("score", row.get("similarity_score", 0.0)))
            merged[chunk_id] = row

    return sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)
