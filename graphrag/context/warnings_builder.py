def build_warnings(evidence_chunks: list[dict], threshold: float = 0.65) -> list[dict]:
    warnings = []
    if not evidence_chunks:
        warnings.append({
            "type": "MISSING_EVIDENCE",
            "message": "No evidence chunks found; downstream generation must produce open questions rather than guessed expected results.",
        })
        return warnings

    for row in evidence_chunks:
        confidence = float(row.get("confidence", row.get("score", 0.0)))
        provenance = row.get("provenance", "graph")
        if provenance != "graph" and confidence < threshold:
            warnings.append({
                "type": "LOW_CONFIDENCE_VECTOR",
                "chunk_id": row.get("chunk_id") or row.get("node_id"),
                "confidence": confidence,
            })
    return warnings
