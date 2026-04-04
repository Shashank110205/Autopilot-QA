import argparse
import json
import pathlib

import duckdb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_index(db_path: str, vector_dir: str):
    vector_dir = pathlib.Path(vector_dir)
    vector_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    conn = duckdb.connect(db_path)

    # Get all text-bearing CHUNK nodes
    rows = conn.execute("""
        SELECT node_id, node_type, text, module, version
        FROM nodes 
        WHERE node_type = 'CHUNK' AND text IS NOT NULL AND LENGTH(text) > 20
        ORDER BY module, version
    """).fetchall()

    node_ids = []
    texts = []
    for row in rows:
        node_ids.append(row[0])
        texts.append(row[2])

    embeddings = model.encode(texts, normalize_embeddings=True).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, str(vector_dir / "chunk_index.faiss"))
    (vector_dir / "chunk_index_ids.json").write_text(
        json.dumps(node_ids), encoding="utf-8"
    )
    (vector_dir / "embedding_manifest.json").write_text(
        json.dumps({
            "model": model.get_sentence_embedding_dimension(),
            "embedding_model": model[0].name_or_path,
            "total_embeddings": len(node_ids),
            "source_db": db_path,
        }),
        encoding="utf-8",
    )

    print(f"Indexed {len(node_ids)} chunk nodes")
    print(f"Saved to {vector_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--vector-dir", required=True)
    args = parser.parse_args()
    build_index(args.db, args.vector_dir)