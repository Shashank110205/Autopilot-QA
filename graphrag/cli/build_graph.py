import argparse
import json
import pathlib
from datetime import datetime, timezone

from graphrag.storage.graph_store import GraphStore
from graphrag.builders.cru_builder import build_cru_nodes
from graphrag.builders.chunk_builder import build_chunk_nodes
from graphrag.builders.test_builder import build_test_nodes
from graphrag.builders.edge_builder import (
    build_parent_of_edges,
    build_supported_by_edges,
    build_test_edges,
)


def write_report(out_dir: pathlib.Path, report: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build GraphRAG graph")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--cru", required=True, help="Path to cru_units.json")
    parser.add_argument("--chunks", required=True, help="Path to chunked_crus_with_domain.json")
    parser.add_argument("--tests", required=False, help="Path to optimized test cases JSON")
    parser.add_argument("--out", required=True, help="Output directory for build report")
    args = parser.parse_args()

    db_path = pathlib.Path(args.db)
    cru_path = pathlib.Path(args.cru)
    chunks_path = pathlib.Path(args.chunks)
    tests_path = pathlib.Path(args.tests) if args.tests else None
    out_dir = pathlib.Path(args.out)

    graph_store = GraphStore(str(db_path))

    try:
        print(f"[BUILD] DB: {db_path}")
        print(f"[BUILD] CRU file: {cru_path} (exists={cru_path.exists()})")
        print(f"[BUILD] CHUNKS file: {chunks_path} (exists={chunks_path.exists()})")
        if tests_path:
            print(f"[BUILD] TESTS file: {tests_path} (exists={tests_path.exists()})")
        else:
            print("[BUILD] TESTS file: not provided")

        if not cru_path.exists():
            raise FileNotFoundError(f"CRU file not found: {cru_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        build_cru_nodes(graph_store, str(cru_path))
        build_chunk_nodes(graph_store, str(chunks_path))

        edge_reports = [
            build_supported_by_edges(graph_store, str(cru_path), str(chunks_path)),
            build_parent_of_edges(graph_store, str(chunks_path)),
        ]

        if tests_path and tests_path.exists():
            print(f"[BUILD] Loading tests from: {tests_path}")

            test_node_report = build_test_nodes(graph_store, str(tests_path))
            edge_reports.append(test_node_report)

            test_edge_report = build_test_edges(graph_store, str(tests_path))
            edge_reports.append(test_edge_report)
        else:
            print("[BUILD] Skipping tests: file missing or --tests not provided")

        stats = graph_store.stats()
        report = {
            "status": "success",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "db": str(db_path),
                "cru": str(cru_path),
                "chunks": str(chunks_path),
                "tests": str(tests_path) if tests_path else None,
            },
            "stats": stats,
            "edge_reports": edge_reports,
        }
        write_report(out_dir, report)
        print(f"[BUILD] Success. Report written to: {out_dir / 'build_report.json'}")
        print(f"[BUILD] Stats: {stats}")

    except Exception as exc:
        error_report = {
            "status": "failed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
            "inputs": {
                "db": str(db_path),
                "cru": str(cru_path),
                "chunks": str(chunks_path),
                "tests": str(tests_path) if tests_path else None,
            },
        }
        write_report(out_dir, error_report)
        print(f"[BUILD] Failed: {exc}")
        raise

    finally:
        graph_store.close()


if __name__ == "__main__":
    main()