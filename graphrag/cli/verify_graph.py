#!/usr/bin/env python3
import sys
import duckdb
from pathlib import Path

def verify_graph(db_path: str):
    con = duckdb.connect(db_path)
    
    print("=== GRAPH STATS ===")
    stats = con.execute("SELECT COUNT(*) as total_nodes FROM nodes").fetchone()[0]
    edges = con.execute("SELECT COUNT(*) as total_edges FROM edges").fetchone()[0]
    print(f"Total nodes: {stats}")
    print(f"Total edges: {edges}")
    
    print("\n=== NODE TYPES ===")
    nodes = con.execute("""
        SELECT node_type, COUNT(*) as count 
        FROM nodes GROUP BY node_type ORDER BY count DESC
    """).fetchdf()
    print(nodes)
    
    print("\n=== EDGE TYPES ===")
    edges = con.execute("""
        SELECT rel_type, COUNT(*) as count, AVG(confidence) as avg_conf 
        FROM edges GROUP BY rel_type ORDER BY count DESC
    """).fetchdf()
    print(edges)
    
    print("\n=== TEST→CRU EDGES SAMPLE ===")
    sample = con.execute("""
        SELECT src_id, rel_type, dst_id, confidence 
        FROM edges 
        WHERE rel_type = 'TESTS' 
        LIMIT 5
    """).fetchdf()
    print(sample)
    
    con.close()

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "output/test_graph.duckdb"
    verify_graph(db_path)