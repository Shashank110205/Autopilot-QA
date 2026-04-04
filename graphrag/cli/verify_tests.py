#!/usr/bin/env python3
import json
from pathlib import Path
import duckdb

# Fix Windows paths - use Path objects
TEST_FILE = Path("../../../05_AI_powered_TestCaseGeneration/output/optimized_test_cases_20260403_225627.json")
DB_PATH = Path("../../output/test_graph.duckdb")

print("=== TEST FILE STRUCTURE ===")
try:
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"File loaded: {TEST_FILE}")
    print(f"Keys: {list(data.keys())}")
    print(f"phase1_test_cases: {len(data.get('phase1_test_cases', []))}")
    print(f"phase2_test_cases: {len(data.get('phase2_test_cases', []))}")
    
    print("\n=== FIRST 3 TESTS ===")
    tests = data.get('phase1_test_cases', []) + data.get('phase2_test_cases', [])
    for i, test in enumerate(tests[:3]):
        print(f"  {i+1}. {test['test_id']} -> {test['requirement_id']}")
        
except FileNotFoundError:
    print(f"TEST FILE NOT FOUND: {TEST_FILE}")
    print("Run from graphrag/cli/ directory, adjust path if needed")

print("\n=== GRAPH DATABASE ===")
con = duckdb.connect(DB_PATH)
print(f"DB connected: {DB_PATH}")

print("\n=== CRU NODES SAMPLE ===")
crus = con.execute("SELECT node_id FROM nodes WHERE node_type = 'CRU' LIMIT 10").fetchall()
cru_ids = [row[0] for row in crus]
print("CRU IDs:", cru_ids)

print("\n=== TEST NODES COUNT ===")
test_count = con.execute("SELECT COUNT(*) FROM nodes WHERE node_type = 'TEST'").fetchone()[0]
print(f"TEST nodes: {test_count}")

print("\n=== TEST→CRU EDGES ===")
edge_count = con.execute("SELECT COUNT(*) FROM edges WHERE rel_type = 'TESTS'").fetchone()[0]
print(f"TESTS edges: {edge_count}")

print("\n=== MISSING TEST EDGE REASON? ===")
# Check if any TEST nodes exist but no edges
test_ids = con.execute("SELECT node_id FROM nodes WHERE node_type = 'TEST' LIMIT 5").fetchall()
if test_ids:
    print("Sample TEST IDs:", [row[0] for row in test_ids])
    
    # Check if their requirement_ids exist as CRU
    missing_links = 0
    for test_row in test_ids:
        test_id = test_row[0]
        # Extract req_id from test_id pattern TC_CRU-FR10-01_001 -> CRU-FR10-01
        if '_00' in test_id:
            req_id = test_id.split('_')[1]  # Extract CRU-FR10-01
            if not con.execute("SELECT 1 FROM nodes WHERE node_id = ?", [req_id]).fetchone():
                missing_links += 1
                print(f"  MISSING: {test_id} needs {req_id} (CRU not found)")
    print(f"Missing CRU links: {missing_links}")

con.close()