"""
compare_graphrag.py
====================
Research paper comparison tool.

Captures a complete "snapshot" of:
  1. Graph statistics (node/edge counts per type)
  2. Query result for a given req_id (context pack + generated tests)
  3. Integrity report summary

Usage
-----
# STEP A – capture BEFORE snapshot (before your new build)
python compare_graphrag.py snapshot \
    --db graphrag/output/graphrag.duckdb \
    --req-id CRU-FR10-01 \
    --label before \
    --out comparison/

# STEP B – rebuild the graph with new data, then capture AFTER snapshot
python -m graphrag.cli.build_graph \
    --db graphrag/output/graphrag.duckdb \
    --cru 03_CRU_Normalization/output/cru_units.json \
    --chunks 04_Semantic_Chunking_and_Domain_Tagging/output/chunked_crus_with_domain.json \
    --tests 05_AI_powered_TestCaseGeneration/output/llama3-8b/after/llama.json \
    --out graphrag/output/runtime_check_new

python compare_graphrag.py snapshot \
    --db graphrag/output/graphrag.duckdb \
    --req-id CRU-FR10-01 \
    --label after \
    --out comparison/

# STEP C – generate the HTML comparison report
python compare_graphrag.py report \
    --before comparison/snapshot_before.json \
    --after  comparison/snapshot_after.json \
    --out    comparison/comparison_report.html
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── Snapshot capture ──────────────────────────────────────────────────────────

def capture_snapshot(db_path: str, req_id: str, label: str, out_dir: str) -> dict:
    """
    Runs graph stats + query against the current DB state and saves a snapshot JSON.
    Does NOT require sentence_transformers (graph-only path via req_id).
    """
    from graphrag.storage.graph_store import GraphStore
    from graphrag.retrieval.query_router import route_query
    from graphrag.retrieval.anchor_resolver import resolve_anchors
    from graphrag.retrieval.graph_retriever import graph_retrieve
    from graphrag.context.context_pack_builder import build_context_pack

    print(f"[COMPARE] Capturing '{label}' snapshot from {db_path} ...")
    gs = GraphStore(db_path)

    try:
        # ── 1. Graph statistics ───────────────────────────────────────────────
        stats = gs.stats()

        # ── 2. Per-type node samples (first 3 of each type for inspection) ────
        node_samples = {}
        for node_type in stats["nodes"]:
            rows = gs.query(
                "SELECT node_id, node_type, title, text, module, version "
                "FROM nodes WHERE node_type = ? LIMIT 3",
                [node_type]
            )
            node_samples[node_type] = rows

        # ── 3. Edge samples ───────────────────────────────────────────────────
        edge_samples = {}
        for rel_type in stats["edges"]:
            rows = gs.query(
                "SELECT src_id, rel_type, dst_id, confidence, extra_json "
                "FROM edges WHERE rel_type = ? LIMIT 3",
                [rel_type]
            )
            edge_samples[rel_type] = rows

        # ── 4. Integrity check summary ────────────────────────────────────────
        try:
            from graphrag.validation.integrity_checks import run_integrity_checks
            integrity = run_integrity_checks(db_path)
            integrity_summary = integrity["summary"]
        except Exception as e:
            integrity_summary = {"error": str(e)}

        # ── 5. Query result (graph-only via req_id) ───────────────────────────
        query_result = {}
        context_pack_raw = {}
        try:
            query = route_query({
                "task": "test_generation",
                "req_id": req_id,
                "query_text": None,
                "filters": {},
                "k_evidence": 8,
                "k_parent": 3,
            })
            anchors = resolve_anchors(gs, query)
            graph_result = graph_retrieve(gs, anchors, query.task, filters={})
            context_pack = build_context_pack(
                graph_store=gs,
                anchors=anchors,
                graph_result=graph_result,
                k_evidence=8,
                k_parent=3,
            )

            # Serialise context pack fields
            def _s(obj):
                if hasattr(obj, "__dict__"):
                    return obj.__dict__
                return str(obj)

            context_pack_raw = {
                "anchors": [_s(a) for a in context_pack.anchors],
                "evidence_chunks": [_s(c) for c in context_pack.evidence_chunks],
                "parent_context":  [_s(c) for c in context_pack.parent_context],
                "trace_paths":     [_s(t) for t in context_pack.trace_paths],
                "related_nodes":   [_s(r) for r in context_pack.related_nodes],
                "warnings":        [_s(w) for w in context_pack.warnings],
                "open_questions":  [_s(q) for q in context_pack.open_questions],
            }
            query_result = {
                "req_id":             req_id,
                "anchor_count":       len(context_pack.anchors),
                "evidence_count":     len(context_pack.evidence_chunks),
                "parent_count":       len(context_pack.parent_context),
                "trace_path_count":   len(context_pack.trace_paths),
                "related_node_count": len(context_pack.related_nodes),
                "warning_count":      len(context_pack.warnings),
                "open_question_count":len(context_pack.open_questions),
                "avg_confidence":     round(
                    sum(c.confidence for c in context_pack.evidence_chunks) /
                    max(len(context_pack.evidence_chunks), 1), 4
                ),
                "avg_score":          round(
                    sum(c.score for c in context_pack.evidence_chunks) /
                    max(len(context_pack.evidence_chunks), 1), 4
                ),
                "warnings_text":      [_s(w) for w in context_pack.warnings],
            }
        except Exception as e:
            query_result = {"error": str(e)}

        snapshot = {
            "label":            label,
            "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
            "db_path":          db_path,
            "req_id":           req_id,
            "graph_stats":      stats,
            "node_samples":     node_samples,
            "edge_samples":     edge_samples,
            "integrity":        integrity_summary,
            "query_result":     query_result,
            "context_pack":     context_pack_raw,
        }

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / f"snapshot_{label}.json"
        out_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        print(f"[COMPARE] Snapshot saved: {out_path}")
        print(f"[COMPARE] Nodes: {stats['nodes']}")
        print(f"[COMPARE] Edges: {stats['edges']}")
        return snapshot

    finally:
        gs.close()


# ── HTML report generation ────────────────────────────────────────────────────

def generate_report(before_path: str, after_path: str, out_path: str):
    before = json.loads(Path(before_path).read_text())
    after  = json.loads(Path(after_path).read_text())

    html = _build_html(before, after)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"[COMPARE] Report saved: {out_path}")


def _delta(a, b):
    """Return signed delta string."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = b - a
        if d > 0: return f"+{d}"
        if d < 0: return str(d)
        return "±0"
    return "–"


def _build_html(before: dict, after: dict) -> str:
    b_stats = before.get("graph_stats", {})
    a_stats = after.get("graph_stats",  {})
    b_q = before.get("query_result", {})
    a_q = after.get("query_result",  {})
    b_int = before.get("integrity", {})
    a_int = after.get("integrity",  {})

    all_node_types = sorted(set(b_stats.get("nodes", {}).keys()) | set(a_stats.get("nodes", {}).keys()))
    all_edge_types = sorted(set(b_stats.get("edges", {}).keys()) | set(a_stats.get("edges", {}).keys()))

    def node_rows():
        rows = ""
        for t in all_node_types:
            bv = b_stats.get("nodes", {}).get(t, 0)
            av = a_stats.get("nodes", {}).get(t, 0)
            d  = _delta(bv, av)
            cls = "pos" if av > bv else ("neg" if av < bv else "neu")
            rows += f"<tr><td class='type'>{t}</td><td>{bv}</td><td>{av}</td><td class='{cls}'>{d}</td></tr>"
        return rows

    def edge_rows():
        rows = ""
        for t in all_edge_types:
            bv = b_stats.get("edges", {}).get(t, 0)
            av = a_stats.get("edges", {}).get(t, 0)
            d  = _delta(bv, av)
            cls = "pos" if av > bv else ("neg" if av < bv else "neu")
            rows += f"<tr><td class='type'>{t}</td><td>{bv}</td><td>{av}</td><td class='{cls}'>{d}</td></tr>"
        return rows

    def query_rows():
        keys = [
            ("evidence_count",      "Evidence chunks"),
            ("parent_count",        "Parent context"),
            ("trace_path_count",    "Trace paths"),
            ("related_node_count",  "Related nodes"),
            ("warning_count",       "Warnings"),
            ("open_question_count", "Open questions"),
            ("avg_confidence",      "Avg confidence"),
            ("avg_score",           "Avg score"),
        ]
        rows = ""
        for key, label in keys:
            bv = b_q.get(key, "–")
            av = a_q.get(key, "–")
            try:
                d  = _delta(float(bv), float(av))
                cls = "pos" if float(av) > float(bv) else ("neg" if float(av) < float(bv) else "neu")
            except Exception:
                d = "–"; cls = "neu"
            rows += f"<tr><td class='type'>{label}</td><td>{bv}</td><td>{av}</td><td class='{cls}'>{d}</td></tr>"
        return rows

    def chunk_cards(pack, label):
        chunks = pack.get("evidence_chunks", [])
        if not chunks:
            return f"<p class='empty'>No evidence chunks in {label} snapshot.</p>"
        cards = ""
        for c in chunks:
            conf  = c.get("confidence", 0)
            score = c.get("score", 0)
            prov  = c.get("provenance", "graph")
            text  = (c.get("text") or "")[:220]
            cid   = c.get("chunk_id", "?")
            sec   = c.get("section_path", "")
            bar_w = int(float(conf) * 100)
            cards += f"""
            <div class="chunk-card">
              <div class="chunk-header">
                <span class="cid">{cid}</span>
                <span class="prov prov-{prov}">{prov}</span>
              </div>
              <div class="chunk-sec">{sec}</div>
              <div class="chunk-text">{text}{'…' if len(c.get('text',''))>220 else ''}</div>
              <div class="chunk-meta">
                <span>score {score:.3f}</span>
                <span class="conf-bar-wrap"><span class="conf-bar" style="width:{bar_w}%"></span></span>
                <span>conf {conf:.3f}</span>
              </div>
            </div>"""
        return cards

    def warning_list(pack):
        warns = pack.get("warnings", [])
        if not warns:
            return "<p class='empty'>No warnings.</p>"
        items = ""
        for w in warns:
            wtype = w.get("type","?") if isinstance(w,dict) else str(w)
            wmsg  = w.get("message","") if isinstance(w,dict) else ""
            items += f"<div class='warn-item'><span class='warn-type'>{wtype}</span> {wmsg}</div>"
        return items

    b_ts = before.get("timestamp_utc","–")
    a_ts = after.get("timestamp_utc","–")
    req  = before.get("req_id","–")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GraphRAG Before / After – {req}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  :root {{
    --bg:       #0d1117;
    --surface:  #161b22;
    --surface2: #1e2530;
    --border:   #30363d;
    --text:     #e6edf3;
    --muted:    #8b949e;
    --before:   #58a6ff;
    --after:    #3fb950;
    --pos:      #3fb950;
    --neg:      #f85149;
    --neu:      #8b949e;
    --accent:   #f0883e;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;
  }}

  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.6;
  }}

  /* ── Header ── */
  .header {{
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 100%);
    border-bottom: 1px solid var(--border);
    padding: 40px 48px 32px;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content:'';
    position:absolute; inset:0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(88,166,255,.07) 0%, transparent 70%);
  }}
  .header h1 {{
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
  }}
  .header h1 span {{ color: var(--accent); }}
  .header p {{ color: var(--muted); font-size: 13px; }}
  .req-badge {{
    display:inline-block;
    background: rgba(88,166,255,.15);
    border: 1px solid rgba(88,166,255,.3);
    color: var(--before);
    font-family: var(--mono);
    font-size: 12px;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 12px;
  }}

  /* ── Layout ── */
  .container {{ max-width: 1280px; margin: 0 auto; padding: 32px 48px; }}
  .section {{ margin-bottom: 40px; }}
  .section-title {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-bottom: 20px;
  }}

  /* ── Timeline bar ── */
  .timeline {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 32px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
  }}
  .tl-node {{
    flex: 1;
    text-align: center;
  }}
  .tl-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}
  .tl-label.before {{ color: var(--before); }}
  .tl-label.after  {{ color: var(--after);  }}
  .tl-ts {{ font-family: var(--mono); font-size: 11px; color: var(--muted); }}
  .tl-arrow {{
    font-size: 24px;
    color: var(--border);
    flex-shrink: 0;
  }}

  /* ── Tables ── */
  table {{ width:100%; border-collapse:collapse; }}
  th {{
    background: var(--surface2);
    color: var(--muted);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 13px;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,.02); }}
  td.type {{ font-weight: 600; color: var(--text); font-family: var(--sans); }}
  .table-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }}
  th:nth-child(2) {{ color: var(--before); }}
  th:nth-child(3) {{ color: var(--after);  }}

  /* ── Delta colours ── */
  .pos {{ color: var(--pos); font-weight: 600; }}
  .neg {{ color: var(--neg); font-weight: 600; }}
  .neu {{ color: var(--neu); }}

  /* ── Two-column grid ── */
  .cols2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .col-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
  }}
  .col-label.before {{ color: var(--before); }}
  .col-label.after  {{ color: var(--after);  }}

  /* ── Chunk cards ── */
  .chunk-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
    transition: border-color .15s;
  }}
  .chunk-card:hover {{ border-color: #444d56; }}
  .chunk-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom: 4px; }}
  .cid {{ font-family: var(--mono); font-size: 11px; color: var(--muted); }}
  .prov {{ font-size: 10px; font-weight:700; letter-spacing:1px; text-transform:uppercase; padding: 2px 7px; border-radius: 4px; }}
  .prov-graph  {{ background:rgba(63,185,80,.15);  color:var(--pos); }}
  .prov-vector {{ background:rgba(240,136,62,.15); color:var(--accent); }}
  .chunk-sec {{ font-size: 11px; color: var(--muted); margin-bottom: 6px; }}
  .chunk-text {{ font-size: 13px; line-height:1.55; color: #cdd5e0; margin-bottom: 10px; }}
  .chunk-meta {{ display:flex; align-items:center; gap:10px; font-family:var(--mono); font-size:11px; color:var(--muted); }}
  .conf-bar-wrap {{ flex:1; height:4px; background:var(--border); border-radius:2px; }}
  .conf-bar {{ display:block; height:100%; background: var(--after); border-radius:2px; }}
  .empty {{ color: var(--muted); font-style:italic; padding: 12px 0; }}

  /* ── Warning items ── */
  .warn-item {{ display:flex; gap:10px; align-items:flex-start; padding: 10px 14px; border-bottom: 1px solid var(--border); }}
  .warn-item:last-child {{ border-bottom:none; }}
  .warn-type {{ font-family:var(--mono); font-size:11px; font-weight:600; color:var(--neg); white-space:nowrap; padding: 1px 7px; background:rgba(248,81,73,.1); border-radius:4px; }}

  /* ── Integrity cards ── */
  .int-grid {{ display:grid; grid-template-columns: repeat(3,1fr); gap:16px; }}
  .int-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }}
  .int-card .val {{ font-size: 36px; font-weight:700; font-family:var(--mono); margin-bottom:4px; }}
  .int-card .lbl {{ font-size: 11px; color: var(--muted); letter-spacing:1px; text-transform:uppercase; }}
  .int-card.ok .val  {{ color: var(--pos); }}
  .int-card.err .val {{ color: var(--neg); }}
  .int-card.warn .val{{ color: var(--accent); }}

  /* ── Key findings box ── */
  .finding {{
    background: var(--surface);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 13px;
  }}
  .finding strong {{ color: var(--accent); }}
</style>
</head>
<body>

<div class="header">
  <div class="req-badge">{req}</div>
  <h1>GraphRAG <span>Before / After</span> Analysis</h1>
  <p>Research comparison — task: test_generation &nbsp;·&nbsp; req_id: {req}</p>
</div>

<div class="container">

  <!-- Timeline -->
  <div class="timeline">
    <div class="tl-node">
      <div class="tl-label before">Before</div>
      <div class="tl-ts">{b_ts}</div>
    </div>
    <div class="tl-arrow">→</div>
    <div class="tl-node">
      <div class="tl-label after">After</div>
      <div class="tl-ts">{a_ts}</div>
    </div>
  </div>

  <!-- Key Findings -->
  <div class="section">
    <div class="section-title">Key Findings</div>
    {_findings(before, after)}
  </div>

  <!-- Graph Node Stats -->
  <div class="section">
    <div class="section-title">Graph Node Counts</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Node Type</th><th>Before</th><th>After</th><th>Δ Delta</th></tr></thead>
        <tbody>{node_rows()}</tbody>
      </table>
    </div>
  </div>

  <!-- Graph Edge Stats -->
  <div class="section">
    <div class="section-title">Graph Edge Counts (Relations)</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Edge / Relation</th><th>Before</th><th>After</th><th>Δ Delta</th></tr></thead>
        <tbody>{edge_rows()}</tbody>
      </table>
    </div>
  </div>

  <!-- Query Result -->
  <div class="section">
    <div class="section-title">Query Result Metrics — req_id: {req}</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Δ Delta</th></tr></thead>
        <tbody>{query_rows()}</tbody>
      </table>
    </div>
  </div>

  <!-- Integrity -->
  <div class="section">
    <div class="section-title">Integrity Check Summary</div>
    <div class="cols2">
      <div>
        <div class="col-label before">Before</div>
        <div class="int-grid">
          <div class="int-card {'ok' if b_int.get('errors',0)==0 else 'err'}">
            <div class="val">{b_int.get('errors','–')}</div>
            <div class="lbl">Errors</div>
          </div>
          <div class="int-card warn">
            <div class="val">{b_int.get('warnings','–')}</div>
            <div class="lbl">Warnings</div>
          </div>
          <div class="int-card {'ok' if b_int.get('passed') else 'err'}">
            <div class="val">{'✓' if b_int.get('passed') else '✗'}</div>
            <div class="lbl">Passed</div>
          </div>
        </div>
      </div>
      <div>
        <div class="col-label after">After</div>
        <div class="int-grid">
          <div class="int-card {'ok' if a_int.get('errors',0)==0 else 'err'}">
            <div class="val">{a_int.get('errors','–')}</div>
            <div class="lbl">Errors</div>
          </div>
          <div class="int-card warn">
            <div class="val">{a_int.get('warnings','–')}</div>
            <div class="lbl">Warnings</div>
          </div>
          <div class="int-card {'ok' if a_int.get('passed') else 'err'}">
            <div class="val">{'✓' if a_int.get('passed') else '✗'}</div>
            <div class="lbl">Passed</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Evidence Chunks -->
  <div class="section">
    <div class="section-title">Evidence Chunks Retrieved</div>
    <div class="cols2">
      <div>
        <div class="col-label before">Before</div>
        {chunk_cards(before.get('context_pack',{}), 'before')}
      </div>
      <div>
        <div class="col-label after">After</div>
        {chunk_cards(after.get('context_pack',{}), 'after')}
      </div>
    </div>
  </div>

  <!-- Warnings -->
  <div class="section">
    <div class="section-title">Retrieval Warnings</div>
    <div class="cols2">
      <div>
        <div class="col-label before">Before</div>
        <div class="table-wrap">{warning_list(before.get('context_pack',{}))}</div>
      </div>
      <div>
        <div class="col-label after">After</div>
        <div class="table-wrap">{warning_list(after.get('context_pack',{}))}</div>
      </div>
    </div>
  </div>

</div>
</body>
</html>"""


def _findings(before: dict, after: dict) -> str:
    b_stats = before.get("graph_stats", {})
    a_stats = after.get("graph_stats",  {})
    b_q = before.get("query_result", {})
    a_q = after.get("query_result",  {})

    findings = []

    # Node growth
    b_total = sum(b_stats.get("nodes", {}).values())
    a_total = sum(a_stats.get("nodes", {}).values())
    if a_total != b_total:
        d = a_total - b_total
        findings.append(f"<strong>Graph grew by {abs(d)} nodes</strong> ({b_total} → {a_total})")

    # Edge growth
    b_edges = sum(b_stats.get("edges", {}).values())
    a_edges = sum(a_stats.get("edges", {}).values())
    if a_edges != b_edges:
        d = a_edges - b_edges
        findings.append(f"<strong>{abs(d)} new edges</strong> added ({b_edges} → {a_edges})")

    # New node types
    new_types = set(a_stats.get("nodes", {}).keys()) - set(b_stats.get("nodes", {}).keys())
    if new_types:
        findings.append(f"<strong>New node types introduced:</strong> {', '.join(sorted(new_types))}")

    # New edge types
    new_rels = set(a_stats.get("edges", {}).keys()) - set(b_stats.get("edges", {}).keys())
    if new_rels:
        findings.append(f"<strong>New relation types:</strong> {', '.join(sorted(new_rels))}")

    # Evidence quality
    b_ev = b_q.get("evidence_count", 0)
    a_ev = a_q.get("evidence_count", 0)
    if isinstance(b_ev, int) and isinstance(a_ev, int) and a_ev != b_ev:
        findings.append(f"<strong>Evidence chunks retrieved:</strong> {b_ev} → {a_ev}")

    b_conf = b_q.get("avg_confidence", 0)
    a_conf = a_q.get("avg_confidence", 0)
    try:
        if abs(float(a_conf) - float(b_conf)) > 0.01:
            findings.append(f"<strong>Average confidence changed:</strong> {b_conf} → {a_conf}")
    except Exception:
        pass

    # Warnings
    b_w = b_q.get("warning_count", 0)
    a_w = a_q.get("warning_count", 0)
    if b_w != a_w:
        direction = "reduced" if a_w < b_w else "increased"
        findings.append(f"<strong>Warnings {direction}:</strong> {b_w} → {a_w}")

    if not findings:
        findings = ["No significant differences detected between snapshots."]

    return "".join(f'<div class="finding">{f}</div>' for f in findings)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG before/after comparison tool for research"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # snapshot sub-command
    snap = sub.add_parser("snapshot", help="Capture a graph + query snapshot")
    snap.add_argument("--db",     required=True, help="Path to DuckDB database")
    snap.add_argument("--req-id", required=True, help="CRU req_id to query")
    snap.add_argument("--label",  required=True, help="'before' or 'after'")
    snap.add_argument("--out",    default="comparison", help="Output directory")

    # report sub-command
    rep = sub.add_parser("report", help="Generate HTML comparison report")
    rep.add_argument("--before", required=True, help="Path to before snapshot JSON")
    rep.add_argument("--after",  required=True, help="Path to after snapshot JSON")
    rep.add_argument("--out",    default="comparison/comparison_report.html")

    args = parser.parse_args()

    if args.command == "snapshot":
        capture_snapshot(args.db, args.req_id, args.label, args.out)
    elif args.command == "report":
        generate_report(args.before, args.after, args.out)


if __name__ == "__main__":
    main()