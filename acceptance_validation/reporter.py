# reporter.py — Autopilot-QA CAU Layer
# Assembles cau_output.json and prints the console summary.

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cau_output(
    cau_units: list[dict],
    gap_report: dict,
) -> dict:
    """
    Assemble the top-level cau_output.json structure.
    """
    summary = _build_summary(cau_units, gap_report)

    output = {
        'metadata': {
            'pipeline': config.PIPELINE_NAME,
            'version':  config.PIPELINE_VERSION,
        },
        'summary':          summary,
        'cau_units':        cau_units,
        'traceability_gaps': gap_report,
    }
    return output


def write_cau_json(output: dict, out_dir: Path) -> Path:
    """Write cau_output.json to out_dir and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / config.CAU_JSON_FILENAME
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", path)
    return path


def print_summary(output: dict) -> None:
    """Print a concise run summary to stdout."""
    s = output['summary']
    gaps = output['traceability_gaps']

    print("\n" + "=" * 60)
    print(f"  Autopilot-QA  —  CAU Layer  v{config.PIPELINE_VERSION}")
    print("=" * 60)
    print(f"  CAU units parsed          : {s['total_cau_units']}")
    print(f"  UAT status breakdown      : {s['uat_status_breakdown']}")
    print(f"  Coverage classifications  : {s['coverage_classification']}")
    print(f"  Total CRUs linked         : {s['total_crus_linked']}")
    print(f"  Total test cases linked   : {s['total_test_cases_linked']}")
    print(f"  Coverage rate             : {s['coverage_rate_percent']:.1f}%")
    print("-" * 60)
    print(f"  Uncovered CRUs            : {s['uncovered_crus_count']}")
    print(f"  Missing req_ids           : {s['missing_req_ids_count']}")

    if gaps['missing_req_ids']:
        ids = ', '.join(g['req_id'] for g in gaps['missing_req_ids'])
        print(f"  → Missing req_ids         : {ids}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_summary(cau_units: list[dict], gap_report: dict) -> dict:
    total = len(cau_units)

    status_counts: Counter = Counter()
    coverage_counts: Counter = Counter()
    total_crus = 0
    total_tcs = 0
    covered_count = 0

    for cau in cau_units:
        status = cau.get('status', '').upper() or 'UNKNOWN'
        status_counts[status] += 1

        cov = cau.get('coverage', {})
        classification = cov.get('classification', 'UNKNOWN')
        coverage_counts[classification] += 1

        total_crus += cov.get('cru_count', 0)
        total_tcs  += cov.get('test_case_count', 0)

        if classification == 'FULL_COVERAGE':
            covered_count += 1

    coverage_rate = (covered_count / total * 100) if total else 0.0

    return {
        'total_cau_units':         total,
        'uat_status_breakdown':    dict(status_counts),
        'coverage_classification': dict(coverage_counts),
        'total_crus_linked':       total_crus,
        'total_test_cases_linked': total_tcs,
        'uncovered_crus_count':    len(gap_report.get('uncovered_crus', [])),
        'missing_req_ids_count':   len(gap_report.get('missing_req_ids', [])),
        'coverage_rate_percent':   round(coverage_rate, 1),
    }