"""
graphrag/models/contracts.py
==============================
FIX SUMMARY:
  [1] Added open_questions: List[OpenQuestion] to ContextPack – required by architecture.
      Downstream generators must emit open_questions when evidence is missing, NOT guess.
  [2] EvidenceChunk: score/confidence/provenance are real pass-through fields (not hardcoded).
  [3] Added QueryInput dataclass (used by anchor_resolver + query_router).
  [4] Anchor dataclass made explicit (was imported but not defined here in old version).
  [5] TracePath: path_confidence field added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional



# ── Task type enum ────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    TEST_GENERATION       = "test_generation"
    DEBUG                 = "debug"
    IMPACT                = "impact"
    ACCEPTANCE_VALIDATION = "acceptance_validation"


# ── Query input contract ──────────────────────────────────────────────────────

@dataclass
class QueryInput:
    task: TaskType
    req_id: Optional[str] = None
    query_text: Optional[str] = None
    filters: Dict[str, str] = field(default_factory=dict)
    k_evidence: int = 8
    k_parent: int = 3


# ── Anchor ────────────────────────────────────────────────────────────────────

@dataclass
class Anchor:
    node_id: str
    node_type: str
    score: float = 1.0
    provenance: str = "graph"   # "graph" | "vector"


# ── Evidence chunk ────────────────────────────────────────────────────────────

@dataclass
class EvidenceChunk:
    chunk_id: str
    chunk_type: str                          # "child" | "parent"
    text: str
    doc_id: str
    section_path: str
    source_locator: Dict[str, Any]
    module: Optional[str] = None
    version: Optional[str] = None
    # These must come from real retrieval metadata – never hardcoded to 1.0
    score: float = 0.0
    confidence: float = 0.0
    provenance: str = "graph"               # "graph" | "vector" | "inferred"
    similarity_score: Optional[float] = None
    needs_confirmation: bool = False        # True when provenance==vector & conf < threshold


# ── Trace path ────────────────────────────────────────────────────────────────

@dataclass
class TracePath:
    why: str
    path: List[Dict[str, Any]]
    path_confidence: float = 0.0            # product of edge confidences along path


# ── Related node ─────────────────────────────────────────────────────────────

@dataclass
class RelatedNode:
    node_type: str
    node_id: str
    relation: str


# ── Warning ───────────────────────────────────────────────────────────────────

@dataclass
class Warning:
    type: str
    message: str
    chunk_id: Optional[str] = None


# ── Open question (REQUIRED – replaces guessing when evidence is missing) ─────

@dataclass
class OpenQuestion:
    """
    Emitted when a generated step has no valid evidence_chunk_ids.
    Downstream code must NOT invent a citation – it must emit an OpenQuestion instead.
    """
    step_index: int
    reason: str
    chunk_ids_available: List[str] = field(default_factory=list)


# ── Context Pack (the only output contract of the RAG layer) ──────────────────

@dataclass
class ContextPack:
    anchors: List[Anchor]
    evidence_chunks: List[EvidenceChunk]
    parent_context: List[EvidenceChunk]
    trace_paths: List[TracePath]
    related_nodes: List[RelatedNode]
    warnings: List[Warning]
    open_questions: List[OpenQuestion]      # REQUIRED – was missing in old version