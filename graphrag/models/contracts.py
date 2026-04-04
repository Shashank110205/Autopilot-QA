from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class TaskType(str, Enum):
    TEST_GENERATION = "test_generation"
    DEBUG = "debug"
    IMPACT = "impact"


@dataclass
class QueryInput:
    task: TaskType
    req_id: Optional[str] = None
    query_text: Optional[str] = None
    filters: Dict[str, str] = field(default_factory=dict)
    k_evidence: int = 8
    k_parent: int = 3


@dataclass
class Anchor:
    node_id: str
    node_type: str
    score: float
    provenance: str  # "graph" or "vector"


@dataclass
class EvidenceChunk:
    chunk_id: str
    chunk_type: str  # "child" or "parent"
    text: str
    doc_id: str
    section_path: str
    source_locator: Dict[str, Any]
    score: float
    confidence: float
    provenance: str


@dataclass
class TracePath:
    why: str
    path: List[Dict[str, Any]]


@dataclass
class RelatedNode:
    node_type: str
    node_id: str
    relation: str


@dataclass
class ContextPack:
    anchors: List[Anchor]
    evidence_chunks: List[EvidenceChunk]
    parent_context: List[EvidenceChunk]
    trace_paths: List[TracePath]
    related_nodes: List[RelatedNode]
    warnings: List[Dict[str, Any]]