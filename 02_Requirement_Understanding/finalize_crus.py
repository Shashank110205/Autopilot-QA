"""
CRU Finalization - Module 3, Phase B
Semantic Validation & Normalization

Purpose:
- Validate and normalize Candidate Requirement Assemblies (CRAs)
- Produce final Canonical Requirement Units (CRUs)
- Enforce semantic validity, testability, and atomicity
- Apply deterministic normalization rules
- Action Anchoring: Ensure single canonical action per requirement
- Grammar Enforcement: Ensure test-executable verb phrases

This module DECIDES based on evidence already present.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict


# ======================
# Configuration
# ======================

# Generic agent ontology (domain-agnostic)
VALID_ACTORS = {
    "System",
    "User", 
    "Admin",
    "External Service",
    "UI",
    "Backend"
}

# Modal verbs to strip
MODAL_VERBS = {
    "shall", "must", "should", "may", "will", 
    "would", "could", "can", "might"
}

# CRU types
VALID_TYPES = {
    "functional",
    "performance",
    "security",
    "usability",
    "reliability",
    "portability",
    "other"
}


# ======================
# Data Structures
# ======================

@dataclass
class CanonicalRequirementUnit:
    """Final validated CRU."""
    cru_id: str
    parent_requirement_id: str
    actor: str
    action: str
    constraint: Optional[str]
    outcome: Optional[str]
    type: str
    confidence: str  # high | medium | low
    traceability: Dict[str, Any]
    derived_from_cra: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ======================
# Actor Validation
# ======================

def normalize_actor(candidate_actor: Optional[str]) -> str:
    """Normalize actor to ontology.
    
    Rules:
    - If actor in ontology → keep
    - Else → default to "System"
    
    Returns: Valid actor from ontology
    """
    if not candidate_actor:
        return "System"
    
    # Clean and normalize
    actor = candidate_actor.strip()
    
    # Remove articles
    actor = re.sub(r'^(the|a|an)\s+', '', actor, flags=re.IGNORECASE)
    
    # Capitalize properly
    actor = actor.title()
    
    # Check if in ontology
    if actor in VALID_ACTORS:
        return actor
    
    # Map common variations
    actor_lower = actor.lower()
    
    # System variations
    if any(kw in actor_lower for kw in ['system', 'application', 'service', 'backend', 'database']):
        return "System"
    
    # User variations  
    if any(kw in actor_lower for kw in ['user', 'client', 'customer']):
        return "User"
    
    # Admin variations
    if any(kw in actor_lower for kw in ['admin', 'administrator', 'super']):
        return "Admin"
    
    # UI variations
    if any(kw in actor_lower for kw in ['ui', 'interface', 'display', 'screen']):
        return "UI"
    
    # Backend variations
    if any(kw in actor_lower for kw in ['backend', 'server', 'api']):
        return "Backend"
    
    # External service variations
    if any(kw in actor_lower for kw in ['external', 'third-party', 'service']):
        return "External Service"
    
    # Default fallback
    return "System"


# ======================
# Action Normalization
# ======================

def normalize_action(candidate_action: Optional[str]) -> Optional[str]:
    """Normalize action to clean verb phrase.
    
    Rules:
    - Remove modal verbs
    - Ensure verb + object/complement
    - Lowercase, concise
    
    Returns: Normalized action or None if invalid
    """
    if not candidate_action:
        return None
    
    action = candidate_action.strip()
    
    # Remove modal verbs
    for modal in MODAL_VERBS:
        # Remove modal at start with optional space
        action = re.sub(rf'\b{modal}\s+', '', action, flags=re.IGNORECASE)
    
    # Clean up whitespace
    action = re.sub(r'\s+', ' ', action).strip()
    
    # Lowercase
    action = action.lower()
    
    # Validation: Must have at least 2 words (verb + object/complement)
    words = action.split()
    if len(words) < 2:
        return None
    
    # Validation: Must contain a verb-like word
    # Check for common verb patterns
    has_verb = any(
        word.endswith(('s', 'es', 'ed', 'ing')) or
        word in ['allow', 'create', 'update', 'delete', 'authenticate', 
                'validate', 'insert', 'query', 'display', 'handle',
                'maintain', 'ensure', 'provide', 'establish', 'redirect']
        for word in words[:3]  # Check first 3 words
    )
    
    if not has_verb:
        return None
    
    # Remove trailing prepositions/incomplete phrases
    action = re.sub(r'\s+(via|by|with|using|for|to|from|in|at|on)$', '', action)
    
    # Final cleanup
    action = action.strip()
    
    # Must still have 2+ words after cleanup
    if len(action.split()) < 2:
        return None
    
    return action


# ======================
# Action Grammar Validation (Regex-based)
# ======================

# Common verb patterns (must be explicit verbs)
VERB_PATTERNS = [
    r'\b(allow|create|update|delete|authenticate|validate|insert|query|display|handle|maintain|ensure|provide|establish|redirect|filter|persist|trigger|hash|store|fetch|process|execute|generate|return|respond|render|load|support|enable)\w*\b',
]

# Verb suffixes that indicate action
VERB_SUFFIXES = ['s', 'es', 'ed', 'ing', 'ate', 'ates', 'ated', 'ating']

# Common noun patterns  
NOUN_PATTERNS = [
    r'\b(user|system|data|task|account|password|session|dashboard|credential|record|list|filter|request|response|message|error|page|view|file|report)\b',
    r'\b[A-Z][a-z]+\b',  # Capitalized words (potential nouns)
]

# Prepositions
PREPOSITIONS = {'by', 'with', 'using', 'for', 'to', 'from', 'in', 'at', 'on', 'upon', 'via', 'of', 'about'}


def has_verb_pattern(text: str) -> bool:
    """Check if text contains an explicit verb pattern."""
    # Check explicit verb list
    for pattern in VERB_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Check for words ending in verb suffixes, but exclude common nouns
    words = text.lower().split()
    for word in words:
        # Skip obvious nouns
        if word in ['users', 'tasks', 'accounts', 'messages', 'files', 'records', 'lists', 'pages', 'views', 'errors', 'exists']:
            continue
        # Check if ends with verb suffix
        for suffix in VERB_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return True
    
    return False


def has_noun_pattern(text: str) -> bool:
    """Check if text contains a noun pattern."""
    for pattern in NOUN_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def validate_action_grammar(action: str) -> Tuple[bool, int]:
    """Validate that action is a test-executable verb phrase using regex patterns.
    
    Grammar rules:
    - Must contain at least one verb pattern
    - Must contain at least one noun/object pattern
    - Must NOT start with preposition
    - Must NOT end with preposition
    - Must NOT be noun-only phrase
    - Length >= 2 tokens
    
    Returns: (is_valid, score) where score ranks action quality
    """
    if not action or len(action.strip()) < 3:
        return False, 0
    
    words = action.split()
    
    if len(words) < 2:
        return False, 0
    
    # Rule 1: Must contain at least one verb pattern
    has_verb = has_verb_pattern(action)
    if not has_verb:
        return False, 0
    
    # Rule 2: Must contain at least one noun/object
    has_noun = has_noun_pattern(action)
    if not has_noun:
        return False, 0
    
    # Rule 3: Must NOT start with preposition
    if words[0].lower() in PREPOSITIONS:
        return False, 0
    
    # Rule 4: Must NOT end with preposition
    if words[-1].lower() in PREPOSITIONS:
        return False, 0
    
    # Rule 5: Must NOT be all nouns (check for verb presence)
    if not has_verb:
        return False, 0
    
    # Calculate quality score
    score = 0
    score += 10 if has_verb else 0
    score += 10 if has_noun else 0
    score += len(words)  # Longer phrases generally better
    score += (3 if has_verb_pattern(words[0]) else 0)  # Starts with verb
    
    return True, score


def score_action_quality(action: str, source_field: str) -> int:
    """Score action quality for anchoring selection.
    
    Returns: Integer score (higher is better)
    """
    is_valid, base_score = validate_action_grammar(action)
    
    if not is_valid:
        return 0
    
    # Bonus for preferred source fields
    if source_field == "description":
        base_score += 20
    elif source_field == "system_behavior":
        base_score += 10
    
    return base_score


# ======================
# Action Anchoring
# ======================

def select_action_anchor(cras: List[Dict[str, Any]], parent_req_id: str) -> Optional[str]:
    """Select a single canonical action anchor for a requirement.
    
    Strategy:
    1. Collect all candidate actions from CRAs
    2. Score each action using grammar validation
    3. Select highest-scoring, longest valid action
    4. Prefer actions from description or system_behavior fields
    
    Returns: Anchor action or None if no valid action found
    """
    req_cras = [cra for cra in cras if cra.get("parent_requirement_id") == parent_req_id]
    
    if not req_cras:
        return None
    
    # Collect actions with metadata
    action_candidates = []
    
    for cra in req_cras:
        action = cra.get("candidate_action")
        if not action:
            continue
        
        # Normalize first
        normalized = normalize_action(action)
        if not normalized:
            continue
        
        # Infer source field from CRA ID
        cra_id = cra.get("cra_id", "")
        source_field = "other"
        if "description" in cra_id:
            source_field = "description"
        elif "system_behavior" in cra_id:
            source_field = "system_behavior"
        elif "outputs" in cra_id:
            source_field = "outputs"
        
        # Score the action
        score = score_action_quality(normalized, source_field)
        
        if score > 0:
            action_candidates.append({
                "action": normalized,
                "score": score,
                "source_field": source_field,
                "length": len(normalized.split())
            })
    
    if not action_candidates:
        return None
    
    # Sort by score (descending), then length (descending)
    action_candidates.sort(key=lambda x: (x["score"], x["length"]), reverse=True)
    
    # Return best action
    return action_candidates[0]["action"]


def create_action_anchors(cras: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Create action anchors for all requirements.
    
    Returns: Dict mapping parent_requirement_id -> anchor_action
    """
    # Get unique requirement IDs
    req_ids = list(set(cra.get("parent_requirement_id") for cra in cras))
    
    anchors = {}
    
    for req_id in req_ids:
        anchor = select_action_anchor(cras, req_id)
        anchors[req_id] = anchor
    
    return anchors


# ======================
# Constraint Processing
# ======================

def split_constraints(constraint_text: Optional[str]) -> List[str]:
    """Split compound constraints into atomic constraints.
    
    Returns: List of individual constraints
    """
    if not constraint_text:
        return []
    
    # Split on pipe separator (from Module 2)
    constraints = constraint_text.split('|')
    
    # Clean and filter
    cleaned = []
    for c in constraints:
        c = c.strip()
        if len(c) > 3:  # Minimum meaningful constraint
            cleaned.append(c)
    
    return cleaned


# ======================
# Outcome Validation
# ======================

def validate_outcome(outcome_text: Optional[str]) -> Optional[str]:
    """Validate that outcome is observable.
    
    Rules:
    - Must be observable (messages, UI response, returned value)
    - Drop if explanatory or vague
    
    Returns: Valid outcome or None
    """
    if not outcome_text:
        return None
    
    outcome = outcome_text.strip()
    
    # Too short
    if len(outcome) < 5:
        return None
    
    # Explanatory phrases to reject
    explanatory_patterns = [
        r'^(this|that|it)\s+',
        r'establishes',
        r'ensures',
        r'provides',
        r'supports'
    ]
    
    for pattern in explanatory_patterns:
        if re.search(pattern, outcome, re.IGNORECASE):
            return None
    
    # Must contain observable indicators
    observable_indicators = [
        'message', 'error', 'success', 'confirmation',
        'display', 'show', 'render', 'page', 'view',
        'return', 'response', 'output', 'result'
    ]
    
    outcome_lower = outcome.lower()
    if any(indicator in outcome_lower for indicator in observable_indicators):
        return outcome
    
    return None


# ======================
# Type Inference
# ======================

def infer_cru_type(
    parent_req_id: str,
    action: str,
    constraint: Optional[str]
) -> str:
    """Infer CRU type using lexical cues.
    
    Returns: One of VALID_TYPES
    """
    # Combine text for analysis
    text = f"{action} {constraint or ''}"
    text_lower = text.lower()
    
    # Performance indicators
    if any(kw in text_lower for kw in [
        'latency', 'throughput', 'percentile', 'concurrent',
        'response time', 'ms', 'second', 'performance',
        'speed', 'load time'
    ]):
        return "performance"
    
    # Security indicators
    if any(kw in text_lower for kw in [
        'auth', 'role', 'permission', 'encryption', 'hash',
        'security', 'password', 'credential', 'access control',
        'https', 'ssl', 'tls'
    ]):
        return "security"
    
    # Usability indicators
    if any(kw in text_lower for kw in [
        'ui', 'display', 'accessibility', 'navigation',
        'usability', 'interface', 'responsive', 'user experience',
        'feedback', 'modal'
    ]):
        return "usability"
    
    # Reliability indicators
    if any(kw in text_lower for kw in [
        'uptime', 'failure', 'recovery', 'backup',
        'reliability', 'available', 'graceful', 'degradation',
        'acid', 'transaction'
    ]):
        return "reliability"
    
    # Portability indicators
    if any(kw in text_lower for kw in [
        'browser', 'os', 'platform', 'portability',
        'compatibility', 'device', 'resolution'
    ]):
        return "portability"
    
    # Check parent requirement ID pattern
    if parent_req_id.startswith("QR"):
        return "other"  # Non-functional but unclassified
    
    # Default to functional
    return "functional"


# ======================
# Confidence Calculation
# ======================

def calculate_confidence(
    actor: str,
    action: str,
    constraint: Optional[str],
    outcome: Optional[str]
) -> str:
    """Calculate CRU confidence.
    
    Rules:
    - high: valid actor + valid action + (constraint OR outcome)
    - medium: valid action only
    - low: everything else
    
    Returns: 'high' | 'medium' | 'low'
    """
    has_valid_actor = actor in VALID_ACTORS
    has_valid_action = action and len(action.split()) >= 2
    has_constraint_or_outcome = constraint is not None or outcome is not None
    
    if has_valid_actor and has_valid_action and has_constraint_or_outcome:
        return "high"
    
    if has_valid_action:
        return "medium"
    
    return "low"


# ======================
# Deduplication
# ======================

def create_dedup_key(actor: str, action: str, constraint: Optional[str]) -> str:
    """Create deduplication key from CRU components.
    
    Returns: Normalized key for deduplication
    """
    key_parts = [
        actor.lower(),
        action.lower(),
        (constraint or "").lower()
    ]
    return "|".join(key_parts)


def deduplicate_crus(crus: List[CanonicalRequirementUnit]) -> List[CanonicalRequirementUnit]:
    """Deduplicate CRUs using (actor, action, constraint) key.
    
    Returns: Deduplicated list of CRUs
    """
    seen = {}
    deduplicated = []
    
    for cru in crus:
        key = create_dedup_key(cru.actor, cru.action, cru.constraint)
        
        if key not in seen:
            seen[key] = cru
            deduplicated.append(cru)
        # If duplicate, keep first occurrence (earlier traceability)
    
    return deduplicated


# ======================
# Main Validation Pipeline
# ======================

def validate_and_finalize_cra(
    cra: Dict[str, Any], 
    cru_counter: int,
    action_anchor: Optional[str]
) -> List[CanonicalRequirementUnit]:
    """Validate and finalize a single CRA into CRU(s) using action anchor.
    
    May produce:
    - 0 CRUs (if action invalid and no anchor)
    - 1 CRU (normal case)
    - N CRUs (if multiple constraints split)
    
    Args:
        cra: Candidate Requirement Assembly
        cru_counter: Counter for CRU ID generation
        action_anchor: Canonical action for this requirement (from anchoring step)
    
    Returns: List of CRUs
    """
    parent_req_id = cra.get("parent_requirement_id", "UNKNOWN")
    cra_id = cra.get("cra_id", "UNKNOWN")
    traceability = cra.get("traceability", {})
    
    # Extract candidates
    candidate_actor = cra.get("candidate_actor")
    candidate_action = cra.get("candidate_action")
    candidate_constraint = cra.get("candidate_constraint")
    candidate_outcome = cra.get("candidate_outcome")
    
    # 1. Normalize actor (always succeeds)
    actor = normalize_actor(candidate_actor)
    
    # 2. Determine action using anchoring + grammar enforcement
    action = None
    
    # Try to normalize the candidate action
    normalized_action = normalize_action(candidate_action)
    
    if normalized_action:
        # Check grammar
        is_valid, _ = validate_action_grammar(normalized_action)
        if is_valid:
            action = normalized_action
        elif action_anchor:
            # Grammar failed, use anchor
            action = action_anchor
    elif action_anchor:
        # Normalization failed, use anchor
        action = action_anchor
    
    # If still no valid action, DISCARD
    if not action:
        return []
    
    # 3. Validate outcome
    outcome = validate_outcome(candidate_outcome)
    
    # 4. Split constraints
    constraints = split_constraints(candidate_constraint)
    
    # 5. Create CRU(s)
    crus = []
    
    if constraints:
        # Create one CRU per constraint
        for idx, constraint in enumerate(constraints):
            cru_type = infer_cru_type(parent_req_id, action, constraint)
            confidence = calculate_confidence(actor, action, constraint, outcome)
            
            cru = CanonicalRequirementUnit(
                cru_id=f"CRU_{parent_req_id}_{cru_counter + idx:02d}",
                parent_requirement_id=parent_req_id,
                actor=actor,
                action=action,
                constraint=constraint,
                outcome=outcome,
                type=cru_type,
                confidence=confidence,
                traceability=traceability,
                derived_from_cra=cra_id
            )
            crus.append(cru)
    else:
        # Single CRU without constraint
        cru_type = infer_cru_type(parent_req_id, action, None)
        confidence = calculate_confidence(actor, action, None, outcome)
        
        cru = CanonicalRequirementUnit(
            cru_id=f"CRU_{parent_req_id}_{cru_counter:02d}",
            parent_requirement_id=parent_req_id,
            actor=actor,
            action=action,
            constraint=None,
            outcome=outcome,
            type=cru_type,
            confidence=confidence,
            traceability=traceability,
            derived_from_cra=cra_id
        )
        crus.append(cru)
    
    return crus


def finalize_crus(cras: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Main pipeline: Validate and finalize all CRAs into CRUs with action anchoring.
    
    Pipeline:
    1. Create action anchors for each requirement
    2. Validate and finalize each CRA using anchors
    3. Deduplicate CRUs
    
    Args:
        cras: List of Candidate Requirement Assemblies
        
    Returns:
        Dictionary with metadata and final CRUs
    """
    # Step 1: Create action anchors for all requirements
    print("🎯 Creating action anchors...")
    action_anchors = create_action_anchors(cras)
    
    # Report anchors
    print("\n📍 Action Anchors Selected:")
    for req_id, anchor in sorted(action_anchors.items()):
        if anchor:
            print(f"   {req_id}: \"{anchor}\"")
        else:
            print(f"   {req_id}: [NO VALID ANCHOR - CRAs may be discarded]")
    
    all_crus = []
    cru_counter_by_req = defaultdict(int)
    
    discarded_count = 0
    discarded_no_anchor = 0
    discarded_grammar_failure = 0
    
    # Step 2: Process each CRA with anchor
    for cra in cras:
        parent_req_id = cra.get("parent_requirement_id", "UNKNOWN")
        counter = cru_counter_by_req[parent_req_id]
        
        # Get anchor for this requirement
        anchor = action_anchors.get(parent_req_id)
        
        crus = validate_and_finalize_cra(cra, counter + 1, anchor)
        
        if not crus:
            discarded_count += 1
            if not anchor:
                discarded_no_anchor += 1
            else:
                discarded_grammar_failure += 1
        else:
            all_crus.extend(crus)
            cru_counter_by_req[parent_req_id] += len(crus)
    
    # Step 3: Deduplicate
    before_dedup = len(all_crus)
    all_crus = deduplicate_crus(all_crus)
    duplicates_removed = before_dedup - len(all_crus)
    
    # Compile statistics
    type_counts = defaultdict(int)
    confidence_counts = defaultdict(int)
    
    for cru in all_crus:
        type_counts[cru.type] += 1
        confidence_counts[cru.confidence] += 1
    
    # Count CRUs with constraints/outcomes
    with_constraint = sum(1 for cru in all_crus if cru.constraint)
    with_outcome = sum(1 for cru in all_crus if cru.outcome)
    
    # Count anchored CRUs
    crus_by_req = defaultdict(list)
    for cru in all_crus:
        crus_by_req[cru.parent_requirement_id].append(cru)
    
    anchored_requirements = sum(1 for req_id, crus_list in crus_by_req.items() 
                                if len(crus_list) > 0 and action_anchors.get(req_id))
    
    output = {
        "metadata": {
            "total_crus": len(all_crus),
            "total_cras_processed": len(cras),
            "cras_discarded": discarded_count,
            "cras_discarded_no_anchor": discarded_no_anchor,
            "cras_discarded_grammar_failure": discarded_grammar_failure,
            "duplicates_removed": duplicates_removed,
            "requirements_with_anchors": len([a for a in action_anchors.values() if a]),
            "requirements_anchored": anchored_requirements,
            "crus_with_constraints": with_constraint,
            "crus_with_outcomes": with_outcome,
            "type_distribution": dict(type_counts),
            "confidence_distribution": dict(confidence_counts),
            "action_anchors": {k: v for k, v in action_anchors.items() if v},
            "finalization_version": "2.0"
        },
        "crus": [cru.to_dict() for cru in all_crus]
    }
    
    return output


# ======================
# CLI Entry Point
# ======================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finalize CRAs into validated CRUs")
    parser.add_argument("--input", required=True, help="Input JSON file (candidate_requirement_assemblies.json)")
    parser.add_argument("--output", required=True, help="Output JSON file (cru_units.json)")
    args = parser.parse_args()
    
    # Load input
    print(f"📥 Loading CRAs from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cras = data.get("candidate_requirement_assemblies", [])
    print(f"✅ Loaded {len(cras)} candidate requirement assemblies")
    
    # Finalize CRUs
    print("🔍 Validating and finalizing CRUs...")
    output = finalize_crus(cras)
    
    # Save output
    print(f"💾 Saving CRUs to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 FINALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total CRUs:           {output['metadata']['total_crus']}")
    print(f"CRAs Processed:       {output['metadata']['total_cras_processed']}")
    print(f"CRAs Discarded:       {output['metadata']['cras_discarded']}")
    print(f"  - No anchor:        {output['metadata']['cras_discarded_no_anchor']}")
    print(f"  - Grammar failure:  {output['metadata']['cras_discarded_grammar_failure']}")
    print(f"Duplicates Removed:   {output['metadata']['duplicates_removed']}")
    print(f"Requirements with Anchors: {output['metadata']['requirements_with_anchors']}")
    print(f"CRUs with Constraints: {output['metadata']['crus_with_constraints']}")
    print(f"CRUs with Outcomes:   {output['metadata']['crus_with_outcomes']}")
    
    print(f"\nType Distribution:")
    for cru_type, count in sorted(output['metadata']['type_distribution'].items()):
        print(f"  {cru_type.title():15s}  {count}")
    
    print(f"\nConfidence Distribution:")
    for conf, count in sorted(output['metadata']['confidence_distribution'].items()):
        print(f"  {conf.title():10s}  {count}")
    
    print(f"{'='*60}")
    print("✅ CRU finalization complete!")