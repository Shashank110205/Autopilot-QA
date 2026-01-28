"""
Candidate Requirement Assembly - Module 2, Phase B
Signal Composition & Semantic Assembly

Purpose:
- Compose canonical signals into semantic units
- Group signals by linguistic proximity
- Create Candidate Requirement Assemblies (CRAs)
- Do NOT validate correctness (that's Module 3)

This module ASSEMBLES, it does not DECIDE.
Output contains intentionally imperfect candidates.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict


# ======================
# Data Structures
# ======================

@dataclass
class CandidateRequirementAssembly:
    """Represents a candidate requirement assembled from multiple signals."""
    cra_id: str
    parent_requirement_id: str
    assembled_from_signals: List[str]
    candidate_actor: Optional[str]
    candidate_action: Optional[str]
    candidate_constraint: Optional[str]
    candidate_outcome: Optional[str]
    assembly_confidence: str  # low | medium | high
    traceability: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ======================
# Signal Grouping
# ======================

def group_signals_by_requirement(signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group signals by parent requirement ID.
    
    Returns: Dict mapping requirement_id -> list of signals
    """
    grouped = defaultdict(list)
    
    for signal in signals:
        req_id = signal.get("parent_requirement_id", "UNKNOWN")
        grouped[req_id].append(signal)
    
    return dict(grouped)


def group_signals_by_field(signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group signals by source field within a requirement.
    
    Returns: Dict mapping source_field -> list of signals
    """
    grouped = defaultdict(list)
    
    for signal in signals:
        field = signal.get("source_field", "unknown")
        grouped[field].append(signal)
    
    return dict(grouped)


# ======================
# Signal Classification
# ======================

def is_potential_actor(signal: Dict[str, Any]) -> bool:
    """Check if signal could be an actor based on linguistic features."""
    sig_type = signal.get("signal_type", "")
    
    if sig_type == "actor":
        return True
    
    # Check linguistic features
    features = signal.get("linguistic_features", {})
    dep_roles = features.get("dependency_roles", [])
    pos_tags = features.get("pos_tags", [])
    
    # Subject with noun
    if "nsubj" in dep_roles or "nsubjpass" in dep_roles:
        if "NOUN" in pos_tags or "PROPN" in pos_tags:
            return True
    
    return False


def is_potential_action(signal: Dict[str, Any]) -> bool:
    """Check if signal could be an action based on linguistic features."""
    sig_type = signal.get("signal_type", "")
    
    if sig_type == "action":
        return True
    
    # Check linguistic features
    features = signal.get("linguistic_features", {})
    dep_roles = features.get("dependency_roles", [])
    pos_tags = features.get("pos_tags", [])
    
    # ROOT verb
    if "ROOT" in dep_roles and "VERB" in pos_tags:
        return True
    
    return False


def is_potential_constraint(signal: Dict[str, Any]) -> bool:
    """Check if signal could be a constraint."""
    return signal.get("signal_type", "") == "constraint"


def is_potential_outcome(signal: Dict[str, Any]) -> bool:
    """Check if signal could be an outcome."""
    return signal.get("signal_type", "") == "outcome"


# ======================
# Proximity Detection
# ======================

def signals_are_proximate(signal1: Dict[str, Any], signal2: Dict[str, Any]) -> bool:
    """Determine if two signals are linguistically proximate.
    
    Signals are proximate if they:
    - Share the same source_field
    - OR are in description + system_behavior (adjacent fields)
    - OR are in description + outputs (related fields)
    """
    field1 = signal1.get("source_field", "")
    field2 = signal2.get("source_field", "")
    
    # Same field = proximate
    if field1 == field2:
        return True
    
    # Adjacent or related fields
    adjacent_pairs = [
        ("description", "system_behavior"),
        ("description", "outputs"),
        ("system_behavior", "outputs"),
        ("description", "inputs"),
    ]
    
    for f1, f2 in adjacent_pairs:
        if (field1 == f1 and field2 == f2) or (field1 == f2 and field2 == f1):
            return True
    
    return False


# ======================
# Assembly Logic
# ======================

def find_best_actor(signals: List[Dict[str, Any]]) -> Optional[str]:
    """Find the best candidate actor from a list of signals.
    
    Priority:
    1. Signals marked as actor type
    2. Signals with nsubj dependency
    3. First noun phrase
    4. None
    """
    # Priority 1: Actor signals
    actor_signals = [s for s in signals if is_potential_actor(s)]
    if actor_signals:
        # Prefer signals with nsubj
        for sig in actor_signals:
            dep_roles = sig.get("linguistic_features", {}).get("dependency_roles", [])
            if "nsubj" in dep_roles or "nsubjpass" in dep_roles:
                return sig.get("text_span")
        # Fallback to first actor
        return actor_signals[0].get("text_span")
    
    # Priority 2: Noun phrases near action
    noun_signals = [s for s in signals 
                   if "NOUN" in s.get("linguistic_features", {}).get("pos_tags", []) or
                      "PROPN" in s.get("linguistic_features", {}).get("pos_tags", [])]
    if noun_signals:
        return noun_signals[0].get("text_span")
    
    return None


def find_best_action(signals: List[Dict[str, Any]]) -> Optional[str]:
    """Find the best candidate action from a list of signals.
    
    Priority:
    1. Signals marked as action type
    2. Signals with ROOT verb
    3. Verb phrases with modal removed
    4. None
    """
    # Priority 1: Action signals
    action_signals = [s for s in signals if is_potential_action(s)]
    if action_signals:
        # Prefer ROOT verbs without modals
        for sig in action_signals:
            dep_roles = sig.get("linguistic_features", {}).get("dependency_roles", [])
            contains_modal = sig.get("linguistic_features", {}).get("contains_modal", False)
            if "ROOT" in dep_roles and not contains_modal:
                return sig.get("text_span")
        # Fallback to first action
        return action_signals[0].get("text_span")
    
    # Priority 2: ROOT verbs
    verb_signals = [s for s in signals 
                   if "ROOT" in s.get("linguistic_features", {}).get("dependency_roles", [])
                   and "VERB" in s.get("linguistic_features", {}).get("pos_tags", [])]
    if verb_signals:
        text = verb_signals[0].get("text_span", "")
        # Remove modal verbs
        modals = ["shall", "must", "should", "will", "may", "would", "could", "can"]
        for modal in modals:
            text = text.replace(f"{modal} ", "")
        return text.strip() if text.strip() else None
    
    return None


def collect_constraints(signals: List[Dict[str, Any]]) -> Optional[str]:
    """Collect all constraint signals and concatenate them.
    
    Returns: Concatenated constraint text or None
    """
    constraint_signals = [s for s in signals if is_potential_constraint(s)]
    
    if not constraint_signals:
        return None
    
    # Concatenate unique constraints
    constraints = []
    seen = set()
    
    for sig in constraint_signals:
        text = sig.get("text_span", "").strip()
        if text and text not in seen:
            constraints.append(text)
            seen.add(text)
    
    return " | ".join(constraints) if constraints else None


def collect_outcomes(signals: List[Dict[str, Any]]) -> Optional[str]:
    """Collect all outcome signals and concatenate them.
    
    Returns: Concatenated outcome text or None
    """
    outcome_signals = [s for s in signals if is_potential_outcome(s)]
    
    if not outcome_signals:
        return None
    
    # Concatenate unique outcomes
    outcomes = []
    seen = set()
    
    for sig in outcome_signals:
        text = sig.get("text_span", "").strip()
        if text and text not in seen:
            outcomes.append(text)
            seen.add(text)
    
    return " | ".join(outcomes) if outcomes else None


def compute_confidence(
    candidate_actor: Optional[str],
    candidate_action: Optional[str],
    candidate_constraint: Optional[str],
    candidate_outcome: Optional[str]
) -> str:
    """Compute assembly confidence based on completeness.
    
    Returns: 'high' | 'medium' | 'low'
    """
    # High: actor + action present
    if candidate_actor and candidate_action:
        return "high"
    
    # Medium: action present, actor missing
    if candidate_action:
        return "medium"
    
    # Low: only fragments (constraint or outcome without action)
    return "low"


# ======================
# Main Assembly Pipeline
# ======================

def assemble_signals_for_field(
    signals: List[Dict[str, Any]],
    parent_req_id: str,
    field_name: str,
    traceability: Dict[str, Any]
) -> List[CandidateRequirementAssembly]:
    """Assemble signals from a single field into CRAs.
    
    Strategy: Group proximate signals into assemblies
    """
    if len(signals) < 2:
        return []
    
    assemblies = []
    
    # Strategy 1: If we have clear actor + action in same field, create assembly
    actors = [s for s in signals if is_potential_actor(s)]
    actions = [s for s in signals if is_potential_action(s)]
    
    if actions:
        # Create assembly for each action with related signals
        for action_sig in actions:
            # Collect all signals in this field that could relate to this action
            related_signals = [action_sig]
            
            # Add actor if present
            if actors:
                related_signals.append(actors[0])
            
            # Add constraints
            constraints = [s for s in signals if is_potential_constraint(s)]
            related_signals.extend(constraints)
            
            # Add outcomes
            outcomes = [s for s in signals if is_potential_outcome(s)]
            related_signals.extend(outcomes)
            
            # Build CRA
            cra = create_assembly(
                signals=related_signals,
                parent_req_id=parent_req_id,
                traceability=traceability,
                cra_suffix=f"{field_name}_{len(assemblies)+1}"
            )
            
            if cra:
                assemblies.append(cra)
    
    # Strategy 2: If we have constraints without actions, group them separately
    lone_constraints = [s for s in signals if is_potential_constraint(s)]
    if lone_constraints and not actions:
        cra = create_assembly(
            signals=lone_constraints,
            parent_req_id=parent_req_id,
            traceability=traceability,
            cra_suffix=f"{field_name}_constraints"
        )
        if cra:
            assemblies.append(cra)
    
    return assemblies


def create_assembly(
    signals: List[Dict[str, Any]],
    parent_req_id: str,
    traceability: Dict[str, Any],
    cra_suffix: str
) -> Optional[CandidateRequirementAssembly]:
    """Create a single CRA from a list of signals.
    
    Returns: CandidateRequirementAssembly or None if invalid
    """
    # Validation: Must have ≥2 signals
    if len(signals) < 2:
        return None
    
    # Extract components
    candidate_actor = find_best_actor(signals)
    candidate_action = find_best_action(signals)
    candidate_constraint = collect_constraints(signals)
    candidate_outcome = collect_outcomes(signals)
    
    # Validation: Must have at least action OR constraint
    if not candidate_action and not candidate_constraint:
        return None
    
    # Compute confidence
    confidence = compute_confidence(
        candidate_actor,
        candidate_action,
        candidate_constraint,
        candidate_outcome
    )
    
    # Build CRA
    cra = CandidateRequirementAssembly(
        cra_id=f"CRA_{parent_req_id}_{cra_suffix}",
        parent_requirement_id=parent_req_id,
        assembled_from_signals=[s.get("signal_id") for s in signals],
        candidate_actor=candidate_actor,
        candidate_action=candidate_action,
        candidate_constraint=candidate_constraint,
        candidate_outcome=candidate_outcome,
        assembly_confidence=confidence,
        traceability=traceability
    )
    
    return cra


def assemble_requirement_signals(
    req_signals: List[Dict[str, Any]],
    parent_req_id: str
) -> List[CandidateRequirementAssembly]:
    """Assemble all signals for a single requirement into CRAs.
    
    Args:
        req_signals: All signals for one requirement
        parent_req_id: Requirement ID
        
    Returns:
        List of CandidateRequirementAssembly objects
    """
    if not req_signals:
        return []
    
    # Get traceability from first signal
    traceability = req_signals[0].get("traceability", {})
    
    # Group by field
    field_groups = group_signals_by_field(req_signals)
    
    all_assemblies = []
    
    # Assemble signals from each field
    for field_name, field_signals in field_groups.items():
        assemblies = assemble_signals_for_field(
            signals=field_signals,
            parent_req_id=parent_req_id,
            field_name=field_name,
            traceability=traceability
        )
        all_assemblies.extend(assemblies)
    
    # Strategy 3: Cross-field assemblies (description + outputs)
    if "description" in field_groups and "outputs" in field_groups:
        desc_actions = [s for s in field_groups["description"] if is_potential_action(s)]
        output_outcomes = [s for s in field_groups["outputs"] if is_potential_outcome(s)]
        
        if desc_actions and output_outcomes:
            # Create cross-field assembly
            cross_signals = [desc_actions[0]] + output_outcomes
            cra = create_assembly(
                signals=cross_signals,
                parent_req_id=parent_req_id,
                traceability=traceability,
                cra_suffix="cross_field"
            )
            if cra:
                all_assemblies.append(cra)
    
    return all_assemblies


# ======================
# Main Pipeline
# ======================

def assemble_candidate_requirements(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Main pipeline: Assemble signals into CRAs.
    
    Args:
        signals: List of canonical signals from Module 1
        
    Returns:
        Dictionary with metadata and CRAs
    """
    # Group signals by requirement
    req_groups = group_signals_by_requirement(signals)
    
    all_cras = []
    
    # Process each requirement
    for req_id, req_signals in req_groups.items():
        cras = assemble_requirement_signals(req_signals, req_id)
        all_cras.extend(cras)
    
    # Compile statistics
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    for cra in all_cras:
        conf = cra.assembly_confidence
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
    
    # Count completeness
    complete_cras = sum(1 for cra in all_cras 
                       if cra.candidate_actor and cra.candidate_action)
    incomplete_cras = len(all_cras) - complete_cras
    
    output = {
        "metadata": {
            "total_cras": len(all_cras),
            "total_requirements": len(req_groups),
            "complete_assemblies": complete_cras,
            "incomplete_assemblies": incomplete_cras,
            "confidence_distribution": confidence_counts,
            "assembly_version": "1.0"
        },
        "candidate_requirement_assemblies": [cra.to_dict() for cra in all_cras]
    }
    
    return output


# ======================
# CLI Entry Point
# ======================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Assemble canonical signals into candidate requirements")
    parser.add_argument("--input", required=True, help="Input JSON file (canonical_signals.json)")
    parser.add_argument("--output", required=True, help="Output JSON file (candidate_requirement_assemblies.json)")
    args = parser.parse_args()
    
    # Load input
    print(f"📥 Loading signals from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    signals = data.get("signals", [])
    print(f"✅ Loaded {len(signals)} signals")
    
    # Assemble CRAs
    print("🔧 Assembling candidate requirements...")
    output = assemble_candidate_requirements(signals)
    
    # Save output
    print(f"💾 Saving assemblies to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 ASSEMBLY SUMMARY")
    print(f"{'='*60}")
    print(f"Total CRAs:            {output['metadata']['total_cras']}")
    print(f"Source Requirements:   {output['metadata']['total_requirements']}")
    print(f"Complete Assemblies:   {output['metadata']['complete_assemblies']}")
    print(f"Incomplete Assemblies: {output['metadata']['incomplete_assemblies']}")
    print(f"\nConfidence Distribution:")
    for conf, count in output['metadata']['confidence_distribution'].items():
        print(f"  {conf.title():10s}  {count}")
    print(f"{'='*60}")
    print("✅ Assembly complete!")