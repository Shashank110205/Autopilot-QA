"""
Canonical Signal Extraction - Module 1, Phase B
Hybrid Parsing & Linguistic Feature Extraction

Purpose:
- Parse extracted requirement text into atomic linguistic signals
- Extract potential actors, actions, constraints, outcomes
- Preserve traceability and reversibility
- Operate domain-agnostically using grammatical structure

This is a PRE-normalization step. Outputs are CANDIDATES, not final CRUs.
"""

import json
import re
import spacy
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

# ======================
# Configuration
# ======================

# Valid signal types
SIGNAL_TYPES = {"actor", "action", "constraint", "outcome", "other"}

# Load spaCy model
print("⚙️ Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Attempting to download it now...")
    import subprocess, sys
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print("Automatic download failed:", e)
        print("Please install the model manually by running:")
        print("  python -m spacy download en_core_web_sm")
        print("or, if preferred, install the pip package:")
        print("  pip install en-core-web-sm")
        sys.exit(1)


# ======================
# Data Structures
# ======================

@dataclass
class CanonicalSignal:
    """Represents a single linguistic signal extracted from requirement text."""
    signal_id: str
    parent_requirement_id: str
    signal_type: str  # actor | action | constraint | outcome | other
    text_span: str
    source_field: str
    linguistic_features: Dict[str, Any]
    traceability: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ======================
# Core Signal Extractors
# ======================

def extract_noun_phrases(doc: spacy.tokens.Doc) -> List[tuple]:
    """Extract noun phrases as potential actors.
    Returns: List of (text_span, pos_tags, dep_roles)"""
    signals = []
    
    for chunk in doc.noun_chunks:
        # Get POS tags for the chunk
        pos_tags = [token.pos_ for token in chunk]
        
        # Get dependency roles
        dep_roles = [token.dep_ for token in chunk]
        
        # Only extract if meaningful (3+ tokens or contains NOUN/PROPN)
        if len(chunk.text.split()) >= 2 or any(token.pos_ in ('NOUN', 'PROPN') for token in chunk):
            signals.append((
                chunk.text.strip(),
                pos_tags,
                dep_roles
            ))
    
    return signals


def extract_verb_phrases(doc: spacy.tokens.Doc) -> List[tuple]:
    """Extract verb phrases as potential actions.
    Returns: List of (text_span, pos_tags, dep_roles, contains_modal)"""
    signals = []
    
    modals = {'shall', 'must', 'should', 'will', 'may', 'would', 'could', 'can'}
    
    for token in doc:
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            # Build verb phrase from ROOT verb and its children
            verb_components = [token]
            
            for child in token.children:
                if child.dep_ in ('dobj', 'prep', 'pobj', 'advmod', 'acomp', 'xcomp', 'aux', 'prt', 'neg'):
                    verb_components.append(child)
            
            # Sort by position in sentence
            verb_components.sort(key=lambda t: t.i)
            
            text_span = ' '.join([t.text for t in verb_components])
            pos_tags = [t.pos_ for t in verb_components]
            dep_roles = [t.dep_ for t in verb_components]
            contains_modal = any(t.text.lower() in modals for t in verb_components)
            
            # Only extract if meaningful
            if len(text_span.split()) >= 2:
                signals.append((
                    text_span.strip(),
                    pos_tags,
                    dep_roles,
                    contains_modal
                ))
    
    return signals


def extract_metric_expressions(text: str) -> List[str]:
    """Extract numeric/metric expressions as potential constraints.
    Returns: List of text spans"""
    signals = []
    
    # Patterns for quantifiable metrics
    patterns = [
        r'\b\d+(?:\.\d+)?\s*(?:%|percent|ms|milliseconds?|seconds?|minutes?|hours?|users?|concurrent|requests?)\b[^.;]*',
        r'\b(?:within|under|below|above|at least|at most|maximum|minimum|up to|no more than)\s+\d+[^.;]*',
        r'\b(?:99(?:\.\d+)?(?:th)?\s*percentile)[^.;]*',
        r'\b(?:uptime|availability|latency|throughput|response\s+time)[^.;]*',
        r'\b(?:timeout|time\s+limit)[^.;]*',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            span = match.group(0).strip()
            if len(span.split()) >= 3:  # Minimum 3 tokens
                signals.append(span)
    
    return signals


def extract_grammatical_subjects(doc: spacy.tokens.Doc) -> List[tuple]:
    """Extract grammatical subjects as potential actors.
    Returns: List of (text_span, pos_tags, dep_roles)"""
    signals = []
    
    reference_pronouns = {'this', 'that', 'it', 'these', 'those'}
    
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass'):
            # Build noun phrase around subject
            subject_tokens = [token]
            
            # Add modifiers
            for child in token.children:
                if child.dep_ in ('compound', 'amod', 'det', 'poss'):
                    subject_tokens.append(child)
            
            # Sort by position
            subject_tokens.sort(key=lambda t: t.i)
            
            text_span = ' '.join([t.text for t in subject_tokens])
            pos_tags = [t.pos_ for t in subject_tokens]
            dep_roles = [t.dep_ for t in subject_tokens]
            
            # Filter out reference pronouns
            if token.text.lower() not in reference_pronouns:
                if token.pos_ in ('NOUN', 'PROPN'):
                    signals.append((
                        text_span.strip(),
                        pos_tags,
                        dep_roles
                    ))
    
    return signals


def extract_outcome_patterns(text: str, doc: spacy.tokens.Doc) -> List[str]:
    """Extract observable outcomes from text.
    Returns: List of text spans"""
    signals = []
    
    # Pattern-based extraction
    patterns = [
        r'(?:results? in|leads to|produces|generates|creates?|returns?)\s+([^.;]+)',
        r'(?:displays?|shows?|renders?|presents?)\s+([^.;]+)',
        r'(?:outputs?|responses?)\s*:\s*([^.;]+)',
        r'(?:confirmation|message|notification|error)\s*:\s*["\']([^"\']+)["\']',
        r'(?:confirmation|message|notification)\s+([^.;]+)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            span = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
            if len(span.split()) >= 3:
                signals.append(span)
    
    return signals


def extract_constraint_patterns(text: str) -> List[str]:
    """Extract qualitative constraints from text.
    Returns: List of text spans"""
    signals = []
    
    patterns = [
        r'\b(?:must not|shall not|cannot|should not)\s+([^.;]+)',
        r'\b(?:only|exclusively|solely)\s+([^.;]+)',
        r'\b(?:restricted to|limited to|scoped to)\s+([^.;]+)',
        r'\b(?:without|except|unless)\s+([^.;]+)',
        r'\b(?:HTTPS|SSL|TLS|encrypted|hashed)\b[^.;]*',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            span = match.group(0).strip()
            if len(span.split()) >= 3:
                signals.append(span)
    
    return signals


# ======================
# Signal Classification
# ======================

def classify_signal_type(text_span: str, pos_tags: List[str], dep_roles: List[str]) -> str:
    """Classify a text span into signal type based on linguistic features.
    Returns: 'actor' | 'action' | 'constraint' | 'outcome' | 'other'"""
    
    # Actor heuristics
    if any(dep in dep_roles for dep in ['nsubj', 'nsubjpass']):
        if any(pos in pos_tags for pos in ['NOUN', 'PROPN']):
            return 'actor'
    
    # Action heuristics
    if 'VERB' in pos_tags and 'ROOT' in dep_roles:
        return 'action'
    
    # Constraint heuristics (numeric/metric)
    if re.search(r'\d+(?:\.\d+)?\s*(?:%|ms|seconds?|users?)', text_span, re.IGNORECASE):
        return 'constraint'
    
    # Constraint heuristics (qualitative)
    if any(kw in text_span.lower() for kw in ['must not', 'shall not', 'only', 'restricted']):
        return 'constraint'
    
    # Outcome heuristics
    if any(kw in text_span.lower() for kw in ['result', 'display', 'output', 'message', 'confirmation']):
        return 'outcome'
    
    return 'other'


# ======================
# Main Extraction Pipeline
# ======================

def extract_signals_from_text(
    text: str,
    parent_req_id: str,
    source_field: str,
    traceability: Dict[str, Any],
    signal_counter: int
) -> List[CanonicalSignal]:
    """Extract all signals from a text field.
    
    Args:
        text: Raw text from requirement field
        parent_req_id: Requirement ID (e.g., "FR3")
        source_field: Field name (e.g., "description")
        traceability: Traceability metadata
        signal_counter: Starting counter for signal IDs
        
    Returns:
        List of CanonicalSignal objects
    """
    if not text or len(text.strip()) < 10:
        return []
    
    signals = []
    
    # Parse with spaCy
    doc = nlp(text[:2000])  # Limit to 2000 chars for performance
    
    # Extract grammatical subjects (potential actors)
    subjects = extract_grammatical_subjects(doc)
    for text_span, pos_tags, dep_roles in subjects:
        signal_type = 'actor'
        
        signal = CanonicalSignal(
            signal_id=f"SIG_{parent_req_id}_{signal_counter:02d}",
            parent_requirement_id=parent_req_id,
            signal_type=signal_type,
            text_span=text_span,
            source_field=source_field,
            linguistic_features={
                "pos_tags": pos_tags,
                "dependency_roles": dep_roles,
                "contains_modal": False
            },
            traceability=traceability
        )
        signals.append(signal)
        signal_counter += 1
    
    # Extract verb phrases (potential actions)
    verbs = extract_verb_phrases(doc)
    for text_span, pos_tags, dep_roles, contains_modal in verbs:
        signal_type = 'action'
        
        signal = CanonicalSignal(
            signal_id=f"SIG_{parent_req_id}_{signal_counter:02d}",
            parent_requirement_id=parent_req_id,
            signal_type=signal_type,
            text_span=text_span,
            source_field=source_field,
            linguistic_features={
                "pos_tags": pos_tags,
                "dependency_roles": dep_roles,
                "contains_modal": contains_modal
            },
            traceability=traceability
        )
        signals.append(signal)
        signal_counter += 1
    
    # Extract metric expressions (potential constraints)
    metrics = extract_metric_expressions(text)
    for text_span in metrics:
        signal = CanonicalSignal(
            signal_id=f"SIG_{parent_req_id}_{signal_counter:02d}",
            parent_requirement_id=parent_req_id,
            signal_type='constraint',
            text_span=text_span,
            source_field=source_field,
            linguistic_features={
                "pos_tags": [],
                "dependency_roles": [],
                "contains_modal": False,
                "metric_type": "quantitative"
            },
            traceability=traceability
        )
        signals.append(signal)
        signal_counter += 1
    
    # Extract constraint patterns (potential constraints)
    constraints = extract_constraint_patterns(text)
    for text_span in constraints:
        signal = CanonicalSignal(
            signal_id=f"SIG_{parent_req_id}_{signal_counter:02d}",
            parent_requirement_id=parent_req_id,
            signal_type='constraint',
            text_span=text_span,
            source_field=source_field,
            linguistic_features={
                "pos_tags": [],
                "dependency_roles": [],
                "contains_modal": False,
                "metric_type": "qualitative"
            },
            traceability=traceability
        )
        signals.append(signal)
        signal_counter += 1
    
    # Extract outcome patterns
    outcomes = extract_outcome_patterns(text, doc)
    for text_span in outcomes:
        signal = CanonicalSignal(
            signal_id=f"SIG_{parent_req_id}_{signal_counter:02d}",
            parent_requirement_id=parent_req_id,
            signal_type='outcome',
            text_span=text_span,
            source_field=source_field,
            linguistic_features={
                "pos_tags": [],
                "dependency_roles": [],
                "contains_modal": False
            },
            traceability=traceability
        )
        signals.append(signal)
        signal_counter += 1
    
    # Extract noun phrases not captured as subjects
    noun_phrases = extract_noun_phrases(doc)
    for text_span, pos_tags, dep_roles in noun_phrases:
        # Skip if already captured as subject
        if any(s.text_span == text_span and s.signal_type == 'actor' for s in signals):
            continue
        
        signal_type = classify_signal_type(text_span, pos_tags, dep_roles)
        
        signal = CanonicalSignal(
            signal_id=f"SIG_{parent_req_id}_{signal_counter:02d}",
            parent_requirement_id=parent_req_id,
            signal_type=signal_type,
            text_span=text_span,
            source_field=source_field,
            linguistic_features={
                "pos_tags": pos_tags,
                "dependency_roles": dep_roles,
                "contains_modal": False
            },
            traceability=traceability
        )
        signals.append(signal)
        signal_counter += 1
    
    return signals


def extract_signals_from_requirement(req: Dict[str, Any]) -> List[CanonicalSignal]:
    """Extract all signals from a single requirement.
    
    Args:
        req: Requirement dictionary
        
    Returns:
        List of CanonicalSignal objects
    """
    req_id = req.get("id", "UNKNOWN")
    traceability = req.get("traceability", req.get("source_ref", {}))
    
    # Fields to process
    fields_to_process = [
        ("description", req.get("description", "")),
        ("system_behavior", req.get("system_behavior", "")),
        ("outputs", req.get("outputs", "")),
        ("constraints", req.get("constraints", "")),
        ("inputs", req.get("inputs", "")),
    ]
    
    all_signals = []
    signal_counter = 1
    
    for field_name, field_value in fields_to_process:
        if not field_value:
            continue
        
        # Handle both string and list fields
        if isinstance(field_value, list):
            field_text = ' '.join(str(item) for item in field_value)
        else:
            field_text = str(field_value)
        
        if len(field_text.strip()) < 10:
            continue
        
        signals = extract_signals_from_text(
            text=field_text,
            parent_req_id=req_id,
            source_field=field_name,
            traceability=traceability,
            signal_counter=signal_counter
        )
        
        all_signals.extend(signals)
        signal_counter += len(signals)
    
    return all_signals


# ======================
# Validation
# ======================

def validate_signal(signal: CanonicalSignal) -> bool:
    """Validate a signal before output.
    
    Returns: True if valid, False otherwise
    """
    # Check signal type
    if signal.signal_type not in SIGNAL_TYPES:
        return False
    
    # Check text span length (minimum 3 tokens)
    if len(signal.text_span.split()) < 3:
        return False
    
    # Check linguistic features present
    if not signal.linguistic_features:
        return False
    
    return True


# ======================
# Main Pipeline
# ======================

def extract_canonical_signals(requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Main pipeline: Extract canonical signals from all requirements.
    
    Args:
        requirements: List of requirement dictionaries
        
    Returns:
        Dictionary with metadata and signals
    """
    all_signals = []
    
    for req in requirements:
        signals = extract_signals_from_requirement(req)
        
        # Validate and add
        for signal in signals:
            if validate_signal(signal):
                all_signals.append(signal.to_dict())
            else:
                # Emit as 'other' if validation fails
                signal.signal_type = 'other'
                all_signals.append(signal.to_dict())
    
    # Compile statistics
    signal_type_counts = {}
    for signal in all_signals:
        sig_type = signal['signal_type']
        signal_type_counts[sig_type] = signal_type_counts.get(sig_type, 0) + 1
    
    output = {
        "metadata": {
            "total_signals": len(all_signals),
            "total_requirements": len(requirements),
            "signal_type_counts": signal_type_counts,
            "extraction_version": "1.0"
        },
        "signals": all_signals
    }
    
    return output


# ======================
# CLI Entry Point
# ======================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract canonical signals from requirements")
    parser.add_argument("--input", required=True, help="Input JSON file (requirements_extracted_grouped.json)")
    parser.add_argument("--output", required=True, help="Output JSON file (canonical_signals.json)")
    args = parser.parse_args()
    
    # Load input
    print(f"📥 Loading requirements from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Collect all requirements
    all_requirements = []
    
    if isinstance(data, dict):
        # Handle grouped format
        for key in ["functional_requirements", "quality_requirements", "performance_requirements", 
                    "use_cases", "constraints"]:
            all_requirements.extend(data.get(key, []))
    elif isinstance(data, list):
        # Handle flat list
        all_requirements = data
    
    print(f"✅ Loaded {len(all_requirements)} requirements")
    
    # Extract signals
    print("🔍 Extracting canonical signals...")
    output = extract_canonical_signals(all_requirements)
    
    # Save output
    print(f"💾 Saving signals to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 SIGNAL EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Signals:        {output['metadata']['total_signals']}")
    print(f"Source Requirements:  {output['metadata']['total_requirements']}")
    print(f"\nSignal Type Breakdown:")
    for sig_type, count in sorted(output['metadata']['signal_type_counts'].items()):
        print(f"  {sig_type.title():15s}  {count}")
    print(f"{'='*60}")
    print("✅ Signal extraction complete!")