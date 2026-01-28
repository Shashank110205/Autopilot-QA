import json
import re
import spacy
from typing import Dict, List, Any, Optional, Tuple
import os

# ======================
# Configuration
# ======================
INPUT_PATH = "../02_Requirement_Understanding/output/requirements_extracted_grouped.json"
OUTPUT_PATH = "../02_Requirement_Understanding/output/candidate_crus.json"

# ======================
# Load NLP
# ======================
print("⚙️ Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# ======================
# Load Requirements
# ======================
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"❌ {INPUT_PATH} not found! Run requirement_extraction.py first.")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded requirements from {INPUT_PATH}")


# ======================
# Core Extraction Functions
# ======================

def extract_grammatical_subject(text: str) -> Optional[str]:
    """Extract grammatical subject using dependency parsing.
    STRICT: Actor must be NOUN or PROPN, not verbs, adjectives, or pronouns."""
    if not text or len(text) < 5:
        return None
    
    doc = nlp(text[:300])
    
    # Invalid pronouns that should not be actors
    invalid_pronouns = {'this', 'that', 'it', 'these', 'those'}
    
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass'):
            # STRICT: Must be NOUN or PROPN
            if token.pos_ not in ('NOUN', 'PROPN'):
                continue
            
            # Reject pronouns
            if token.text.lower() in invalid_pronouns:
                continue
            
            # Get full noun phrase if available
            subject_tokens = [token]
            for child in token.children:
                if child.dep_ in ('compound', 'amod', 'det'):
                    subject_tokens.append(child)
            
            subject = ' '.join(sorted([t.text for t in subject_tokens], 
                                     key=lambda x: [t.i for t in subject_tokens if t.text == x][0]))
            return subject.strip()
    
    return None


def extract_main_verb_phrase(text: str) -> Optional[str]:
    """Extract main verb phrase using dependency parsing.
    Removes modal/auxiliary verbs (shall, must, should, will, may)."""
    if not text or len(text) < 5:
        return None
    
    doc = nlp(text[:300])
    
    # Modal and auxiliary verbs to exclude
    modals_to_remove = {'shall', 'must', 'should', 'will', 'may', 'would', 'could', 'can'}
    
    for token in doc:
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            verb_components = [token.text]
            
            for child in token.children:
                # Skip modal/auxiliary verbs
                if child.text.lower() in modals_to_remove:
                    continue
                
                if child.dep_ in ('dobj', 'prep', 'pobj', 'advmod', 'acomp', 'xcomp', 'neg'):
                    verb_components.append(child.text)
                elif child.dep_ == 'prt':  # Particle (e.g., "log in")
                    verb_components.append(child.text)
                elif child.dep_ == 'aux' and child.text.lower() not in modals_to_remove:
                    # Keep non-modal auxiliaries like "be", "have"
                    verb_components.append(child.text)
            
            return ' '.join(verb_components).strip()
    
    return None


def detect_constraint_or_metric(text: str) -> List[str]:
    """Detect constraints, metrics, or bounds using pattern matching.
    Returns LIST of constraints when multiple independent ones exist."""
    if not text:
        return []
    
    constraints_found = []
    
    # Patterns for quantifiable constraints
    metric_patterns = [
        r'\b\d+(?:\.\d+)?\s*(?:%|percent|ms|seconds?|minutes?|hours?|users?|concurrent|requests?)\b',
        r'\b(?:within|under|below|above|at least|at most|maximum|minimum|up to)\s+\d+[^.;,]*',
        r'\b(?:99(?:\.\d+)?%)\s*\w*',
        r'\b(?:timeout|latency|response time|uptime|availability)\s+[^.;,]*',
    ]
    
    for pattern in metric_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract surrounding context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()
            
            # Clean up partial sentences
            context = re.sub(r'^\W+', '', context)
            context = re.sub(r'\W+$', '', context)
            
            if len(context) > 10 and context not in constraints_found:
                constraints_found.append(context)
    
    # Patterns for qualitative constraints
    constraint_patterns = [
        (r'\b(?:must not|shall not|cannot|should not)\s+([^.;,]+)', 'negative'),
        (r'\b(?:only|exclusively|solely)\s+([^.;,]+)', 'restriction'),
        (r'\b(?:restricted to|limited to|scoped to)\s+([^.;,]+)', 'scope'),
        (r'\bHTTPS\s+enforcement\b', 'security'),
        (r'\bwith\s+([A-Z][^.;,]*(?:enforcement|requirement|constraint))', 'additional'),
    ]
    
    for pattern, constraint_type in constraint_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            constraint_text = match.group(0).strip()
            if len(constraint_text) > 5 and constraint_text not in constraints_found:
                constraints_found.append(constraint_text)
    
    return constraints_found


def detect_observable_outcome(text: str) -> Optional[str]:
    """Detect observable outcomes or system responses."""
    if not text:
        return None
    
    outcome_patterns = [
        r'(?:result(?:s)? in|leads to|produces|generates|creates|returns?)\s+([^.;]+)',
        r'(?:display|show|render|present)(?:s|ing)?\s+([^.;]+)',
        r'(?:output|response)(?:\s+is)?\s*:\s*([^.;]+)',
        r'(?:confirmation|message|notification)(?:\s+is)?\s*:\s*([^.;]+)',
    ]
    
    for pattern in outcome_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            outcome = match.group(1).strip()
            if len(outcome) > 5:
                return outcome
    
    return None


def is_explanatory_sentence(segment: str) -> bool:
    """Detect explanatory or meta-sentences that should not become standalone CRUs.
    These typically start with pronouns referring to prior actions."""
    if not segment or len(segment) < 5:
        return False
    
    # Get first few words
    first_words = segment.strip().split()[:3]
    if not first_words:
        return False
    
    first_word = first_words[0].lower()
    
    # Reject sentences starting with reference pronouns
    explanatory_starters = {'this', 'that', 'it', 'these', 'those'}
    
    if first_word in explanatory_starters:
        # Check if it's truly explanatory (not a new action)
        # If the sentence has verbs like "establishes", "ensures", "provides" after the pronoun,
        # it's likely explaining a consequence, not defining a new requirement
        explanatory_verbs = ['establish', 'ensure', 'provide', 'result', 'lead', 
                            'cause', 'mean', 'indicate', 'show', 'demonstrate']
        
        segment_lower = segment.lower()
        if any(verb in segment_lower for verb in explanatory_verbs):
            return True
    
    return False


def split_compound_sentence(text: str) -> List[str]:
    """Split compound sentences based on coordination and subordination."""
    if not text or len(text) < 10:
        return []
    
    # Split on coordinating conjunctions that indicate separate behaviors
    split_patterns = [
        r'\.\s+',  # Period separation
        r';\s+',   # Semicolon separation
    ]
    
    segments = [text]
    for pattern in split_patterns:
        new_segments = []
        for seg in segments:
            new_segments.extend(re.split(pattern, seg))
        segments = new_segments
    
    # Clean and filter - remove explanatory sentences
    segments = [s.strip() for s in segments 
                if len(s.strip()) > 10 and not is_explanatory_sentence(s)]
    
    return segments if segments else []


def detect_multiple_actions(text: str) -> List[Tuple[str, str]]:
    """Detect multiple distinct actions in a single text span.
    Returns list of (actor, action) tuples.
    STRICT: Only returns valid actors (NOUN/PROPN, not pronouns)."""
    if not text:
        return []
    
    doc = nlp(text[:500])
    
    actions = []
    invalid_pronouns = {'this', 'that', 'it', 'these', 'those'}
    
    # Look for multiple main verbs
    verbs = [token for token in doc if token.pos_ == 'VERB' and token.dep_ in ('ROOT', 'conj')]
    
    if len(verbs) > 1:
        for verb in verbs:
            # Find subject for this verb
            subject = None
            for child in verb.children:
                if child.dep_ in ('nsubj', 'nsubjpass'):
                    # STRICT: Must be NOUN or PROPN
                    if child.pos_ in ('NOUN', 'PROPN'):
                        # Reject pronouns
                        if child.text.lower() not in invalid_pronouns:
                            subject = child.text
                    break
            
            # Build verb phrase
            verb_phrase_parts = [verb.text]
            modals_to_remove = {'shall', 'must', 'should', 'will', 'may', 'would', 'could', 'can'}
            
            for child in verb.children:
                # Skip modals
                if child.text.lower() in modals_to_remove:
                    continue
                    
                if child.dep_ in ('dobj', 'prep', 'pobj', 'advmod', 'acomp', 'xcomp'):
                    verb_phrase_parts.append(child.text)
                elif child.dep_ == 'aux' and child.text.lower() not in modals_to_remove:
                    verb_phrase_parts.append(child.text)
            
            verb_phrase = ' '.join(verb_phrase_parts)
            
            if subject and verb_phrase:
                actions.append((subject, verb_phrase))
    
    return actions


def infer_cru_type(req_id: str, text: str, category: str) -> str:
    """Infer CRU type based on requirement ID pattern and category.
    STRICT: Never return 'quality' - use specific type or 'other'."""
    if category == "functional":
        return "functional"
    elif category == "performance":
        return "performance"
    elif category == "quality":
        # Determine sub-type based on content indicators
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['security', 'password', 'hash', 'encrypt', 'authenticate', 'authorize']):
            return "security"
        elif any(kw in text_lower for kw in ['usability', 'responsive', 'interface', 'navigation', 'accessible']):
            return "usability"
        elif any(kw in text_lower for kw in ['reliability', 'uptime', 'available', 'backup', 'recovery']):
            return "reliability"
        elif any(kw in text_lower for kw in ['portable', 'browser', 'platform', 'device']):
            return "portability"
        else:
            # NEVER return "quality" - use "other" as final fallback
            return "other"
    else:
        return "other"


# ======================
# Multi-Field CRU Extraction
# ======================

def extract_crus_from_fields(req: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    """Extract CRUs from all meaningful fields in a requirement.
    
    This is the main extraction algorithm that:
    1. Processes each semantic field (description, system_behavior, outputs, etc.)
    2. Splits compound statements
    3. Extracts atomic behaviors
    4. Maintains traceability
    """
    req_id = req.get("id", "UNKNOWN")
    source_ref = req.get("source_ref", {})
    dependencies = req.get("dependencies", [])
    
    # Fields to process in order of semantic priority
    fields_to_process = [
        ("description", req.get("description", "")),
        ("system_behavior", req.get("system_behavior", "")),
        ("outputs", req.get("outputs", "")),
        ("inputs", req.get("inputs", "")),
    ]
    
    all_crus = []
    cru_counter = 1
    
    for field_name, field_text in fields_to_process:
        if not field_text or len(field_text.strip()) < 10:
            continue
        
        # Split into segments
        segments = split_compound_sentence(field_text)
        
        for segment in segments:
            if len(segment.strip()) < 10:
                continue
            
            # Check for multiple actions in segment
            multiple_actions = detect_multiple_actions(segment)
            
            if len(multiple_actions) > 1:
                # Create separate CRU for each action
                for actor_raw, action_raw in multiple_actions:
                    crus = create_cru_from_segment(
                        segment=segment,
                        req_id=req_id,
                        cru_counter=cru_counter,
                        category=category,
                        source_ref=source_ref,
                        dependencies=dependencies,
                        field_name=field_name,
                        actor_override=actor_raw,
                        action_override=action_raw
                    )
                    if crus:
                        all_crus.extend(crus)
                        cru_counter += len(crus)
            else:
                # Create CRU(s) for this segment
                crus = create_cru_from_segment(
                    segment=segment,
                    req_id=req_id,
                    cru_counter=cru_counter,
                    category=category,
                    source_ref=source_ref,
                    dependencies=dependencies,
                    field_name=field_name
                )
                if crus:
                    all_crus.extend(crus)
                    cru_counter += len(crus)
    
    return all_crus


def create_cru_from_segment(
    segment: str,
    req_id: str,
    cru_counter: int,
    category: str,
    source_ref: Dict,
    dependencies: List,
    field_name: str,
    actor_override: Optional[str] = None,
    action_override: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Create CRU(s) from a text segment.
    Returns list because multiple constraints may create multiple CRUs."""
    
    # Extract components
    actor = actor_override if actor_override else extract_grammatical_subject(segment)
    action = action_override if action_override else extract_main_verb_phrase(segment)
    constraints = detect_constraint_or_metric(segment)  # Now returns list
    outcome = detect_observable_outcome(segment)
    
    # Default actor to "System" if not found
    if not actor:
        actor = "System"
    
    # Skip if no meaningful action found
    if not action or len(action.split()) < 2:
        return []
    
    # Determine CRU type
    cru_type = infer_cru_type(req_id, segment, category)
    
    crus = []
    
    # If multiple constraints, create separate CRU for each
    if constraints and len(constraints) > 1:
        for idx, constraint in enumerate(constraints):
            cru = {
                "cru_id": f"CRU_{req_id}_{cru_counter + idx:02d}",
                "parent_requirement_id": req_id,
                "type": cru_type,
                "actor": actor.strip(),
                "action": action.strip(),
                "observable_outcome": outcome.strip() if outcome else None,
                "constraint_or_metric": constraint.strip(),
                "source": "SRS",
                "traceability": {
                    "section": source_ref.get("section"),
                    "page": source_ref.get("page")
                },
                "original_text": segment.strip()
            }
            crus.append(cru)
    else:
        # Single CRU
        cru = {
            "cru_id": f"CRU_{req_id}_{cru_counter:02d}",
            "parent_requirement_id": req_id,
            "type": cru_type,
            "actor": actor.strip(),
            "action": action.strip(),
            "observable_outcome": outcome.strip() if outcome else None,
            "constraint_or_metric": constraints[0].strip() if constraints else None,
            "source": "SRS",
            "traceability": {
                "section": source_ref.get("section"),
                "page": source_ref.get("page")
            },
            "original_text": segment.strip()
        }
        crus.append(cru)
    
    return crus


# ======================
# Process All Requirements
# ======================
print("\n📄 Generating Canonical Requirement Units (CRUs)...")

all_crus = []
stats = {
    "functional": 0,
    "performance": 0,
    "security": 0,
    "usability": 0,
    "reliability": 0,
    "portability": 0,
    "quality": 0,
    "other": 0
}

# Process functional requirements
for req in data.get("functional_requirements", []):
    crus = extract_crus_from_fields(req, "functional")
    all_crus.extend(crus)
    for cru in crus:
        stats[cru["type"]] += 1

# Process quality requirements
for req in data.get("quality_requirements", []):
    crus = extract_crus_from_fields(req, "quality")
    all_crus.extend(crus)
    for cru in crus:
        stats[cru["type"]] += 1

# Process performance requirements
for req in data.get("performance_requirements", []):
    crus = extract_crus_from_fields(req, "performance")
    all_crus.extend(crus)
    for cru in crus:
        stats[cru["type"]] += 1

# Process constraints
for req in data.get("constraints", []):
    crus = extract_crus_from_fields(req, "constraint")
    all_crus.extend(crus)
    for cru in crus:
        stats[cru["type"]] += 1

# Process use cases
for req in data.get("use_cases", []):
    crus = extract_crus_from_fields(req, "use_case")
    all_crus.extend(crus)
    for cru in crus:
        stats[cru["type"]] += 1

print(f"✅ Generated {len(all_crus)} CRUs from all requirements")

# ======================
# Save Output
# ======================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

output = {
    "metadata": {
        "total_crus": len(all_crus),
        "source_requirements": sum(len(data.get(k, [])) for k in [
            'functional_requirements', 'quality_requirements', 'use_cases', 
            'constraints', 'performance_requirements'
        ]),
        "cru_types": stats
    },
    "crus": all_crus
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"💾 Saved to {OUTPUT_PATH}")

# ======================
# Summary Report
# ======================
print(f"\n{'='*60}")
print("📊 CRU GENERATION SUMMARY")
print(f"{'='*60}")
print(f"Total CRUs Generated:        {len(all_crus)}")
print(f"Source Requirements:         {output['metadata']['source_requirements']}")
print(f"\nCRU Type Breakdown:")
for cru_type, count in stats.items():
    if count > 0:
        print(f"  {cru_type.title():20s}  {count}")
print(f"{'='*60}")

# Sample CRUs
print(f"\n{'='*60}")
print("SAMPLE CRUs (First 5)")
print(f"{'='*60}")

for idx, cru in enumerate(all_crus[:5], 1):
    print(f"\n🔹 CRU {idx}:")
    print(f"   ID: {cru['cru_id']}")
    print(f"   Parent: {cru['parent_requirement_id']}")
    print(f"   Type: {cru['type']}")
    print(f"   Actor: {cru['actor']}")
    print(f"   Action: {cru['action'][:80]}...")
    if cru.get('constraint_or_metric'):
        print(f"   Constraint: {cru['constraint_or_metric'][:80]}...")
    if cru.get('observable_outcome'):
        print(f"   Outcome: {cru['observable_outcome'][:80]}...")

print(f"\n{'='*60}")
print("✅ CRU normalization complete!")
print(f"{'='*60}")