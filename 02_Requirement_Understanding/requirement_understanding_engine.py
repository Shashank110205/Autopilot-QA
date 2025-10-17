import json
import spacy
import re
from transformers import pipeline
from tqdm import tqdm
import os
from typing import Dict, List, Any, Optional

# ======================
# Load Input Data
# ======================
INPUT_PATH = "../01_Multi_Source_Document_Ingestion/output/structured_output.json"
OUTPUT_PATH = "../02_Requirement_Understanding/output/requirements_extracted.json"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError("❌ structured_output.json not found! Run document_ingestion_engine.py first.")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("📄 Loaded structured_output.json")

# ======================
# Load NLP Models
# ======================
print("⚙️ Loading NLP models...")
nlp = spacy.load("en_core_web_sm")
ner_model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# ======================
# Helper Functions
# ======================

def normalize_id(id_str: str, context_text: str = "") -> Optional[str]:
    """Normalize requirement IDs to proper format (FR1, QR6, etc.)"""
    if not id_str:
        return None
    
    id_str = str(id_str).strip().upper()
    
    # If just FR or QR without number, try to extract from context
    if id_str in ["FR", "QR", "NFR"]:
        match = re.search(rf'{id_str}(\d+)', context_text)
        if match:
            return f"{id_str}{match.group(1)}"
        return None
    
    # If just a number, infer type from context
    if id_str.isdigit():
        # Look for FR/QR nearby in context
        if re.search(r'\bFR[:\s]', context_text[:100]):
            return f"FR{id_str}"
        elif re.search(r'\bQR[:\s]', context_text[:100]):
            return f"QR{id_str}"
        # Default to FR if ambiguous
        return f"FR{id_str}"
    
    # Validate format
    if re.match(r'^(FR|QR|NFR)\d+$', id_str):
        return id_str
    
    return None

def parse_dependencies(dep_text: str) -> List[str]:
    """Parse dependency text into list of requirement IDs."""
    if not dep_text or dep_text.strip().lower() in ['none', '-', 'n/a']:
        return []
    
    # Extract all FR/QR patterns
    deps = re.findall(r'\b(FR|QR|NFR)[\s\-]*(\d+)\b', dep_text)
    return sorted(list(set([f"{prefix}{num}" for prefix, num in deps])))

def clean_text(text: str) -> str:
    """Clean extracted text of artifacts and section headers."""
    if not text:
        return ""
    
    # Remove section number contamination
    text = re.sub(r'\n+\d+\.\d+(\.\d+)?\s+[A-Z][a-z].*$', '', text, flags=re.MULTILINE)
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def extract_complete_field(text: str, field_name: str, next_field: str = None) -> str:
    """Extract a complete field value without truncation."""
    # Build regex pattern
    if next_field:
        pattern = rf'{field_name}:\s*(.+?)(?=\n\s*(?:{next_field}|ID\s*:)|$)'
    else:
        pattern = rf'{field_name}:\s*(.+?)(?=\n\s*(?:ID\s*:|TITLE|DESC|RAT|DEP|TAG|GIST)|$)'
    
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return clean_text(match.group(1))
    return ""

def extract_standard_requirements_v2(text: str) -> List[Dict[str, Any]]:
    """Enhanced extraction with complete field capture."""
    requirements = []
    
    # Find all requirement blocks (ID to next ID)
    req_blocks = re.split(r'\n(?=ID:\s*(?:FR|QR|NFR))', text)
    
    for block in req_blocks:
        if not block.strip() or 'ID:' not in block:
            continue
        
        # Extract ID
        id_match = re.search(r'ID:\s*([A-Z]*\d*)', block, re.IGNORECASE)
        if not id_match:
            continue
        
        raw_id = id_match.group(1).strip()
        req_id = normalize_id(raw_id, block[:500])
        
        if not req_id:
            continue
        
        # Extract all fields completely
        title = extract_complete_field(block, 'TITLE', 'DESC')
        description = extract_complete_field(block, 'DESC', 'RAT')
        rationale = extract_complete_field(block, 'RAT', 'DEP')
        dep_text = extract_complete_field(block, 'DEP')
        
        req = {
            "id": req_id,
            "title": title,
            "description": description,
            "rationale": rationale,
            "dependencies": parse_dependencies(dep_text)
        }
        
        # Only add if we have meaningful content
        if title or description:
            requirements.append(req)
    
    return requirements

def extract_planguage_requirements_v2(text: str) -> List[Dict[str, Any]]:
    """Enhanced PLanguage extraction with all metrics."""
    requirements = []
    
    # Find all PLanguage requirement blocks
    req_blocks = re.split(r'\n(?=ID:\s*QR)', text)
    
    for block in req_blocks:
        if not block.strip() or 'TAG:' not in block:
            continue
        
        # Extract ID
        id_match = re.search(r'ID:\s*(QR\d+)', block, re.IGNORECASE)
        if not id_match:
            continue
        
        req_id = id_match.group(1).strip().upper()
        
        # Extract all PLanguage fields
        tag = extract_complete_field(block, 'TAG', 'GIST')
        gist = extract_complete_field(block, 'GIST', 'SCALE')
        scale = extract_complete_field(block, 'SCALE', 'METER')
        meter = extract_complete_field(block, 'METER', 'MUST')
        must = extract_complete_field(block, 'MUST', 'PLAN')
        plan = extract_complete_field(block, 'PLAN', 'WISH')
        wish = extract_complete_field(block, 'WISH')
        
        # Also try to get TITLE and DESC if present
        title = extract_complete_field(block, 'TITLE', 'DESC')
        description = extract_complete_field(block, 'DESC', 'RAT')
        rationale = extract_complete_field(block, 'RAT', 'DEP')
        
        req = {
            "id": req_id,
            "title": title if title else gist,
            "tag": tag,
            "gist": gist,
            "scale": scale,
            "meter": meter,
            "must": must,
            "plan": plan,
            "wish": wish,
            "description": description,
            "rationale": rationale,
            "dependencies": []
        }
        
        requirements.append(req)
    
    return requirements

def extract_gherkin_requirements_v2(text: str) -> List[Dict[str, Any]]:
    """Enhanced Gherkin/BDD extraction with proper actor handling."""
    requirements = []
    
    # Find feature blocks
    feature_pattern = r'ID:\s*(FR\d+)\s*\n\s*Feature:\s*(.+?)\n\s*In order to\s+(.+?)\n\s*(?:A|An)\s+(.+?)\n\s*Should\s+(.+?)(?=\n\s*Scenario|ID\s*:|$)'
    
    features = re.finditer(feature_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for feature_match in features:
        req_id = feature_match.group(1).strip().upper()
        feature_name = clean_text(feature_match.group(2))
        business_value = clean_text(feature_match.group(3))
        actor = clean_text(feature_match.group(4))
        capability = clean_text(feature_match.group(5))
        
        # Fix actor formatting (restore article)
        if not actor.startswith(('A ', 'An ', 'The ')):
            # Determine article
            if actor[0].lower() in 'aeiou':
                actor = f"An {actor}"
            else:
                actor = f"A {actor}"
        
        # Find the full feature block for scenario extraction
        feature_end = feature_match.end()
        next_id = re.search(r'\nID:\s*FR', text[feature_end:])
        feature_block_end = feature_end + next_id.start() if next_id else len(text)
        feature_block = text[feature_match.start():feature_block_end]
        
        # Extract scenarios
        scenarios = []
        scenario_pattern = r'Scenario:\s*(.+?)\n((?:\s*(?:Given|When|And|Then).+?\n)+)'
        
        for scenario_match in re.finditer(scenario_pattern, feature_block, re.MULTILINE):
            scenario_name = clean_text(scenario_match.group(1))
            steps_text = scenario_match.group(2)
            
            # Extract steps
            steps = []
            step_pattern = r'\s*(Given|When|And|Then)\s+(.+?)(?=\n|$)'
            for step_match in re.finditer(step_pattern, steps_text, re.MULTILINE):
                steps.append({
                    "keyword": step_match.group(1).strip(),
                    "text": clean_text(step_match.group(2))
                })
            
            if steps:
                scenarios.append({
                    "name": scenario_name,
                    "steps": steps
                })
        
        req = {
            "id": req_id,
            "title": feature_name,
            "feature": feature_name,
            "business_value": business_value,
            "actor": actor,
            "capability": capability,
            "scenarios": scenarios,
            "description": f"Feature: {feature_name}. {business_value}",
            "rationale": f"In order to {business_value}",
            "dependencies": []
        }
        
        requirements.append(req)
    
    return requirements

def merge_subword_tokens(entities: List[Dict]) -> List[Dict]:
    """Merge BERT subword tokens (## prefixes) into complete words."""
    if not entities:
        return []
    
    merged = []
    current_entity = None
    
    for ent in entities:
        text = ent.get("word", "")
        
        if text.startswith("##"):
            if current_entity:
                current_entity["text"] += text[2:]
            else:
                current_entity = {
                    "entity": ent["entity_group"],
                    "text": text[2:]
                }
        else:
            if current_entity:
                merged.append(current_entity)
            current_entity = {
                "entity": ent["entity_group"],
                "text": text
            }
    
    if current_entity:
        merged.append(current_entity)
    
    return merged

def consolidate_requirements_v2(all_reqs: List[List[Dict]]) -> Dict[str, Dict]:
    """Advanced consolidation with field-level merging."""
    consolidated = {}
    
    for req_list in all_reqs:
        for req in req_list:
            req_id = req.get("id")
            if not req_id:
                continue
            
            if req_id not in consolidated:
                consolidated[req_id] = req
            else:
                # Merge fields intelligently
                existing = consolidated[req_id]
                
                for key, value in req.items():
                    if key == "id":
                        continue
                    
                    # For lists (dependencies, scenarios)
                    if isinstance(value, list) and value:
                        if not existing.get(key):
                            existing[key] = value
                        elif isinstance(existing[key], list):
                            # Merge unique items
                            existing[key] = list(set(existing[key] + value))
                    
                    # For strings, prefer longer/more complete values
                    elif isinstance(value, str) and value:
                        if not existing.get(key) or len(value) > len(str(existing.get(key, ""))):
                            existing[key] = value
                    
                    # For dicts (scenarios)
                    elif isinstance(value, dict) and value:
                        if not existing.get(key):
                            existing[key] = value
    
    return consolidated

def extract_entities(text: str, max_length: int = 10000) -> List[Dict]:
    """Extract and merge named entities using BERT NER."""
    if len(text) > max_length:
        text = text[:max_length]
    
    try:
        raw_entities = ner_model(text)
        merged = merge_subword_tokens(raw_entities)
        
        noise_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'for'}
        return [e for e in merged if e['text'].lower() not in noise_words and len(e['text']) > 1]
    except Exception as e:
        print(f"⚠️ Entity extraction failed: {e}")
        return []

# ======================
# Main Extraction
# ======================
print("\n🔍 Starting multi-pass requirement extraction...")

# Concatenate all pages
full_text = "\n\n".join([page["content"] for page in data["text"]])

# Pass 1: Standard requirements
print("   Pass 1: Standard FR/QR requirements...")
standard_reqs = extract_standard_requirements_v2(full_text)
print(f"   ✓ Found {len(standard_reqs)} standard requirements")

# Pass 2: PLanguage quality requirements
print("   Pass 2: PLanguage quality requirements...")
planguage_reqs = extract_planguage_requirements_v2(full_text)
print(f"   ✓ Found {len(planguage_reqs)} PLanguage requirements")

# Pass 3: Gherkin/BDD use cases
print("   Pass 3: Gherkin/BDD use cases...")
gherkin_reqs = extract_gherkin_requirements_v2(full_text)
print(f"   ✓ Found {len(gherkin_reqs)} Gherkin requirements")

# Consolidate
print("\n📋 Consolidating requirements...")
all_reqs_dict = consolidate_requirements_v2([standard_reqs, planguage_reqs, gherkin_reqs])
all_requirements = list(all_reqs_dict.values())

# Sort by ID
all_requirements.sort(key=lambda x: (
    x.get("id", "ZZZ")[0:2],  # FR, QR, etc.
    int(re.search(r'\d+', x.get("id", "0")).group()) if re.search(r'\d+', x.get("id", "0")) else 0
))

# Process page metadata
print("\n📄 Processing page metadata...")
page_results = []
for page in tqdm(data["text"], desc="Processing pages"):
    page_text = page["content"].strip()
    if not page_text:
        continue
    
    req_ids = list(set(re.findall(r'\b(FR|QR|NFR)\d+\b', page_text)))
    entities = extract_entities(page_text[:8000])
    
    page_results.append({
        "page": page["page"],
        "requirement_ids": sorted(req_ids),
        "entities": entities,
        "text_length": len(page_text)
    })

# ======================
# Group Requirements
# ======================
print("\n📊 Grouping requirements...")

grouped = {
    "functional_requirements": [],
    "quality_requirements": [],
    "use_cases": [],
    "constraints": [],
    "performance_requirements": []
}

for req in all_requirements:
    req_id = req.get("id", "")
    
    if req_id.startswith("FR"):
        if "scenarios" in req and req["scenarios"]:
            grouped["use_cases"].append(req)
        else:
            grouped["functional_requirements"].append(req)
    elif req_id.startswith("QR"):
        gist_lower = req.get("gist", "").lower()
        scale_lower = req.get("scale", "").lower()
        
        if any(kw in gist_lower or kw in scale_lower for kw in ["hard drive", "memory", "space", "mb"]):
            grouped["constraints"].append(req)
        elif any(kw in gist_lower or kw in scale_lower for kw in ["response", "performance", "speed", "time", "availability"]):
            grouped["performance_requirements"].append(req)
        else:
            grouped["quality_requirements"].append(req)

# ======================
# Validation & Statistics
# ======================
print("\n🔍 Validating extraction...")

# Check for incomplete requirements
incomplete = []
for req in all_requirements:
    issues = []
    if not req.get("description") and not req.get("scenarios"):
        issues.append("missing description")
    if not req.get("title") and not req.get("feature"):
        issues.append("missing title")
    
    if issues:
        incomplete.append({
            "id": req.get("id"),
            "issues": issues
        })

if incomplete:
    print(f"   ⚠️ {len(incomplete)} requirements have incomplete data")
    for item in incomplete[:5]:
        print(f"      - {item['id']}: {', '.join(item['issues'])}")

# ======================
# Save Outputs
# ======================
os.makedirs("../02_Requirement_Understanding/output", exist_ok=True)

complete_output = {
    "metadata": {
        "total_requirements": len(all_requirements),
        "functional_requirements": len(grouped["functional_requirements"]),
        "quality_requirements": len(grouped["quality_requirements"]),
        "use_cases": len(grouped["use_cases"]),
        "constraints": len(grouped["constraints"]),
        "performance_requirements": len(grouped["performance_requirements"]),
        "incomplete_requirements": len(incomplete),
        "pages_processed": len(page_results)
    },
    "grouped_requirements": grouped,
    "all_requirements": all_requirements,
    "page_metadata": page_results,
    "validation": {
        "incomplete_requirements": incomplete
    }
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(complete_output, f, ensure_ascii=False, indent=2)

grouped_output_path = OUTPUT_PATH.replace(".json", "_grouped.json")
with open(grouped_output_path, "w", encoding="utf-8") as f:
    json.dump(grouped, f, ensure_ascii=False, indent=2)

# ======================
# Summary Report
# ======================
print(f"\n✅ Extraction complete!")
print(f"   📄 Complete output: {OUTPUT_PATH}")
print(f"   📊 Grouped output: {grouped_output_path}")

print(f"\n{'='*60}")
print(f"📈 EXTRACTION SUMMARY")
print(f"{'='*60}")
print(f"Total Requirements:           {len(all_requirements)}")
print(f"Functional Requirements:      {len(grouped['functional_requirements'])}")
print(f"Use Cases (Gherkin):          {len(grouped['use_cases'])}")
print(f"Quality Requirements:         {len(grouped['quality_requirements'])}")
print(f"Performance Requirements:     {len(grouped['performance_requirements'])}")
print(f"Constraints:                  {len(grouped['constraints'])}")
print(f"{'='*60}")

# Coverage analysis
expected = {"FR": 33, "QR": 23}
actual_fr = len(grouped['functional_requirements']) + len(grouped['use_cases'])
actual_qr = len(grouped['quality_requirements']) + len(grouped['constraints']) + len(grouped['performance_requirements'])

print(f"\n📊 COVERAGE ANALYSIS")
print(f"{'='*60}")
print(f"Functional Requirements: {actual_fr}/{expected['FR']} ({actual_fr/expected['FR']*100:.1f}%)")
print(f"Quality Requirements:    {actual_qr}/{expected['QR']} ({actual_qr/expected['QR']*100:.1f}%)")
print(f"{'='*60}")

# List IDs
all_fr_ids = sorted([r['id'] for r in grouped['functional_requirements'] + grouped['use_cases']])
all_qr_ids = sorted([r['id'] for r in grouped['quality_requirements'] + grouped['performance_requirements'] + grouped['constraints']])

print(f"\n📋 EXTRACTED REQUIREMENT IDs")
print(f"FR: {', '.join(all_fr_ids)}")
print(f"QR: {', '.join(all_qr_ids)}")

# Sample outputs
print(f"\n{'='*60}")
print("SAMPLE REQUIREMENTS")
print(f"{'='*60}")

if grouped["functional_requirements"]:
    print("\n🔹 Functional Requirement:")
    sample = grouped["functional_requirements"][0]
    print(f"   ID: {sample.get('id')}")
    print(f"   Title: {sample.get('title', '')[:60]}...")
    print(f"   Description: {sample.get('description', '')[:80]}...")
    print(f"   Dependencies: {sample.get('dependencies', [])}")

if grouped["use_cases"]:
    print("\n🔹 Use Case:")
    sample = grouped["use_cases"][0]
    print(f"   ID: {sample.get('id')}")
    print(f"   Feature: {sample.get('feature')}")
    print(f"   Actor: {sample.get('actor')}")
    print(f"   Scenarios: {len(sample.get('scenarios', []))}")

if grouped["performance_requirements"]:
    print("\n🔹 Performance Requirement:")
    sample = grouped["performance_requirements"][0]
    print(f"   ID: {sample.get('id')}")
    print(f"   Tag: {sample.get('tag')}")
    print(f"   Must: {sample.get('must', '')[:60]}...")

print(f"\n{'='*60}")
print("✅ Requirement extraction complete!")
print(f"{'='*60}")