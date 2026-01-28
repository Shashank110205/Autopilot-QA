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
ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

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

def extract_standard_requirements_v2(text: str, source_file: str = "structured_output.json") -> List[Dict[str, Any]]:
    """Enhanced extraction for IEEE SRS format with section-based requirements."""
    requirements = []
    
    # Pattern 1: Section headers like "3.1 FR 1 User Signup {#fr-1}"
    section_pattern = r'(\d+\.\d+)\s+(FR|QR|NFR)\s*(\d+)\s+([^\n{]+?)(?:\s*\{#[^}]+\})?\s*\n(.+?)(?=\n\d+\.\d+\s+|$)'
    
    for match in re.finditer(section_pattern, text, re.DOTALL | re.IGNORECASE):
        section_num = match.group(1)
        req_type = match.group(2).upper()
        req_num = match.group(3)
        title = clean_text(match.group(4))
        content = match.group(5)
        
        req_id = f"{req_type}{req_num}"
        
        # Extract Description, Inputs, System Behavior, Outputs
        # Extract Description, Inputs, System Behavior, Outputs
        description = ""
        inputs = ""
        behavior = ""
        outputs = ""
        rationale = ""
        dependencies = []

        # Look for "Description:" section OR take content before "Inputs:"
        desc_match = re.search(r'Description:\s*(.+?)(?=\nInputs?:|$)', content, re.DOTALL | re.IGNORECASE)
        if desc_match:
            description = clean_text(desc_match.group(1))
        else:
            # If no explicit "Description:", take content before "Inputs:" or first 300 chars
            before_inputs = re.search(r'(.+?)(?=\nInputs?:|$)', content, re.DOTALL)
            if before_inputs:
                desc_text = before_inputs.group(1)
                # Stop if we hit another section marker
                desc_text = re.split(r'\n\d+\.\d+\s+', desc_text)[0]
                description = clean_text(desc_text)[:300]
        
        # Look for "Inputs:"
        inputs_match = re.search(r'Inputs?:\s*(.+?)(?=\n(?:System Behavior:|Outputs:|Rationale:|\d+\.\d+)|$)', content, re.DOTALL | re.IGNORECASE)
        if inputs_match:
            inputs = clean_text(inputs_match.group(1))
        
        # Look for "System Behavior:"
        behavior_match = re.search(r'System Behavior:\s*(.+?)(?=\n(?:Outputs:|Rationale:|\d+\.\d+)|$)', content, re.DOTALL | re.IGNORECASE)
        if behavior_match:
            behavior = clean_text(behavior_match.group(1))
        
        # Look for "Outputs:"
        outputs_match = re.search(r'Outputs?:\s*(.+?)(?=\n(?:Rationale:|\d+\.\d+)|$)', content, re.DOTALL | re.IGNORECASE)
        if outputs_match:
            outputs = clean_text(outputs_match.group(1))
        
        # Extract page from nearby context
        page_match = re.search(r'page[:\s]+(\d+)', text[max(0, match.start()-200):match.start()], re.IGNORECASE)
        page_num = int(page_match.group(1)) if page_match else None
        
        req = {
            "id": req_id,
            "title": title,
            "description": description,
            "inputs": inputs,
            "system_behavior": behavior,
            "outputs": outputs,
            "rationale": rationale,
            "dependencies": dependencies,
            "source_ref": {
                "page": page_num,
                "source_file": source_file,
                "extraction_type": "text",
                "requirement_type": "standard",
                "section": section_num
            }
        }
        
        requirements.append(req)
    
    return requirements

def extract_planguage_requirements_v2(text: str, source_file: str = "structured_output.json") -> List[Dict[str, Any]]:
    """Enhanced PLanguage extraction for quality requirements."""
    requirements = []
    
    # Pattern: Look for quality requirement sections in Chapter 4
    # "4.1 Performance", "4.2 Security", etc.
    quality_pattern = r'(\d+\.\d+)\s+([A-Za-z]+)\s*(?:\{#[^}]+\})?\s*\n(.+?)(?=\n\d+\.\d+\s+[A-Z]|\n\d+\.\s+[A-Z]|$)'
    
    qr_counter = 1
    
    for match in re.finditer(quality_pattern, text, re.DOTALL):
        section_num = match.group(1)
        category = match.group(2).strip()
        content = match.group(3)
        
        # Only process if in section 4 (Non-Functional Requirements)
        # Only process if in section 4 (Non-Functional Requirements)
        if not (section_num.startswith('4.') and len(section_num.split('.')) == 2):
            continue

        # Stop content at next section
        content = re.split(r'\n\d+\.\d+\s+', content)[0]
        
        # Skip if too short
        if len(content.strip()) < 50:
            continue
        
        req_id = f"QR{qr_counter}"
        qr_counter += 1
        
        # Clean up content
        description = clean_text(content)
        
        # Extract page
        page_match = re.search(r'page[:\s]+(\d+)', text[max(0, match.start()-200):match.start()], re.IGNORECASE)
        page_num = int(page_match.group(1)) if page_match else None
        
        req = {
            "id": req_id,
            "title": category,
            "tag": category.lower(),
            "gist": category,
            "description": description,
            "category": category,
            "dependencies": [],
            "source_ref": {
                "page": page_num,
                "source_file": source_file,
                "extraction_type": "text",
                "requirement_type": "planguage",
                "section": section_num
            }
        }
        
        requirements.append(req)
        
    
    return requirements

def extract_gherkin_requirements_v2(text: str, source_file: str = "structured_output.json") -> List[Dict[str, Any]]:
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
        
        # Extract page number from nearby context
        context_start = max(0, feature_match.start() - 200)
        context = text[context_start:feature_match.start()]
        page_match = re.search(r'Page[:\s]+(\d+)', context, re.IGNORECASE)
        page_num = int(page_match.group(1)) if page_match else None
        
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
            "dependencies": [],
            "source_ref": {
                "page": page_num,
                "source_file": source_file,
                "extraction_type": "text",
                "requirement_type": "gherkin"
            }
        }
        
        requirements.append(req)
    
    return requirements


def extract_modern_gherkin_scenarios(text: str, source_file: str = "structured_output.json") -> List[Dict[str, Any]]:
    """Extract modern Gherkin scenarios with embedded API specifications."""
    requirements = []
    
    # Pattern for modern Gherkin format
    feature_pattern = r'Feature:\s*(.+?)\s*(?:\(([^)]+)\))?\s*Background:'
    
    for feature_match in re.finditer(feature_pattern, text, re.DOTALL):
        feature_name = clean_text(feature_match.group(1))
        user_story_ids = feature_match.group(2) if feature_match.group(2) else None
        
        # Extract the full feature block
        feature_start = feature_match.start()
        next_feature = re.search(r'\nFeature:', text[feature_start + 10:])
        feature_end = feature_start + next_feature.start() + 10 if next_feature else len(text)
        feature_block = text[feature_start:feature_end]
        
        # Extract Background
        background_match = re.search(r'Background:\s*(.+?)(?=\n\s*Scenario:)', feature_block, re.DOTALL)
        background = []
        if background_match:
            background_text = background_match.group(1)
            for line in background_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('Given') or line.startswith('And')):
                    background.append(clean_text(line))
        
        # Extract all scenarios
        scenarios = []
        scenario_pattern = r'Scenario:\s*(.+?)\n((?:\s*(?:Given|When|Then|And).+?\n)+)'
        
        for scenario_match in re.finditer(scenario_pattern, feature_block, re.MULTILINE):
            scenario_name = clean_text(scenario_match.group(1))
            steps_text = scenario_match.group(2)
            
            # Extract steps with API calls
            steps = []
            api_calls = []
            
            lines = steps_text.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if not line or not any(line.startswith(kw) for kw in ['Given', 'When', 'Then', 'And']):
                    i += 1
                    continue
                
                # Check if next lines contain API spec
                if 'API' in line or 'POST' in line or 'GET' in line or 'PUT' in line or 'DELETE' in line:
                    # Extract API endpoint
                    api_match = re.search(r'(POST|GET|PUT|DELETE)\s+(/[^\s]+)', line)
                    if api_match:
                        method = api_match.group(1)
                        endpoint = api_match.group(2)
                        
                        # Look for JSON response in next lines
                        response = None
                        if i + 1 < len(lines) and '"""' in lines[i + 1]:
                            # Extract JSON block
                            json_lines = []
                            i += 2
                            while i < len(lines) and '"""' not in lines[i]:
                                json_lines.append(lines[i])
                                i += 1
                            response = '\n'.join(json_lines)
                        
                        api_calls.append({
                            "method": method,
                            "endpoint": endpoint,
                            "response": response
                        })
                
                steps.append({
                    "keyword": line.split()[0],
                    "text": clean_text(' '.join(line.split()[1:]))
                })
                i += 1
            
            if steps:
                scenarios.append({
                    "name": scenario_name,
                    "steps": steps,
                    "api_calls": api_calls
                })
        
        # Create requirement ID from user story IDs
        req_id = None
        if user_story_ids:
            # Extract first ID (e.g., "US-U-01" from "US-U-01 to US-U-12")
            id_match = re.search(r'(US-[A-Z]+-\d+)', user_story_ids)
            if id_match:
                req_id = id_match.group(1).replace('-', '_')
        
        if not req_id:
            # Generate from feature name
            req_id = "FEAT_" + re.sub(r'[^A-Z0-9]', '_', feature_name.upper())[:20]
        
        req = {
            "id": req_id,
            "title": feature_name,
            "feature": feature_name,
            "user_story_ids": user_story_ids,
            "background": background,
            "scenarios": scenarios,
            "description": f"Feature: {feature_name}",
            "dependencies": [],
            "source_ref": {
                "page": None,
                "source_file": source_file,
                "extraction_type": "text",
                "requirement_type": "modern_gherkin"
            }
        }
        
        requirements.append(req)
    
    return requirements

def extract_api_specifications(text: str, source_file: str = "structured_output.json") -> List[Dict[str, Any]]:
    """Extract standalone API endpoint specifications."""
    requirements = []
    
    # Pattern for API endpoints
    api_pattern = r'(POST|GET|PUT|DELETE)\s+(/api/[^\s\n]+)'
    
    for match in re.finditer(api_pattern, text):
        method = match.group(1)
        endpoint = match.group(2)
        
        # Extract context around this API
        context_start = max(0, match.start() - 500)
        context_end = min(len(text), match.end() + 1000)
        context = text[context_start:context_end]
        
        # Extract request/response examples
        request = None
        responses = []
        
        # Look for request body
        req_match = re.search(r'(?:requestBody|Request).*?(\{[^}]+\})', context, re.DOTALL)
        if req_match:
            request = req_match.group(1)
        
        # Look for responses
        resp_pattern = r'Response\s+(\d+):\s*(\{[^}]+\}|\[.+?\])'
        for resp_match in re.finditer(resp_pattern, context, re.DOTALL):
            status_code = resp_match.group(1)
            response_body = resp_match.group(2)
            responses.append({
                "status_code": status_code,
                "body": response_body
            })
        
        # Generate requirement ID
        endpoint_clean = re.sub(r'[^a-zA-Z0-9]', '_', endpoint)
        req_id = f"API_{method}_{endpoint_clean}"[:50]
        
        req = {
            "id": req_id,
            "title": f"{method} {endpoint}",
            "method": method,
            "endpoint": endpoint,
            "request": request,
            "responses": responses,
            "description": f"API endpoint: {method} {endpoint}",
            "dependencies": [],
            "source_ref": {
                "page": None,
                "source_file": source_file,
                "extraction_type": "text",
                "requirement_type": "api_spec"
            }
        }
        
        requirements.append(req)
    
    return requirements

def extract_ui_specifications(text: str, source_file: str = "structured_output.json") -> List[Dict[str, Any]]:
    """Extract UI/UX design specifications."""
    requirements = []
    
    # Extract color scheme
    color_match = re.search(r'PRIMARY:\s*#([0-9A-F]{6})', text, re.IGNORECASE)
    if color_match:
        # Find color scheme section
        color_start = color_match.start() - 100
        color_end = color_match.end() + 500
        color_section = text[color_start:color_end]
        
        colors = {}
        for line in color_section.split('\n'):
            color_line_match = re.match(r'([A-Z]+):\s*#([0-9A-F]{6})\s*\((.+?)\)', line)
            if color_line_match:
                colors[color_line_match.group(1)] = {
                    "hex": f"#{color_line_match.group(2)}",
                    "usage": color_line_match.group(3)
                }
        
        if colors:
            requirements.append({
                "id": "UI_COLOR_SCHEME",
                "title": "Color Scheme Specification",
                "colors": colors,
                "description": "Application color palette",
                "source_ref": {
                    "page": None,
                    "source_file": source_file,
                    "extraction_type": "text",
                    "requirement_type": "ui_spec"
                }
            })
    
    # Extract typography
    typo_pattern = r'Typography:\s*(.+?)(?=\n\n|\nResponsive)'
    typo_match = re.search(typo_pattern, text, re.DOTALL)
    if typo_match:
        typography = {}
        for line in typo_match.group(1).split('\n'):
            typo_line_match = re.match(r'-\s*(.+?):\s*(.+)', line.strip())
            if typo_line_match:
                typography[typo_line_match.group(1)] = typo_line_match.group(2)
        
        if typography:
            requirements.append({
                "id": "UI_TYPOGRAPHY",
                "title": "Typography Specification",
                "typography": typography,
                "description": "Application typography system",
                "source_ref": {
                    "page": None,
                    "source_file": source_file,
                    "extraction_type": "text",
                    "requirement_type": "ui_spec"
                }
            })
    
    # Extract responsive breakpoints
    breakpoint_pattern = r'(DESKTOP|TABLET|MOBILE)\s*\(([^)]+)\):\s*(.+?)(?=\n(?:DESKTOP|TABLET|MOBILE)|\n\n|$)'
    for bp_match in re.finditer(breakpoint_pattern, text, re.DOTALL):
        device = bp_match.group(1)
        size = bp_match.group(2)
        specs = bp_match.group(3).strip()
        
        requirements.append({
            "id": f"UI_RESPONSIVE_{device}",
            "title": f"Responsive Design - {device}",
            "device": device,
            "breakpoint": size,
            "specifications": specs,
            "description": f"Responsive behavior for {device}",
            "source_ref": {
                "page": None,
                "source_file": source_file,
                "extraction_type": "text",
                "requirement_type": "ui_spec"
            }
        })
    
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
                    
                    # Preserve source_ref - don't overwrite
                    if key == "source_ref":
                        if not existing.get("source_ref"):
                            existing["source_ref"] = value
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

# Concatenate all pages WITH page tracking
full_text = "\n\n".join([page["content"] for page in data["text"]])

# Get source file name from input path
source_file = os.path.basename(INPUT_PATH).replace("structured_output.json", "")
if not source_file:
    source_file = "SRS.pdf"  # Default fallback

# Pass 1: Standard requirements
print("   Pass 1: Standard FR/QR requirements...")
standard_reqs = extract_standard_requirements_v2(full_text, source_file)
print(f"   ✓ Found {len(standard_reqs)} standard requirements")

# Pass 2: PLanguage quality requirements
print("   Pass 2: PLanguage quality requirements...")
planguage_reqs = extract_planguage_requirements_v2(full_text, source_file)
print(f"   ✓ Found {len(planguage_reqs)} PLanguage requirements")

# Pass 3: OLD Gherkin/BDD use cases
print("   Pass 3: OLD Gherkin/BDD use cases...")
gherkin_reqs = extract_gherkin_requirements_v2(full_text, source_file)
print(f"   ✓ Found {len(gherkin_reqs)} OLD Gherkin requirements")

# Pass 4: NEW Modern Gherkin scenarios
print("   Pass 4: Modern Gherkin scenarios with API specs...")
modern_gherkin_reqs = extract_modern_gherkin_scenarios(full_text, source_file)
print(f"   ✓ Found {len(modern_gherkin_reqs)} modern Gherkin scenarios")

# Pass 5: API Specifications
print("   Pass 5: Standalone API specifications...")
api_reqs = extract_api_specifications(full_text, source_file)
print(f"   ✓ Found {len(api_reqs)} API specifications")

# Pass 6: UI/UX Specifications
print("   Pass 6: UI/UX design specifications...")
ui_reqs = extract_ui_specifications(full_text, source_file)
print(f"   ✓ Found {len(ui_reqs)} UI/UX specifications")

# Consolidate
print("\n📋 Consolidating requirements...")
all_reqs_dict = consolidate_requirements_v2([
    standard_reqs, 
    planguage_reqs, 
    gherkin_reqs,
    modern_gherkin_reqs,
    api_reqs,
    ui_reqs
])

# FIXED: Convert dict to list
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
    "performance_requirements": [],
    "modern_gherkin_scenarios": [],    # ADD THIS
    "api_specifications": [],          # ADD THIS
    "ui_specifications": []            # ADD THIS
}

for req in all_requirements:
    req_id = req.get("id", "")
    req_type = req.get("source_ref", {}).get("requirement_type", "")
    
    # NEW: Route by requirement_type first
    if req_type == "modern_gherkin":
        grouped["modern_gherkin_scenarios"].append(req)
    elif req_type == "api_spec":
        grouped["api_specifications"].append(req)
    elif req_type == "ui_spec":
        grouped["ui_specifications"].append(req)
    elif req_id.startswith("FR"):
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
        "modern_gherkin_scenarios": len(grouped["modern_gherkin_scenarios"]),    # ADD
        "api_specifications": len(grouped["api_specifications"]),                # ADD
        "ui_specifications": len(grouped["ui_specifications"]),                  # ADD
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

if grouped["modern_gherkin_scenarios"]:
    print("\n🔹 Modern Gherkin Scenario:")
    sample = grouped["modern_gherkin_scenarios"][0]
    print(f"   ID: {sample.get('id')}")
    print(f"   Feature: {sample.get('feature')}")
    print(f"   Scenarios: {len(sample.get('scenarios', []))}")

if grouped["api_specifications"]:
    print("\n🔹 API Specification:")
    sample = grouped["api_specifications"][0]
    print(f"   ID: {sample.get('id')}")
    print(f"   Endpoint: {sample.get('method')} {sample.get('endpoint')}")

if grouped["ui_specifications"]:
    print("\n🔹 UI Specification:")
    sample = grouped["ui_specifications"][0]
    print(f"   ID: {sample.get('id')}")
    print(f"   Title: {sample.get('title')}")

print(f"\n{'='*60}")
print("✅ Requirement extraction complete!")
print(f"{'='*60}")