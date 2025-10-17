import json
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# ============================================
# ENHANCED DATA STRUCTURES WITH SRS TRACEABILITY
# ============================================

class TestCaseType(Enum):
    """Complete test case types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    EDGE = "edge"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"
    COMPATIBILITY = "compatibility"
    API = "api"
    DATA_INTEGRITY = "data_integrity"
    RELIABILITY = "reliability"


@dataclass
class TestCase:
    """Enhanced test case with SRS traceability"""
    test_id: str
    requirement_id: str
    test_type: str
    test_title: str
    description: str
    preconditions: List[str]
    test_steps: List[str]
    expected_result: str
    test_data: Dict[str, Any]
    priority: str
    generation_phase: str
    srs_section: str = ""
    depends_on: List[str] = field(default_factory=list)


# ============================================
# SRS-SPECIFIC REQUIREMENT ANALYZER
# ============================================

class SRSRequirementAnalyzer:
    """Analyzes requirements to extract SRS-specific details"""
    
    # Known SRS-specific patterns for critical requirements
    SRS_PATTERNS = {
        "FR6": {
            "must_test": [
                "Search by price (min-max range)",
                "Search by destination",
                "Search by restaurant type",
                "Search by specific dish",
                "Free-text search",
                "COMBINED multi-criteria search (Price + Distance + Type)"
            ]
        },
        "FR7": {
            "must_test": [
                "Maximum 100 results displayed on map",
                "Default zoom level verification",
                "Information link on each pin",
                "Filtering menu button present"
            ]
        },
        "FR8": {
            "must_test": [
                "Maximum 100 results in list view",
                "Sorting when search by price: price → distance → type → dish",
                "Sorting when NOT by price: distance → price → type → dish",
                "Scrollable results",
                "Filtering menu button"
            ]
        },
        "FR11": {
            "must_test": [
                "Picture displayed",
                "Name, address, phone, email displayed",
                "Type of food and average price shown",
                "Full menu with dish names, descriptions, prices"
            ]
        },
        "FR12": {
            "must_test": [
                "Minimum and maximum price input",
                "Results displayed in LIST VIEW by default (not map)",
                "Edge case: min = max",
                "Only integers accepted (reference FR14)"
            ]
        },
        "FR24": {
            "must_test": [
                "MANDATORY fields: average price, address, email, phone, restaurant name",
                "OPTIONAL fields: description, menu, type, picture, mobile",
                "Menu requires: dish name, description, price",
                "Form submission with all mandatory fields",
                "Form submission with missing mandatory field (should fail)"
            ]
        }
    }
    
    @classmethod
    def get_srs_specifics(cls, req_id: str) -> List[str]:
        """Get SRS-specific requirements for a given requirement ID"""
        return cls.SRS_PATTERNS.get(req_id, {}).get("must_test", [])
    
    @classmethod
    def has_srs_specifics(cls, req_id: str) -> bool:
        """Check if requirement has SRS-specific patterns"""
        return req_id in cls.SRS_PATTERNS


# ============================================
# ENHANCED SRS-AWARE PROMPT GENERATOR
# ============================================

class EnhancedSRSPromptGenerator:
    """Enhanced prompt generator with SRS-specific requirements"""
    
    @staticmethod
    def build_srs_enhanced_prompt(requirement: Dict, domain: str, test_types: List[str]) -> str:
        """Generate prompt with SRS-specific enhancements"""
        
        req_id = requirement.get('id', '')
        req_description = requirement.get('description', '')
        req_rationale = requirement.get('rationale', '')
        dependencies = requirement.get('dependencies', [])
        test_types_str = ", ".join(test_types)
        
        # Check if this requirement has SRS-specific patterns
        srs_specifics = SRSRequirementAnalyzer.get_srs_specifics(req_id)
        
        srs_specific_section = ""
        if srs_specifics:
            srs_specific_section = f"""
⚠️ CRITICAL SRS-SPECIFIC REQUIREMENTS FOR {req_id}:
The SRS document EXPLICITLY requires testing these scenarios:
{chr(10).join([f"  {i+1}. {spec}" for i, spec in enumerate(srs_specifics)])}

YOU MUST generate test cases that DIRECTLY test these specific behaviors.
DO NOT generate generic tests. Each test must address one of the above requirements.
"""
        
        prompt = f"""You are a senior QA engineer generating SPECIFIC, DETAILED test cases from an SRS document.

CRITICAL RULES:
1. NO placeholder test cases like "Execute test action" or "Input test data"
2. NO generic "Verify result" steps
3. Use ACTUAL test data from the requirement (URLs, usernames, fields, etc.)
4. Reference SPECIFIC behaviors mentioned in the requirement
5. Each test step must be executable by a human tester
6. Expected results must be MEASURABLE and SPECIFIC

{srs_specific_section}

REQUIREMENT:
ID: {req_id}
Title: {requirement['title']}
Description: {req_description}
Rationale: {req_rationale}
Dependencies: {', '.join(dependencies) if dependencies else 'None'}

DOMAIN: {domain}

ANALYZE THE REQUIREMENT DESCRIPTION CAREFULLY:
- Extract specific data fields mentioned (price, distance, types, etc.)
- Identify exact behaviors (sorting order, display limits, mandatory fields)
- Note any quantitative requirements (max results, time limits, ranges)
- Identify integration points with other features
- Look for "maximum", "minimum", "sorted by", "displayed in", "mandatory", "optional" keywords

GENERATE {len(test_types)} TEST CASES - ONE FOR EACH TYPE:
{test_types_str}

FOR EACH TEST TYPE:
- POSITIVE: Test the main success scenario with real, specific data from the requirement
- NEGATIVE: Test error handling with invalid inputs specific to this requirement
- EDGE: Test boundary conditions mentioned in the requirement (e.g., exactly 100 results, min=max)
- INTEGRATION: Test interaction with dependencies listed above
- PERFORMANCE: Test any performance criteria mentioned (time limits, data limits)
- SECURITY: Test authentication, authorization, data protection for this feature

OUTPUT FORMAT (JSON array only, no markdown):
[
  {{
    "test_type": "positive",
    "test_title": "Specific test title describing exact scenario",
    "description": "Clear explanation of what this test validates",
    "preconditions": ["Specific precondition 1", "Specific precondition 2", "Specific precondition 3"],
    "test_steps": [
      "Navigate to [specific page/URL]",
      "Enter [specific field]: '[actual value]'",
      "Click '[specific button name]' button",
      "Verify [specific element] displays [exact expected value]"
    ],
    "test_data": {{
      "field1": "actual_value",
      "field2": "actual_value",
      "expected_count": 10
    }},
    "expected_result": "Specific, measurable outcome with exact values/behaviors from requirement",
    "priority": "High",
    "srs_section": "Section number if mentioned in requirement",
    "depends_on": {json.dumps(dependencies)}
  }}
]

EXAMPLES OF GOOD VS BAD:
❌ BAD: "Execute test action"
✅ GOOD: "Click 'Search' button with price range 100-500"

❌ BAD: "Verify result"
✅ GOOD: "Verify results are sorted by distance first, then by price, displaying maximum 100 restaurants"

❌ BAD: "test_data": "test_input"
✅ GOOD: "test_data": {{"min_price": 100, "max_price": 500, "distance_km": 5}}

❌ BAD (for FR24): "Enter price range and search"
✅ GOOD (for FR24): "Fill restaurant form: name='Test Restaurant', address='123 Main St', email='test@restaurant.com', phone='555-1234', average_price=25, then click 'Submit'"

GENERATE EXACTLY {len(test_types)} SPECIFIC, DETAILED TEST CASES.
Output ONLY JSON array, no explanations.
"""
        return prompt
    
    @staticmethod
    def build_deep_srs_prompt(requirement: Dict, domain: str, test_type: str) -> str:
        """Generate deep prompt for comprehensive testing with SRS focus"""
        
        req_id = requirement.get('id', '')
        req_description = requirement.get('description', '')
        req_rationale = requirement.get('rationale', '')
        dependencies = requirement.get('dependencies', [])
        
        # Get SRS specifics
        srs_specifics = SRSRequirementAnalyzer.get_srs_specifics(req_id)
        
        srs_reminder = ""
        if srs_specifics:
            srs_reminder = f"""
⚠️ CRITICAL: For {req_id}, the SRS explicitly requires:
{chr(10).join([f"  • {spec}" for spec in srs_specifics])}

Your {test_type.upper()} test cases MUST address these specific SRS requirements.
"""
        
        # Type-specific guidance
        type_guidance = {
            "usability": """
Focus on USER EXPERIENCE aspects mentioned in requirement:
- UI element clarity and placement
- Task completion time (if specified)
- Error message clarity
- Navigation intuitiveness
- Mobile vs desktop experience differences
""",
            "compatibility": """
Focus on PLATFORM/DEVICE compatibility:
- iOS vs Android differences
- Different screen sizes/resolutions
- Browser compatibility (if web-based)
- GPS/location accuracy across devices
- Network type variations (WiFi, 4G, 5G)
""",
            "api": """
Focus on API INTEGRATION details:
- Specific endpoint paths and methods
- Request/response schema validation
- HTTP status codes for different scenarios
- Authentication/authorization headers
- Rate limiting if mentioned
- Response time requirements
""",
            "data_integrity": """
Focus on DATA CONSISTENCY:
- CRUD operations validation
- Calculation accuracy (if formulas mentioned)
- Mandatory vs optional field validation
- Data type constraints (integer, string, etc.)
- Foreign key relationships
- Transaction integrity
""",
            "reliability": """
Focus on FAILURE RECOVERY:
- Network interruption handling
- App crash recovery
- GPS signal loss behavior
- Session timeout handling
- Data persistence across crashes
"""
        }
        
        guidance = type_guidance.get(test_type, "Focus on requirement-specific scenarios")
        
        prompt = f"""You are a senior QA engineer specializing in {test_type.upper()} testing.

{srs_reminder}

REQUIREMENT DETAILS:
ID: {req_id}
Title: {requirement['title']}
Description: {req_description}
Rationale: {req_rationale}
Dependencies: {', '.join(dependencies) if dependencies else 'None'}

DOMAIN: {domain}

{guidance}

CRITICAL: EXTRACT SPECIFIC DETAILS FROM REQUIREMENT DESCRIPTION:
- Look for field names, data types, ranges, limits
- Identify specific UI elements, pages, or workflows
- Note any numerical requirements (max results, time limits, etc.)
- Identify sorting/filtering behaviors
- Note mandatory vs optional elements

GENERATE 2-3 DETAILED {test_type.upper()} TEST CASES.

Each test case MUST:
1. Use ACTUAL data from the requirement (not generic "test_input")
2. Have SPECIFIC, executable test steps (not "Execute test action")
3. Have MEASURABLE expected results (not "Feature works as expected")
4. Reference specific UI elements, fields, or behaviors from requirement
5. Include quantitative expectations if mentioned in requirement

OUTPUT FORMAT (JSON array):
[
  {{
    "test_type": "{test_type}",
    "test_title": "Specific title describing exact {test_type} scenario",
    "description": "What this test validates in the requirement",
    "preconditions": ["Specific setup 1", "Specific setup 2", "Specific setup 3"],
    "test_steps": [
      "Specific step 1 with actual values",
      "Specific step 2 with actual actions",
      "Specific step 3 with exact verification",
      "Specific step 4 with measurable check",
      "Specific step 5 with concrete outcome"
    ],
    "test_data": {{
      "actual_field1": "concrete_value1",
      "actual_field2": "concrete_value2",
      "expected_limit": 100
    }},
    "expected_result": "Specific, measurable outcome with exact values, counts, or behaviors from requirement",
    "priority": "High or Medium based on criticality",
    "srs_section": "Section reference if available",
    "depends_on": {json.dumps(dependencies)}
  }}
]

IMPORTANT: If requirement mentions specific values (e.g., "maximum 100 results", "within 2 seconds", "sorted by distance then price"), include these EXACT values in test cases.

Output ONLY JSON array.
"""
        return prompt


# ============================================
# ENHANCED VALIDATOR WITH STRICTER CHECKS
# ============================================

class EnhancedSRSValidator:
    """Enhanced validation to prevent placeholder test cases"""
    
    FORBIDDEN_PHRASES = [
        "execute test action",
        "input test data",
        "verify result",
        "test_input",
        "test_value",
        "execute action",
        "perform action",
        "check result",
        "validate result",
        "works as expected",
        "behaves as expected"
    ]
    
    @staticmethod
    def validate(test_cases: List[Dict], requirement: Dict, is_comprehensive: bool = False) -> List[Dict]:
        """Strict validation to prevent placeholder test cases"""
        validated = []
        req_id = requirement.get('id', '')
        
        # Get SRS specifics for this requirement
        srs_specifics = SRSRequirementAnalyzer.get_srs_specifics(req_id)
        
        for idx, tc in enumerate(test_cases):
            # 1. Required fields check
            if not tc.get('test_title') or not tc.get('test_steps'):
                print(f"    ⚠️ Test {idx+1}: Missing title or steps")
                continue
            
            # 2. Check for placeholder phrases in steps
            has_placeholder = False
            steps_str = ' '.join(str(s).lower() for s in tc.get('test_steps', []))
            
            for phrase in EnhancedSRSValidator.FORBIDDEN_PHRASES:
                if phrase in steps_str:
                    print(f"    ⚠️ Test {idx+1}: Contains placeholder phrase '{phrase}'")
                    has_placeholder = True
                    break
            
            if has_placeholder:
                continue
            
            # 3. Check expected result for forbidden phrases
            expected = tc.get('expected_result', '').lower()
            for phrase in ["works as expected", "behaves as expected", "feature works"]:
                if phrase in expected:
                    print(f"    ⚠️ Test {idx+1}: Generic expected result")
                    has_placeholder = True
                    break
            
            if has_placeholder:
                continue
            
            # 4. Normalize test steps
            if isinstance(tc['test_steps'], str):
                tc['test_steps'] = [s.strip() for s in tc['test_steps'].split('\n') if s.strip()]
            
            # 5. Ensure minimum quality steps
            min_steps = 5 if is_comprehensive else 4
            if len(tc['test_steps']) < min_steps:
                print(f"    ⚠️ Test {idx+1}: Only {len(tc['test_steps'])} steps (min: {min_steps})")
                continue
            
            # 6. Normalize preconditions
            if isinstance(tc.get('preconditions'), str):
                tc['preconditions'] = [s.strip() for s in tc['preconditions'].split(',') if s.strip()]
            elif not isinstance(tc.get('preconditions'), list):
                tc['preconditions'] = ["Application accessible", "User has necessary permissions"]
            
            # 7. Normalize test data
            if isinstance(tc.get('test_data'), str):
                try:
                    tc['test_data'] = json.loads(tc['test_data'])
                except:
                    tc['test_data'] = {"data": tc['test_data']}
            elif not isinstance(tc.get('test_data'), dict):
                tc['test_data'] = {}
            
            # 8. Check test data quality
            test_data_str = str(tc.get('test_data', {})).lower()
            if 'test_input' in test_data_str or 'test_value' in test_data_str:
                print(f"    ⚠️ Test {idx+1}: Generic test data")
                # Don't skip, but flag
            
            # 9. Check expected result length
            if len(expected) < 30:
                print(f"    ⚠️ Test {idx+1}: Expected result too short ({len(expected)} chars)")
                continue
            
            # 10. For critical SRS requirements, check if test addresses SRS specifics
            if srs_specifics and is_comprehensive:
                # Check if test case title or description references SRS-specific patterns
                tc_text = (tc.get('test_title', '') + ' ' + tc.get('description', '')).lower()
                addresses_srs = any(
                    keyword.lower() in tc_text 
                    for spec in srs_specifics 
                    for keyword in spec.split()[:3]  # Check first 3 words of each spec
                )
                if not addresses_srs:
                    print(f"    ℹ️ Test {idx+1}: May not address SRS-specific requirements for {req_id}")
            
            # 11. Add requirement metadata
            tc['requirement_id'] = requirement['id']
            tc['description'] = tc.get('description', f"Test case for {requirement['title']}")
            
            # 12. Normalize priority
            if tc.get('priority') not in ['High', 'Medium', 'Low']:
                tc['priority'] = 'High' if tc.get('test_type') in ['positive', 'negative', 'security'] else 'Medium'
            
            # 13. Add SRS section
            if not tc.get('srs_section'):
                tc['srs_section'] = requirement.get('srs_section', '')
            
            # 14. Add dependencies
            if not tc.get('depends_on'):
                tc['depends_on'] = requirement.get('dependencies', [])
            
            validated.append(tc)
        
        return validated


# ============================================
# ENHANCED HYBRID ENGINE
# ============================================

class OptimizedHybridEngine:
    """Optimized SRS-aware hybrid test generation engine"""
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b-instruct"):
        self.model_name = model_name
        self.prompt_gen = EnhancedSRSPromptGenerator()
        self.validator = EnhancedSRSValidator()
        self.test_counter = 1
        self.max_workers = 3
        
        print(f"\n{'='*80}")
        print(f"⚡ OPTIMIZED SRS-AWARE HYBRID TEST GENERATION ENGINE")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"✅ Enforces SRS-specific test cases")
        print(f"✅ Blocks placeholder/template tests")
        print(f"✅ Deduplicates overlapping requirements")
        print(f"✅ Targets critical SRS gaps (FR6, FR8, FR24, etc.)")
        print(f"{'='*80}\n")
        
        try:
            ollama.list()
            ollama.generate(model=self.model_name, prompt="test", options={'num_predict': 1})
            print("✅ Model loaded and ready\n")
        except Exception as e:
            print(f"⚠️ Ollama issue: {e}\n")
    
    def _call_llm(self, prompt: str, is_comprehensive: bool = False) -> Optional[str]:
        """Call LLM with appropriate settings"""
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.3 if is_comprehensive else 0.2,
                    'top_p': 0.9,
                    'num_predict': 6000 if is_comprehensive else 4000,
                    'repeat_penalty': 1.1
                }
            )
            return response['response']
        except Exception as e:
            print(f"  ⚠️ LLM error: {e}")
            return None
    
    def _parse_json(self, response: str) -> List[Dict]:
        """Parse JSON response"""
        if not response:
            return []
        
        try:
            cleaned = response.strip()
            if cleaned.startswith('```'):
                lines = cleaned.split('\n')
                cleaned = '\n'.join(line for line in lines if not line.strip().startswith('```'))
            
            start = cleaned.find('[')
            end = cleaned.rfind(']') + 1
            
            if start == -1 or end == 0:
                return []
            
            json_str = cleaned[start:end]
            return json.loads(json_str)
        except Exception as e:
            print(f"  ⚠️ JSON parse error: {e}")
            return []
    
    def generate_fast_batch(self, requirement: Dict, domain: str, test_types: List[str]) -> List[TestCase]:
        """Phase 1: Fast batch with SRS awareness"""
        
        prompt = self.prompt_gen.build_srs_enhanced_prompt(requirement, domain, test_types)
        response = self._call_llm(prompt, is_comprehensive=False)
        generated = self._parse_json(response)
        
        print(f"    Generated {len(generated)} raw test cases")
        
        validated = self.validator.validate(generated, requirement, is_comprehensive=False)
        
        print(f"    ✅ Validated {len(validated)} test cases (rejected {len(generated) - len(validated)})")
        
        test_cases = []
        for tc in validated:
            test_case = TestCase(
                test_id=f"TC_{requirement['id']}_{self.test_counter:03d}",
                requirement_id=tc['requirement_id'],
                test_type=tc.get('test_type', 'positive'),
                test_title=tc['test_title'],
                description=tc['description'],
                preconditions=tc['preconditions'],
                test_steps=tc['test_steps'],
                expected_result=tc['expected_result'],
                test_data=tc['test_data'],
                priority=tc['priority'],
                generation_phase='fast_batch',
                srs_section=tc.get('srs_section', ''),
                depends_on=tc.get('depends_on', [])
            )
            test_cases.append(test_case)
            self.test_counter += 1
        
        return test_cases
    
    def generate_comprehensive(self, requirement: Dict, domain: str, test_types: List[str]) -> List[TestCase]:
        """Phase 2: Comprehensive with SRS details"""
        
        all_test_cases = []
        
        for test_type in test_types:
            print(f"      Generating {test_type} tests...")
            
            prompt = self.prompt_gen.build_deep_srs_prompt(requirement, domain, test_type)
            response = self._call_llm(prompt, is_comprehensive=True)
            generated = self._parse_json(response)
            
            validated = self.validator.validate(generated, requirement, is_comprehensive=True)
            
            print(f"      ✅ {len(validated)} {test_type} tests validated")
            
            for tc in validated:
                test_case = TestCase(
                    test_id=f"TC_{requirement['id']}_{self.test_counter:03d}",
                    requirement_id=tc['requirement_id'],
                    test_type=tc.get('test_type', test_type),
                    test_title=tc['test_title'],
                    description=tc['description'],
                    preconditions=tc['preconditions'],
                    test_steps=tc['test_steps'],
                    expected_result=tc['expected_result'],
                    test_data=tc['test_data'],
                    priority=tc['priority'],
                    generation_phase='comprehensive',
                    srs_section=tc.get('srs_section', ''),
                    depends_on=tc.get('depends_on', [])
                )
                all_test_cases.append(test_case)
                self.test_counter += 1
        
        return all_test_cases
    
    def generate_parallel_fast(self, requirements: List[Dict], domain: str, test_types: List[str]) -> List[TestCase]:
        """Parallel fast batch processing"""
        
        all_test_cases = []
        total = len(requirements)
        
        print(f"🚀 PHASE 1: SRS-aware fast batch for {total} requirements...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.generate_fast_batch, req, domain, test_types): req 
                for req in requirements
            }
            
            completed = 0
            for future in as_completed(futures):
                req = futures[future]
                try:
                    test_cases = future.result()
                    all_test_cases.extend(test_cases)
                    completed += 1
                    
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (total - completed) * avg_time
                    
                    print(f"  ✅ {completed}/{total} | {req['id']} | "
                          f"{len(test_cases)} tests | ETA: {remaining/60:.1f}m")
                except Exception as e:
                    print(f"  ❌ {req['id']}: {e}")
                    completed += 1
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Phase 1 completed in {elapsed/60:.1f} minutes")
        print(f"📊 Generated {len(all_test_cases)} quality test cases\n")
        
        return all_test_cases
    
    def generate_sequential_comprehensive(self, requirements: List[Dict], domain: str, test_types: List[str]) -> List[TestCase]:
        """Sequential comprehensive processing"""
        
        all_test_cases = []
        total = len(requirements)
        
        print(f"🔍 PHASE 2: Comprehensive SRS-specific generation for {total} critical requirements...")
        start_time = time.time()
        
        for idx, req in enumerate(requirements, 1):
            print(f"\n  [{idx}/{total}] {req['id']}: {req['title'][:60]}...")
            
            # Flag if this is a critical SRS requirement
            if SRSRequirementAnalyzer.has_srs_specifics(req['id']):
                print(f"      🎯 SRS-critical requirement - enhanced testing")
            
            try:
                test_cases = self.generate_comprehensive(req, domain, test_types)
                all_test_cases.extend(test_cases)
                
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (total - idx) * avg_time
                
                print(f"  ✅ Total {len(test_cases)} comprehensive tests | ETA: {remaining/60:.1f}m")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Phase 2 completed in {elapsed/60:.1f} minutes")
        print(f"📊 Generated {len(all_test_cases)} comprehensive test cases\n")
        
        return all_test_cases


# ============================================
# OPTIMIZED ORCHESTRATOR WITH DEDUPLICATION
# ============================================

class OptimizedHybridGenerator:
    """Optimized SRS-aware hybrid test generator with deduplication"""
    
    def __init__(self, input_file: str, model_name: str = "qwen2.5-coder:7b-instruct"):
        self.input_file = input_file
        self.engine = OptimizedHybridEngine(model_name)
        self.data = None
        self.phase1_test_cases = []
        self.phase2_test_cases = []
        
        self.phase1_types = ['positive', 'negative', 'edge', 'integration', 'performance', 'security']
        self.phase2_types = ['usability', 'compatibility', 'api', 'data_integrity', 'reliability']
        
        print(f"Phase 1 types: {self.phase1_types}")
        print(f"Phase 2 types: {self.phase2_types}\n")
    
    def load_data(self):
        """Load input data"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Calculate overlap
        total_instances = sum(len(chunk['requirements']) for chunk in self.data['chunks'])
        unique_count = len(self._deduplicate_requirements())
        overlap_count = total_instances - unique_count
        
        print(f"{'='*80}")
        print(f"📄 LOADED DATA")
        print(f"{'='*80}")
        print(f"Total requirement instances: {total_instances}")
        print(f"Unique requirements: {unique_count}")
        if overlap_count > 0:
            print(f"⚠️ Overlap detected: {overlap_count} duplicate instances ({overlap_count/total_instances*100:.1f}%)")
            print(f"✅ Deduplication will be applied")
        print(f"Chunks: {self.data['metadata']['total_chunks']}")
        print(f"Domain: {self.data['domain_classification']['primary_domain']}")
        print(f"{'='*80}\n")
    
    def _deduplicate_requirements(self) -> List[Dict]:
        """Extract unique requirements from all chunks (fixes 22% overlap issue)"""
        all_requirements = []
        seen_req_ids = set()
        
        for chunk in self.data['chunks']:
            for req in chunk['requirements']:
                if req['id'] not in seen_req_ids:
                    all_requirements.append(req)
                    seen_req_ids.add(req['id'])
        
        return all_requirements
    
    def identify_critical_requirements(self, top_n: int = 15) -> List[Dict]:
        """Identify critical requirements (with deduplication)"""
        
        # Get unique requirements first
        all_requirements = self._deduplicate_requirements()
        
        critical_reqs = []
        
        # Priority 1: SRS-critical requirements (FR6, FR8, FR24, etc.)
        for req in all_requirements:
            if SRSRequirementAnalyzer.has_srs_specifics(req['id']):
                critical_reqs.append(req)
        
        # Priority 2: Requirements with dependencies
        for req in all_requirements:
            if req.get('dependencies') and len(req['dependencies']) > 0:
                if req not in critical_reqs:
                    critical_reqs.append(req)
        
        # Priority 3: First N requirements (core features)
        for req in all_requirements[:top_n]:
            if req not in critical_reqs:
                critical_reqs.append(req)
        
        critical_reqs = critical_reqs[:top_n]
        
        print(f"{'='*80}")
        print(f"🎯 CRITICAL REQUIREMENTS FOR PHASE 2")
        print(f"{'='*80}")
        print(f"Total unique requirements: {len(all_requirements)}")
        print(f"Selected for Phase 2: {len(critical_reqs)}")
        for req in critical_reqs:
            deps = ', '.join(req.get('dependencies', [])) if req.get('dependencies') else 'None'
            srs_flag = "🎯 SRS-Critical" if SRSRequirementAnalyzer.has_srs_specifics(req['id']) else ""
            print(f"  • {req['id']}: {req['title'][:45]} | Deps: {deps} {srs_flag}")
        print(f"{'='*80}\n")
        
        return critical_reqs
    
    def generate_phase1(self):
        """Phase 1: Fast batch with deduplication"""
        domain = self.data['domain_classification']['primary_domain']
        
        # Deduplicate requirements (fixes 22% overlap)
        all_requirements = self._deduplicate_requirements()
        
        print(f"✅ Deduplicated: {len(all_requirements)} unique requirements\n")
        
        self.phase1_test_cases = self.engine.generate_parallel_fast(
            all_requirements, domain, self.phase1_types
        )
    
    def generate_phase2(self, critical_requirements: List[Dict]):
        """Phase 2: Comprehensive"""
        domain = self.data['domain_classification']['primary_domain']
        
        self.phase2_test_cases = self.engine.generate_sequential_comprehensive(
            critical_requirements, domain, self.phase2_types
        )
    
    def save_results(self, output_prefix: str = "optimized_test_cases"):
        """Save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_test_cases = self.phase1_test_cases + self.phase2_test_cases
        
        # JSON
        json_data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "model": self.engine.model_name,
                "total_test_cases": len(all_test_cases),
                "phase1_count": len(self.phase1_test_cases),
                "phase2_count": len(self.phase2_test_cases),
                "validation": "SRS-aware with placeholder blocking and deduplication",
                "improvements": [
                    "Deduplication applied (fixed 22% overlap)",
                    "SRS-specific prompts for critical requirements",
                    "Enhanced validation with stricter checks"
                ]
            },
            "phase1_test_cases": [asdict(tc) for tc in self.phase1_test_cases],
            "phase2_test_cases": [asdict(tc) for tc in self.phase2_test_cases]
        }
        
        json_file = f"{output_prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✅ Saved JSON: {json_file}")
        
        # Excel
        df = pd.DataFrame([asdict(tc) for tc in all_test_cases])
        
        df['preconditions'] = df['preconditions'].apply(
            lambda x: '\n'.join([f"• {p}" for p in x]) if isinstance(x, list) else x
        )
        df['test_steps'] = df['test_steps'].apply(
            lambda x: '\n'.join([f"{i+1}. {s}" for i, s in enumerate(x)]) if isinstance(x, list) else x
        )
        df['test_data'] = df['test_data'].apply(
            lambda x: json.dumps(x, indent=2) if isinstance(x, dict) else x
        )
        df['depends_on'] = df['depends_on'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
        
        excel_file = f"{output_prefix}_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            overview = pd.DataFrame([{
                'Total Tests': len(all_test_cases),
                'Phase 1': len(self.phase1_test_cases),
                'Phase 2': len(self.phase2_test_cases),
                'Deduplicated': 'Yes',
                'SRS-Aware': 'Yes',
                'High Priority': len([t for t in all_test_cases if t.priority == 'High'])
            }])
            overview.to_excel(writer, sheet_name='Overview', index=False)
            
            df.to_excel(writer, sheet_name='All Tests', index=False)
            
            phase1_df = df[df['generation_phase'] == 'fast_batch']
            if not phase1_df.empty:
                phase1_df.to_excel(writer, sheet_name='Phase 1 - Fast', index=False)
            
            phase2_df = df[df['generation_phase'] == 'comprehensive']
            if not phase2_df.empty:
                phase2_df.to_excel(writer, sheet_name='Phase 2 - Deep', index=False)
            
            all_types = self.phase1_types + self.phase2_types
            for test_type in all_types:
                type_df = df[df['test_type'] == test_type]
                if not type_df.empty:
                    type_df.to_excel(writer, sheet_name=test_type.capitalize()[:31], index=False)
        
        print(f"✅ Saved Excel: {excel_file}")
        
        # Summary
        summary_file = f"{output_prefix}_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OPTIMIZED SRS-AWARE HYBRID TEST GENERATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total Tests: {len(all_test_cases)}\n\n")
            
            f.write("IMPROVEMENTS APPLIED:\n")
            f.write(" Deduplication (fixed 22% overlap)\n")
            f.write(" SRS-specific prompts for FR6, FR8, FR24, etc.\n")
            f.write("  Enhanced validation blocking placeholders\n")
            f.write("  Priority targeting for critical SRS gaps\n\n")
            
            f.write("PHASE 1 - FAST BATCH:\n")
            f.write(f"  Tests: {len(self.phase1_test_cases)}\n")
            f.write(f"  Types: {', '.join(self.phase1_types)}\n\n")
            
            f.write("PHASE 2 - COMPREHENSIVE:\n")
            f.write(f"  Tests: {len(self.phase2_test_cases)}\n")
            f.write(f"  Types: {', '.join(self.phase2_types)}\n\n")
            
            f.write("BY TYPE:\n")
            for test_type in self.phase1_types + self.phase2_types:
                count = len([t for t in all_test_cases if t.test_type == test_type])
                f.write(f"  {test_type}: {count}\n")
        
        print(f"✅ Saved Summary: {summary_file}")
        
        return json_file, excel_file, summary_file
    
    def show_stats(self):
        """Display statistics"""
        all_test_cases = self.phase1_test_cases + self.phase2_test_cases
        
        print(f"\n{'='*80}")
        print(f"📊 FINAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total Test Cases: {len(all_test_cases)}")
        print(f"\nBy Phase:")
        print(f"  Phase 1: {len(self.phase1_test_cases)}")
        print(f"  Phase 2: {len(self.phase2_test_cases)}")
        print(f"\nQuality Metrics:")
        print(f"  High priority: {len([t for t in all_test_cases if t.priority == 'High'])}")
        print(f"  With dependencies: {len([t for t in all_test_cases if t.depends_on])}")
        print(f"{'='*80}\n")


# ============================================
# MAIN
# ============================================

def main():
    INPUT_FILE = "../03_Chunking_Domain_Understanding/chunked_requirements_with_domain.json"
    OUTPUT_PREFIX = "../04_AI_powered_TestCaseGeneration/optimized_test_cases"
    MODEL = "qwen2.5-coder:7b-instruct"
    NUM_CRITICAL = 15
    
    print("="*80)
    print(" " * 10 + "⚡ OPTIMIZED SRS-AWARE HYBRID TEST GENERATION")
    print("="*80)
    print("✅ Deduplication (fixes 22% overlap)")
    print("✅ SRS-specific prompts (FR6, FR8, FR24, etc.)")
    print("✅ Enhanced validation (blocks placeholders)")
    print("✅ Priority targeting (critical SRS gaps)")
    print("="*80 + "\n")
    
    generator = OptimizedHybridGenerator(INPUT_FILE, model_name=MODEL)
    
    print("[1/4] Loading data...")
    generator.load_data()
    
    print("[2/4] Identifying critical requirements...")
    critical_reqs = generator.identify_critical_requirements(top_n=NUM_CRITICAL)
    
    print(f"[3/4] Phase 1: Optimized fast batch...")
    phase1_start = time.time()
    generator.generate_phase1()
    phase1_time = time.time() - phase1_start
    
    print(f"[4/4] Phase 2: Comprehensive deep generation...")
    phase2_start = time.time()
    generator.generate_phase2(critical_reqs)
    phase2_time = time.time() - phase2_start
    
    print("\n[5/5] Saving results...")
    json_f, excel_f, summary_f = generator.save_results(OUTPUT_PREFIX)
    
    generator.show_stats()
    
    total_time = phase1_time + phase2_time
    print("="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"Phase 1: {phase1_time/60:.1f}m | Phase 2: {phase2_time/60:.1f}m | Total: {total_time/60:.1f}m")
    print(f"Tests: {len(generator.phase1_test_cases) + len(generator.phase2_test_cases)}")
    print(f"\nFiles:\n  • {json_f}\n  • {excel_f}\n  • {summary_f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
