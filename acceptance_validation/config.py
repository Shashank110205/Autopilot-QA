# config.py — Autopilot-QA CAU Layer Configuration
# All patterns and label maps live here. Zero hardcoding in logic files.

# ---------------------------------------------------------------------------
# Requirement ID detection
# ---------------------------------------------------------------------------
# Default fallback pattern — auto-detected at runtime from the UAT PDF.
# Matches: FR1, UC001, REQ1, QR13, SRS102, etc.
REQ_ID_PATTERN = r'\b([A-Z]{1,5}\d+)\b'

# ---------------------------------------------------------------------------
# UAT test-case header pattern
# ---------------------------------------------------------------------------
# Matches lines like:
#   3.1 UAT-UC1-01: Download and Registration
#   UAT-RO-02 – Role Operations
#   TC-001 - Verify login
UAT_HEADER_PATTERN = r'(?:[\d.]+\s+)?((?:UAT|TC|AT)-[A-Z0-9]+-\d+)\s*[:\-\u2013\u2014]\s*(.+)'

# ---------------------------------------------------------------------------
# Actor / Use-Case block header (optional — used to attach actor_class)
# ---------------------------------------------------------------------------
# Matches: "Use Case 1: Actor Name" or "UC-1 Customer Operations"
ACTOR_BLOCK_PATTERN = r'(?:use\s*case\s*[\d.]+|uc-?\d+)[:\s]+(.+)'

# ---------------------------------------------------------------------------
# Field label map
# Keys  → normalised lowercase label as it appears in the UAT PDF
# Values → canonical field name used in the CAU object
# ---------------------------------------------------------------------------
FIELD_LABEL_MAP = {
    # Requirement references
    'requirement ids':          'req_ids',
    'requirement id':           'req_ids',
    'requirement':              'req_ids',
    'requirements':             'req_ids',
    'linked requirements':      'req_ids',
    # Description / objective
    'description':              'description',
    'objective':                'description',
    'test objective':           'description',
    'purpose':                  'description',
    # Pre-conditions
    'pre-condition':            'precondition',
    'precondition':             'precondition',
    'pre-conditions':           'precondition',
    'preconditions':            'precondition',
    'prerequisites':            'precondition',
    # Test steps
    'test steps':               'test_steps',
    'test step':                'test_steps',
    'steps':                    'test_steps',
    'test procedure':           'test_steps',
    # Expected result
    'expected result':          'expected_result',
    'expected results':         'expected_result',
    'expected outcome':         'expected_result',
    # Actual result
    'actual result':            'actual_result',
    'actual results':           'actual_result',
    'actual outcome':           'actual_result',
    # Status
    'status':                   'status',
    'test status':              'status',
    'result':                   'status',
    # Observations
    'tester observations':      'observations',
    'observations':             'observations',
    'comments':                 'observations',
    'remarks':                  'observations',
}

# ---------------------------------------------------------------------------
# Valid UAT status tokens (case-insensitive match)
# ---------------------------------------------------------------------------
STATUS_VALUES = {'PASS', 'FAIL', 'PARTIAL'}

# ---------------------------------------------------------------------------
# Coverage classification rules (evaluated in order — first match wins)
# ---------------------------------------------------------------------------
# Each entry: (condition_fn(cau_obj) -> bool, classification_string)
# Evaluated by reporter.py — do not hardcode classifications elsewhere.
COVERAGE_RULES = [
    # No CRU match at all
    (lambda c: not c.get('linked_crus'),                                   'NO_CRU_MATCH'),
    # CRUs found but no test cases
    (lambda c: c.get('linked_crus') and not c.get('linked_test_cases'),    'NO_TEST_CASE'),
    # Explicit FAIL status
    (lambda c: c.get('status', '').upper() == 'FAIL',                      'FAILED_COVERAGE'),
    # Partial status
    (lambda c: c.get('status', '').upper() == 'PARTIAL',                   'PARTIAL_COVERAGE'),
    # Everything present and PASS
    (lambda c: True,                                                         'FULL_COVERAGE'),
]

# ---------------------------------------------------------------------------
# Output paths (overridable via runner.py CLI args)
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = 'output'
CAU_JSON_FILENAME  = 'cau_output.json'
HTML_FILENAME      = 'cau_traceability_report.html'

# ---------------------------------------------------------------------------
# Pipeline metadata
# ---------------------------------------------------------------------------
PIPELINE_NAME    = 'Autopilot-QA CAU Layer'
PIPELINE_VERSION = '1.0'