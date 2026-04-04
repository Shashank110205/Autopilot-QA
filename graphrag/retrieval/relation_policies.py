TASK_RELATION_WHITELISTS = {
    "test_generation": {
        "forward": ["SUPPORTED_BY", "PARENT_OF", "DECOMPOSED_TO", "TESTS", "EVIDENCE_FOR"],
        "reverse": ["PARENT_OF", "TESTS"],
    },
    "debug": {
        "forward": ["TESTS", "SUPPORTED_BY", "EXECUTED_AS", "RAISED_AS", "EVIDENCE_FOR"],
        "reverse": ["TESTS", "AFFECTS"],
    },
    "impact": {
        "forward": ["DECOMPOSED_TO", "SUPPORTED_BY"],
        "reverse": ["TESTS", "AFFECTS"],
    },
}

VECTOR_FALLBACK_MIN_RESULTS = 3
VECTOR_FALLBACK_CONFIDENCE_THRESHOLD = 0.65