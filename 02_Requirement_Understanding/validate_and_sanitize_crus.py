"""
Post-generation CRU Validator & Sanitizer
Phase B.5 – Deterministic enforcement layer

Purpose:
- Validate CRU invariants
- Repair common LLM extraction errors
- Split / merge constraints correctly
- Remove meta / explanatory noise
- Produce Phase-B clean CRUs suitable for Stage 5

This module MUST be deterministic.
"""

import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import spacy

# ----------------------------
# Configuration
# ----------------------------

ALLOWED_TYPES = {
    "functional",
    "performance",
    "security",
    "usability",
    "reliability",
    "portability",
    "other"
}

MODAL_VERBS = {
    "shall", "must", "should", "will",
    "may", "would", "could", "can"
}

REFERENCE_PRONOUNS = {
    "this", "that", "it", "these", "those"
}

EXPLANATORY_VERBS = {
    "establish", "ensure", "provide", "result",
    "lead", "support", "enable", "represent"
}

METRIC_PATTERN = re.compile(
    r"""
    (?:
        \d+(\.\d+)?\s*(ms|s|seconds|minutes|%|users|requests)
        |
        99th\s+percentile
        |
        uptime
        |
        within\s+\d+
        |
        not\s+exceed
        |
        less\s+than
        |
        below\s+\d+
    )
    """,
    re.IGNORECASE | re.VERBOSE
)

nlp = spacy.load("en_core_web_sm")


# ----------------------------
# Utility helpers
# ----------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def deterministic_id(parent_id: str, text: str, suffix: str = None) -> str:
    h = hashlib.sha1(f"{parent_id}|{text}".encode("utf-8")).hexdigest()[:8]
    return f"CRU_{parent_id}_{h}" + (f"_{suffix}" if suffix else "")


# ----------------------------
# Actor extraction
# ----------------------------

def extract_actor(text: str) -> str:
    doc = nlp(text)

    for token in doc:
        if (
            token.dep_ in ("nsubj", "nsubjpass")
            and token.pos_ in ("NOUN", "PROPN")
            and token.text.lower() not in REFERENCE_PRONOUNS
        ):
            return token.lemma_

    return "System"


# ----------------------------
# Action extraction
# ----------------------------

def extract_action(text: str) -> str | None:
    doc = nlp(text)
    root = None

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root = token
            break

    if not root:
        return None

    parts = [root.lemma_]

    for child in root.children:
        if child.dep_ in ("dobj", "prep", "xcomp", "attr", "acomp"):
            phrase = " ".join(
                t.text
                for t in child.subtree
                if t.text.lower() not in MODAL_VERBS
            )
            parts.append(phrase)

    action = " ".join(parts).strip()

    tokens = action.split()
    if len(tokens) < 2:
        return None

    if any(tok.lower() in MODAL_VERBS for tok in tokens):
        return None

    return action


# ----------------------------
# Explanatory sentence detection
# ----------------------------

def is_explanatory_sentence(text: str) -> bool:
    words = text.strip().split()
    if not words:
        return True

    first = words[0].lower()
    if first in REFERENCE_PRONOUNS:
        return True

    for verb in EXPLANATORY_VERBS:
        if re.search(rf"\b{verb}\b", text, re.IGNORECASE):
            return True

    return False


# ----------------------------
# Constraint splitting
# ----------------------------

def extract_constraints(text: str) -> List[str]:
    matches = METRIC_PATTERN.findall(text)
    if not matches:
        return []

    results = []
    for match in METRIC_PATTERN.finditer(text):
        results.append(match.group(0))

    return list(dict.fromkeys(results))


# ----------------------------
# Type inference
# ----------------------------

def infer_type(text: str) -> str:
    t = text.lower()

    if re.search(r"(ms|seconds|latency|percentile|response)", t):
        return "performance"
    if re.search(r"(password|hash|https|auth|role)", t):
        return "security"
    if re.search(r"(ui|ux|accessibility|responsive|navigation)", t):
        return "usability"
    if re.search(r"(uptime|backup|acid|recovery|failure)", t):
        return "reliability"
    if re.search(r"(browser|os|platform|compatibility)", t):
        return "portability"

    return "functional"


# ----------------------------
# Core validation pipeline
# ----------------------------

def sanitize_crus(candidate_crus: List[Dict]) -> Tuple[List[Dict], Dict]:
    sanitized = []
    report = {
        "input_crus": len(candidate_crus),
        "dropped_explanatory": 0,
        "repaired": 0,
        "split_constraints": 0,
        "merged_duplicates": 0,
        "output_crus": 0
    }

    seen = {}

    for cru in candidate_crus:
        text = cru.get("original_text", "").strip()
        parent_id = cru.get("parent_requirement_id")

        if not text or not parent_id:
            continue

        if is_explanatory_sentence(text):
            report["dropped_explanatory"] += 1
            continue

        actor = extract_actor(text)
        action = extract_action(text)

        if not action:
            report["repaired"] += 1
            continue

        constraints = extract_constraints(text)
        cru_type = infer_type(text)

        if cru_type not in ALLOWED_TYPES:
            cru_type = "other"

        base_key = normalize_text(f"{parent_id}|{actor}|{action}")

        if not constraints:
            cru_id = deterministic_id(parent_id, text)
            new_cru = {
                "cru_id": cru_id,
                "parent_requirement_id": parent_id,
                "type": cru_type,
                "actor": actor,
                "action": action,
                "observable_outcome": cru.get("observable_outcome"),
                "constraint_or_metric": None,
                "source": "SRS",
                "traceability": cru.get("traceability"),
                "original_text": text
            }

            if base_key in seen:
                report["merged_duplicates"] += 1
                continue

            seen[base_key] = True
            sanitized.append(new_cru)
            continue

        for idx, constraint in enumerate(constraints, start=1):
            cru_id = deterministic_id(parent_id, text, f"C{idx}")
            new_cru = {
                "cru_id": cru_id,
                "parent_requirement_id": parent_id,
                "type": cru_type,
                "actor": actor,
                "action": action,
                "observable_outcome": None,
                "constraint_or_metric": constraint,
                "source": "SRS",
                "traceability": cru.get("traceability"),
                "original_text": text
            }

            key = normalize_text(f"{base_key}|{constraint}")
            if key in seen:
                report["merged_duplicates"] += 1
                continue

            seen[key] = True
            sanitized.append(new_cru)
            report["split_constraints"] += 1

    report["output_crus"] = len(sanitized)
    return sanitized, report


# ----------------------------
# Runner
# ----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    parser.add_argument("--report", dest="report_path", required=True)
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        candidate = json.load(f)

    candidate_crus = candidate.get("crus", candidate)

    clean_crus, validation_report = sanitize_crus(candidate_crus)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "total_crus": len(clean_crus)
                },
                "crus": clean_crus
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    print("✅ CRU validation & sanitization complete")
    print(json.dumps(validation_report, indent=2))
