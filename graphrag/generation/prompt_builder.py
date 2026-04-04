# graphrag/generation/prompt_builder.py
import json
from typing import Dict, Any, List


def _format_chunks(chunks: List[Any], label: str) -> str:
    """Format chunks for prompt. Handles dicts, dataclasses, or lists."""
    lines = [f"{label}:"]
    
    if not chunks:
        lines.append("- None")
        return "\n".join(lines)
    
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            chunk_dict = chunk
        elif hasattr(chunk, '__dict__'):
            chunk_dict = chunk.__dict__
        elif isinstance(chunk, str):
            lines.append(f"- chunk_id: unknown\n  text: {chunk[:200]}...")
            continue
        else:
            lines.append(f"- chunk_id: unknown_{i}")
            continue
            
        chunk_id = chunk_dict.get('chunk_id', f'unknown_{i}')
        chunk_type = chunk_dict.get('chunk_type', 'unknown')
        section_path = chunk_dict.get('section_path', 'unknown')
        text = chunk_dict.get('text', '').strip()
        
        lines.append(
            f"- chunk_id: {chunk_id}\n"
            f"  chunk_type: {chunk_type}\n"
            f"  section_path: {section_path}\n"
            f"  text: {text[:200]}{'...' if len(text) > 200 else ''}"
        )
    return "\n".join(lines)


# graphrag/generation/prompt_builder.py
def build_test_generation_prompt(context_pack: Dict[str, Any]) -> str:
    """
    Ultra-strict prompt that forces schema compliance.
    """
    # Extract real chunk IDs
    valid_chunk_ids = []
    evidence_chunks = context_pack.get("evidence_chunks", [])
    
    for c in evidence_chunks:
        if isinstance(c, dict):
            cid = c.get("chunk_id")
        elif hasattr(c, "__dict__"):
            cid = c.__dict__.get("chunk_id")
        else:
            continue
            
        if cid:
            valid_chunk_ids.append(cid)
    
    valid_chunk_ids = list(set(valid_chunk_ids))
    
    # Evidence text (truncated)
    evidence_text = []
    for i, chunk in enumerate(evidence_chunks[:3]):  # Top 3 only
        if isinstance(chunk, dict):
            text = chunk.get("text", "")[:150]
            cid = chunk.get("chunk_id", "")
        elif hasattr(chunk, "__dict__"):
            text = chunk.__dict__.get("text", "")[:150]
            cid = chunk.__dict__.get("chunk_id", "")
        else:
            continue
            
        evidence_text.append(f"[{cid}] {text}")
    
    evidence_summary = "\n".join(evidence_text)

    return f"""Generate 1 test case in EXACT JSON format.

VALID CHUNK IDs ONLY: {valid_chunk_ids}

Evidence:
{evidence_summary}

{{ 
  "test_cases": [{{
    "title": "1 sentence title",
    "steps": [{{
      "step_number": 1,
      "action": "1 sentence action",
      "expected_result": "1 sentence expected result", 
      "evidence_chunk_ids": ["{valid_chunk_ids[0]}"]
    }}]
  }}]
}}

JSON ONLY. No other text."""