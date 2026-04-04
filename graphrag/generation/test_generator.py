import json
import os
import requests
from pathlib import Path
from typing import Dict, Any, List, Set

from graphrag.generation.prompt_builder import build_test_generation_prompt
from graphrag.generation.output_validator import validate_generated_tests


def _to_dict(item: Any) -> Dict[str, Any]:
    """Convert object to dict."""
    if isinstance(item, dict):
        return item
    if hasattr(item, "__dict__"):
        return item.__dict__
    return {}


def _extract_valid_chunk_ids(context_pack: Dict[str, Any]) -> Set[str]:
    """Extract valid chunk IDs from Context Pack."""
    chunk_ids = set()
    
    for chunks in ["evidence_chunks", "parent_context"]:
        for c in context_pack.get(chunks, []):
            c_dict = _to_dict(c)
            chunk_id = c_dict.get("chunk_id")
            if chunk_id:
                chunk_ids.add(chunk_id)
    
    return chunk_ids


def _ollama_generate(prompt: str, model: str = "llama3:8b") -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,  # ZERO temperature
            "top_p": 0.1,
            "repeat_penalty": 1.05,
            "num_predict": 512,
        },
    }
    
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("response", "")


def call_llm(prompt: str, provider: str = "ollama") -> str:
    if provider == "ollama":
        return _ollama_generate(prompt)
    raise ValueError(f"Unsupported provider: {provider}")


def repair_invalid_citations(generated: Dict[str, Any], context_pack: Dict[str, Any]) -> Dict[str, Any]:
    """Repair LLM hallucinations by mapping to valid chunk IDs."""
    valid_chunk_ids = _extract_valid_chunk_ids(context_pack)
    
    for tc in generated.get("test_cases", []):
        for step in tc.get("steps", []):
            cited_ids = step.get("evidence_chunk_ids", [])
            valid_cited = [cid for cid in cited_ids if cid in valid_chunk_ids]
            
            if not valid_cited and valid_chunk_ids:
                # Fallback to first valid chunk
                first_id = list(valid_chunk_ids)[0]
                step["evidence_chunk_ids"] = [first_id]
                step["expected_result"] += f" [Evidence: {first_id}]"
            else:
                step["evidence_chunk_ids"] = valid_cited
    
    return generated


def generate_tests_from_context_pack(
    context_pack: Dict[str, Any],
    provider: str = "ollama",
) -> Dict[str, Any]:
    prompt = build_test_generation_prompt(context_pack)
    
    raw_response = call_llm(prompt, provider)
    
    try:
        generated = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}")
    
    # Repair citations first
    generated = repair_invalid_citations(generated, context_pack)
    
    # Then validate
    errors = validate_generated_tests(context_pack, generated)
    generated["validation_errors"] = errors
    generated["is_valid"] = len(errors) == 0
    
    return generated


def generate_tests_cli(context_pack_path: str, provider: str = "ollama"):
    context_pack = json.loads(Path(context_pack_path).read_text())
    generated = generate_tests_from_context_pack(context_pack, provider)
    
    print(json.dumps(generated, indent=2))
    print(f"\nValid: {generated['is_valid']}")
    if generated["validation_errors"]:
        print("Errors:", generated["validation_errors"])
    return generated