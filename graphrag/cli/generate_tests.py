import argparse
import json
from pathlib import Path

from graphrag.generation.test_generator import generate_tests_cli


def main():
    parser = argparse.ArgumentParser(description="Generate grounded test cases from Context Pack")
    parser.add_argument("--context-pack", required=True)
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--out", default="generated_tests.json")
    args = parser.parse_args()

    generated = generate_tests_cli(args.context_pack, args.provider)

    Path(args.out).write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Generated tests saved: {args.out}")
    print(f"Test cases: {len(generated.get('test_cases', []))}")
    print(f"Valid: {generated['is_valid']}")
    print(f"Validation errors: {len(generated['validation_errors'])}")


if __name__ == "__main__":
    main()