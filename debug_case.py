"""
debug_case.py
-------------
Runs ONE model against ONE corpus case and prints the full raw output,
plus the parse result. Use this to see exactly what's breaking JSON
parsing or the functional (compile) gate, since benchmark.py only
prints pass/fail, not the raw text.

Usage:
    python debug_case.py --model ./models/codeqwen-7b-mlx-4bit --case corpus/cwe89/case_001_login.json
"""

import argparse
import json

from mlx_inference import RefactoringAgentModel
from prompt_templates import CWE_89_SYSTEM_PROMPT, build_cwe_89_prompt
from patch_parser import parse_patch_response

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--case", required=True)
args = parser.parse_args()

case_data = json.loads(open(args.case).read())

user_prompt = build_cwe_89_prompt(
    file_path=case_data["file_path"],
    function_name=case_data["function_name"],
    db_driver=case_data["db_driver"],
    taint_source=case_data["taint_source"],
    taint_sink=case_data["taint_sink"],
    vulnerable_code=case_data["vulnerable_code"],
    safe_idiom_example=case_data.get("safe_idiom_example", ""),
    existing_tests_summary=case_data.get("existing_tests_summary", ""),
)

agent = RefactoringAgentModel(args.model)
gen = agent.generate(CWE_89_SYSTEM_PROMPT, user_prompt, temperature=0.2)

print("=" * 80)
print("RAW MODEL OUTPUT (exactly as generated, no processing):")
print("=" * 80)
print(gen.raw_text)
print("=" * 80)
print("PARSE RESULT:")
print("=" * 80)
result = parse_patch_response(gen.raw_text)
print(f"status: {result.status}")
print(f"is_usable: {result.is_usable()}")
if result.patched_code:
    print("\n--- patched_code (attempting compile) ---")
    print(result.patched_code)
    try:
        compile(result.patched_code, "<patched>", "exec")
        print("\n[compile: OK]")
    except SyntaxError as e:
        print(f"\n[compile: FAILED] {e}")
