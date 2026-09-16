"""
iterative_refine.py
--------------------
Wires together the pieces that already exist but were never actually
exercised end-to-end: generate a patch, validate it (functional + exploit
gates), and on failure feed the specific failure back to the model via
build_refinement_prompt() for a corrected attempt — capped at a retry
budget to avoid infinite cost spirals (per the original architecture's
self-correction loop design).

This is the Refactoring Agent's iterative refinement responsibility,
demonstrated as a real loop rather than an unused function.

Usage:
    python iterative_refine.py --model ./models/codeqwen-7b-mlx-4bit \
        --case corpus/cwe89/case_001_login.json --max-retries 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mlx_inference import RefactoringAgentModel
from prompt_templates import (
    CWE_89_SYSTEM_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    build_cwe_89_prompt,
    build_refinement_prompt,
)
from patch_parser import parse_patch_response, PatchResult

# Reuse the same stub validators benchmark.py uses, so this loop and the
# benchmark agree on what "passing" means. Swap these for the real
# Validation Agent sandbox once it exists — nothing else here needs to change.
from benchmark import stub_functional_runner, stub_exploit_runner, CWE89TestCase


@dataclass
class RefinementAttempt:
    attempt_number: int  # 0 = first try, 1+ = refinement turns
    parsed: bool
    functional_pass: Optional[bool]
    exploit_pass: Optional[bool]
    failure_type: Optional[str]
    failure_details: Optional[str]
    patched_code: Optional[str]
    raw_output: str


@dataclass
class RefinementResult:
    case_id: str
    succeeded: bool
    attempts: list[RefinementAttempt] = field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)


def _describe_failure(result: PatchResult, functional_pass: Optional[bool], exploit_pass: Optional[bool]) -> tuple[str, str]:
    """Turn a validation outcome into (failure_type, failure_details) for the refinement prompt."""
    if not result.is_usable():
        return "parse_error", (
            f"The response could not be parsed into the required JSON schema "
            f"(status: {result.status.value}). Produce valid JSON exactly matching "
            f"the schema, with patched_code as a properly \\n-escaped string — "
            f"not a triple-quoted or markdown-fenced block."
        )
    if functional_pass is False:
        return "functional_failure", (
            "The patched_code did not compile as valid Python. Check for missing "
            "spaces, broken indentation, or stray markdown fences accidentally "
            "left inside the code string."
        )
    if exploit_pass is False:
        return "exploit_still_succeeds", (
            "The patch did not close the vulnerability — the exploit check still "
            "detected an unparameterized/interpolated query pattern. Ensure the "
            "query string uses a placeholder (?, %s, or :name) with NO string "
            "interpolation (f-string, %, .format(), or +) of the tainted input, "
            "and that the value is passed via the parameters argument."
        )
    return "unknown_failure", "Validation failed for an unspecified reason."


def refine_until_valid(
    agent: RefactoringAgentModel,
    case: CWE89TestCase,
    max_retries: int = 3,
) -> RefinementResult:
    """
    Run the generate -> validate -> (on failure) refine loop for one case.
    Stops as soon as a patch passes both gates, or after max_retries
    refinement turns are exhausted.
    """
    original_context = build_cwe_89_prompt(
        file_path=case.file_path,
        function_name=case.function_name,
        db_driver=case.db_driver,
        taint_source=case.taint_source,
        taint_sink=case.taint_sink,
        vulnerable_code=case.vulnerable_code,
        safe_idiom_example=case.safe_idiom_example,
        existing_tests_summary=case.existing_tests_summary,
    )

    attempts: list[RefinementAttempt] = []
    previous_patch = ""
    failure_type = ""
    failure_details = ""

    for attempt_num in range(max_retries + 1):  # attempt 0 = first try
        if attempt_num == 0:
            system_prompt = CWE_89_SYSTEM_PROMPT
            user_prompt = original_context
        else:
            system_prompt = REFINEMENT_SYSTEM_PROMPT
            user_prompt = build_refinement_prompt(
                previous_patch=previous_patch,
                failure_type=failure_type,
                failure_details=failure_details,
                original_context=original_context,
            )

        gen = agent.generate(system_prompt, user_prompt, temperature=0.2)
        result = parse_patch_response(gen.raw_text)

        functional_pass: Optional[bool] = None
        exploit_pass: Optional[bool] = None
        if result.is_usable():
            functional_pass = stub_functional_runner(case, result.patched_code)
            exploit_pass = stub_exploit_runner(case, result.patched_code)

        succeeded = bool(result.is_usable() and functional_pass and exploit_pass)

        if not succeeded:
            failure_type, failure_details = _describe_failure(result, functional_pass, exploit_pass)
            previous_patch = result.patched_code or gen.raw_text[:500]

        attempts.append(RefinementAttempt(
            attempt_number=attempt_num,
            parsed=result.is_usable(),
            functional_pass=functional_pass,
            exploit_pass=exploit_pass,
            failure_type=None if succeeded else failure_type,
            failure_details=None if succeeded else failure_details,
            patched_code=result.patched_code,
            raw_output=gen.raw_text,
        ))

        if succeeded:
            return RefinementResult(case_id=case.case_id, succeeded=True, attempts=attempts)

    return RefinementResult(case_id=case.case_id, succeeded=False, attempts=attempts)


def main():
    parser = argparse.ArgumentParser(description="Iterative refinement loop for the Refactoring Agent")
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", required=True, help="Path to a single corpus case JSON file")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    case_data = json.loads(Path(args.case).read_text())
    case = CWE89TestCase(**case_data)

    agent = RefactoringAgentModel(args.model)
    print(f"[load] model ready in {agent.load_seconds:.2f}s\n")

    result = refine_until_valid(agent, case, max_retries=args.max_retries)

    print(f"case: {result.case_id}")
    print(f"succeeded: {result.succeeded}")
    print(f"total attempts: {result.total_attempts}\n")

    for a in result.attempts:
        label = "PASS" if (a.functional_pass and a.exploit_pass and a.parsed) else "FAIL"
        print(f"--- attempt {a.attempt_number} [{label}] ---")
        print(f"  parsed={a.parsed} functional={a.functional_pass} exploit={a.exploit_pass}")
        if a.failure_type:
            print(f"  failure_type: {a.failure_type}")
            print(f"  failure_details: {a.failure_details}")
        if a.patched_code:
            print(f"  patched_code:\n{a.patched_code}")
        print()


if __name__ == "__main__":
    main()
