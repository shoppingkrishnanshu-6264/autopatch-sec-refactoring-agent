"""
benchmark.py
------------
Compares candidate local models (e.g. CodeQwen1.5-7B-Chat, Llama-3-8B-Instruct,
DeepSeek-Coder-6.7B-Instruct) across quantization levels on the metrics that
actually matter for the Refactoring Agent:

  1. parse_success_rate   - did the output parse into a valid PatchResult?
  2. functional_pass_rate - does the patch keep existing behavior? (sandbox hook)
  3. exploit_pass_rate    - does the patch actually close the vuln? (sandbox hook)
  4. ttft / throughput    - latency, from mlx_inference.py's GenerationMetrics

This file does NOT implement the Docker/microVM sandbox itself (that's the
Validation Agent's job) — it defines a small pluggable interface
(FunctionalTestRunner / ExploitTestRunner) so this harness can be pointed at
your real sandbox once it exists, and run against stubs before that.

Usage:
    python benchmark.py --config models.json --corpus corpus/cwe89

models.json example:
[
  {"name": "codeqwen-7b-4bit", "path": "./models/codeqwen-7b-mlx-4bit"},
  {"name": "codeqwen-7b-8bit", "path": "./models/codeqwen-7b-mlx-8bit"},
  {"name": "llama3-8b-4bit",   "path": "./models/llama3-8b-mlx-4bit"},
  {"name": "deepseek-coder-6.7b-4bit", "path": "./models/deepseek-coder-6.7b-mlx-4bit"}
]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from mlx_inference import RefactoringAgentModel, GenerationMetrics
from prompt_templates import CWE_89_SYSTEM_PROMPT, build_cwe_89_prompt
from patch_parser import parse_patch_response, PatchResult, PatchStatus


# --------------------------------------------------------------------------
# Test corpus schema
# --------------------------------------------------------------------------

@dataclass
class CWE89TestCase:
    case_id: str
    file_path: str
    function_name: str
    db_driver: str
    taint_source: str
    taint_sink: str
    vulnerable_code: str
    safe_idiom_example: str = "(none found in codebase)"
    existing_tests_summary: str = "(none provided)"

    @staticmethod
    def load_corpus(corpus_dir: str) -> list["CWE89TestCase"]:
        """Each case is a .json file in corpus_dir matching this schema."""
        cases = []
        for f in sorted(Path(corpus_dir).glob("*.json")):
            data = json.loads(f.read_text())
            cases.append(CWE89TestCase(**data))
        return cases


# --------------------------------------------------------------------------
# Pluggable validation hooks — wire these to the real Validation Agent sandbox
# --------------------------------------------------------------------------

FunctionalTestRunner = Callable[[CWE89TestCase, str], bool]
ExploitTestRunner = Callable[[CWE89TestCase, str], bool]


def stub_functional_runner(case: CWE89TestCase, patched_code: str) -> bool:
    """
    Placeholder: replace with a call into the Docker/microVM sandbox that
    runs the project's existing test suite against patched_code.
    Stub heuristic: reject only if patched_code is empty or doesn't parse
    as valid Python (a real functional gate obviously needs the actual tests).
    """
    if not patched_code.strip():
        return False
    try:
        compile(patched_code, "<patched>", "exec")
        return True
    except SyntaxError:
        return False


def stub_exploit_runner(case: CWE89TestCase, patched_code: str) -> bool:
    """
    Placeholder: replace with a call into the sandbox that replays a CWE-89
    exploit payload (e.g. `' OR '1'='1`) against the patched function and
    confirms it no longer succeeds.
    Stub heuristic: checks for a real placeholder AND the absence of ACTUAL
    interpolation — an f-string with no {} inside it (e.g. a vestigial
    `f"...?"` the model left over from an earlier attempt) is not a real
    injection vector and should not be flagged just for having an f-prefix.
    NOT a substitute for dynamic exploit replay — use only until the real
    exploit gate is wired in.
    """
    has_placeholder = any(marker in patched_code for marker in ("?", "%s", ":"))
    fstring_has_interp = bool(re.search(r'f["\'][^"\']*\{[^}]+\}', patched_code))
    percent_interp = bool(re.search(r'%\s*\(', patched_code)) or (" % (" in patched_code)
    format_interp = ".format(" in patched_code
    concat_interp = (
        ("+ " + case.taint_source) in patched_code
        or (case.taint_source + " +") in patched_code
    )
    looks_interpolated = fstring_has_interp or percent_interp or format_interp or concat_interp
    return has_placeholder and not looks_interpolated


# --------------------------------------------------------------------------
# Benchmark runner
# --------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    model_name: str
    parsed: bool
    functional_pass: Optional[bool]
    exploit_pass: Optional[bool]
    metrics: dict
    status: str


@dataclass
class ModelSummary:
    model_name: str
    n_cases: int
    parse_success_rate: float
    functional_pass_rate: float
    exploit_pass_rate: float
    end_to_end_pass_rate: float   # parsed AND functional AND exploit
    mean_ttft_s: float
    median_ttft_s: float
    mean_tokens_per_second: float
    mean_completion_tokens: float


def run_case(
    agent: RefactoringAgentModel,
    case: CWE89TestCase,
    functional_runner: FunctionalTestRunner,
    exploit_runner: ExploitTestRunner,
) -> CaseResult:
    user_prompt = build_cwe_89_prompt(
        file_path=case.file_path,
        function_name=case.function_name,
        db_driver=case.db_driver,
        taint_source=case.taint_source,
        taint_sink=case.taint_sink,
        vulnerable_code=case.vulnerable_code,
        safe_idiom_example=case.safe_idiom_example,
        existing_tests_summary=case.existing_tests_summary,
    )

    gen = agent.generate(CWE_89_SYSTEM_PROMPT, user_prompt, temperature=0.2)
    result: PatchResult = parse_patch_response(gen.raw_text)

    functional_pass = None
    exploit_pass = None
    if result.is_usable():
        functional_pass = functional_runner(case, result.patched_code)
        exploit_pass = exploit_runner(case, result.patched_code)

    return CaseResult(
        case_id=case.case_id,
        model_name=agent.model_path,
        parsed=result.is_usable(),
        functional_pass=functional_pass,
        exploit_pass=exploit_pass,
        metrics=gen.metrics.to_dict(),
        status=result.status.value,
    )


def summarize(model_name: str, results: list[CaseResult]) -> ModelSummary:
    n = len(results)
    parsed = [r for r in results if r.parsed]
    functional_ok = [r for r in parsed if r.functional_pass]
    exploit_ok = [r for r in parsed if r.exploit_pass]
    end_to_end = [r for r in parsed if r.functional_pass and r.exploit_pass]

    ttfts = [r.metrics["ttft_seconds"] for r in results]
    tps = [r.metrics["tokens_per_second"] for r in results]
    completion_tokens = [r.metrics["completion_tokens"] for r in results]

    return ModelSummary(
        model_name=model_name,
        n_cases=n,
        parse_success_rate=len(parsed) / n if n else 0.0,
        functional_pass_rate=len(functional_ok) / n if n else 0.0,
        exploit_pass_rate=len(exploit_ok) / n if n else 0.0,
        end_to_end_pass_rate=len(end_to_end) / n if n else 0.0,
        mean_ttft_s=statistics.mean(ttfts) if ttfts else 0.0,
        median_ttft_s=statistics.median(ttfts) if ttfts else 0.0,
        mean_tokens_per_second=statistics.mean(tps) if tps else 0.0,
        mean_completion_tokens=statistics.mean(completion_tokens) if completion_tokens else 0.0,
    )


def run_benchmark(
    model_configs: list[dict],
    corpus: list[CWE89TestCase],
    functional_runner: FunctionalTestRunner = stub_functional_runner,
    exploit_runner: ExploitTestRunner = stub_exploit_runner,
) -> tuple[list[CaseResult], list[ModelSummary]]:
    all_results: list[CaseResult] = []
    summaries: list[ModelSummary] = []

    for cfg in model_configs:
        name, path = cfg["name"], cfg["path"]
        print(f"\n=== Loading {name} ({path}) ===")
        agent = RefactoringAgentModel(path)
        print(f"  load time: {agent.load_seconds:.2f}s")

        model_results = []
        for case in corpus:
            r = run_case(agent, case, functional_runner, exploit_runner)
            model_results.append(r)
            print(
                f"  [{case.case_id}] parsed={r.parsed} "
                f"functional={r.functional_pass} exploit={r.exploit_pass} "
                f"ttft={r.metrics['ttft_seconds']}s "
                f"tps={r.metrics['tokens_per_second']}"
            )

        all_results.extend(model_results)
        summaries.append(summarize(name, model_results))

    return all_results, summaries


def print_leaderboard(summaries: list[ModelSummary]):
    ranked = sorted(summaries, key=lambda s: s.end_to_end_pass_rate, reverse=True)
    print("\n" + "=" * 100)
    print(f"{'model':30} {'e2e_pass':>9} {'parse':>7} {'func':>7} {'exploit':>8} "
          f"{'ttft(s)':>9} {'tok/s':>8}")
    print("-" * 100)
    for s in ranked:
        print(
            f"{s.model_name:30} {s.end_to_end_pass_rate:>9.1%} "
            f"{s.parse_success_rate:>7.1%} {s.functional_pass_rate:>7.1%} "
            f"{s.exploit_pass_rate:>8.1%} {s.mean_ttft_s:>9.3f} "
            f"{s.mean_tokens_per_second:>8.1f}"
        )
    print("=" * 100)
    print(f"Winner (end-to-end validated patch rate): {ranked[0].model_name}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark candidate models for the Refactoring Agent")
    parser.add_argument("--config", required=True, help="JSON list of {name, path} model configs")
    parser.add_argument("--corpus", required=True, help="Directory of CWE-89 test case JSON files")
    parser.add_argument("--out", default="benchmark_results.json")
    args = parser.parse_args()

    model_configs = json.loads(Path(args.config).read_text())
    corpus = CWE89TestCase.load_corpus(args.corpus)
    if not corpus:
        raise SystemExit(f"No test cases found in {args.corpus}")

    all_results, summaries = run_benchmark(model_configs, corpus)
    print_leaderboard(summaries)

    Path(args.out).write_text(json.dumps({
        "results": [asdict(r) for r in all_results],
        "summaries": [asdict(s) for s in summaries],
    }, indent=2))
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
