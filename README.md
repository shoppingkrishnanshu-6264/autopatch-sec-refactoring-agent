# AutoPatch-Sec — Refactoring Agent

Local, MLX-based patch synthesis module for **AutoPatch-Sec**, an autonomous multi-agent DevSecOps pipeline that ingests SAST vulnerability reports, generates secure patches using local models, and validates them in an isolated sandbox against functional and exploit-based tests.

This repo covers the **Refactoring Agent** — the component responsible for synthesizing secure code replacements for vulnerabilities flagged by the Threat Parsing Agent. Current scope: **CWE-89 (SQL Injection)**.

## What this does

Given a vulnerability context (vulnerable function, taint source/sink, DB driver), a locally-served quantized model generates a parameterized-query patch as structured JSON, which is then validated (compile check + exploit heuristic) and, on failure, iteratively refined with the specific failure fed back to the model — capped at a retry budget.

Everything runs entirely on-device via [MLX](https://github.com/ml-explore/mlx) on Apple Silicon — no code or vulnerability context leaves the machine.

## Repo structure

| File | Purpose |
|---|---|
| `mlx_inference.py` | Model loading, streaming generation, TTFT/throughput instrumentation, tokenizer marker cleanup |
| `prompt_templates.py` | CWE-89 system/user prompts + refinement prompt for retry turns |
| `patch_parser.py` | Strict JSON parsing with a salvage fallback for malformed model output |
| `benchmark.py` | Multi-model comparison across a test corpus, with functional/exploit stub gates |
| `iterative_refine.py` | Generate → validate → refine-on-failure loop |
| `debug_case.py` | Single-case raw output inspection, for diagnosing generation/parsing issues |
| `corpus/cwe89/` | CWE-89 test cases (vulnerability context + expected safe idiom) |
| `models.json.example` | Template config for pointing the benchmark at local MLX model paths |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install mlx-lm
```

Convert a model to quantized MLX format (repeat per model):

```bash
mlx_lm.convert --hf-path Qwen/CodeQwen1.5-7B-Chat --mlx-path ./models/codeqwen-7b-mlx-4bit -q --q-bits 4
```

> Model weights are **not** checked into this repo (see `.gitignore`) — convert them locally into `./models/`.

Copy the config template and point it at your converted models:

```bash
cp models.json.example models.json
# edit models.json to match your local model paths
```

## Usage

Run a single case through one model, for debugging raw output:

```bash
python debug_case.py --model ./models/codeqwen-7b-mlx-4bit --case corpus/cwe89/case_001_login.json
```

Run the full benchmark across all configured models:

```bash
python benchmark.py --config models.json --corpus corpus/cwe89
```

Run the iterative refinement loop on a single case:

```bash
python iterative_refine.py --model ./models/codeqwen-7b-mlx-4bit --case corpus/cwe89/case_001_login.json --max-retries 3
```

## Models evaluated

- CodeQwen1.5-7B-Chat (4-bit)
- Meta-Llama-3-8B-Instruct (4-bit)
- DeepSeek-Coder-6.7B-Instruct (4-bit)

## Key findings

- **DeepSeek-Coder-6.7B** reliably produced well-formed JSON with normal prose, but consistently generated Python code with all whitespace collapsed (`defget_user_by_username(...)`) — a generation-level defect, not a decoding artifact, that iterative refinement could not correct across repeated attempts.
- **CodeQwen1.5-7B** showed meaningful output variance run-to-run at temperature 0.2 — successfully self-correcting a real vulnerability via refinement in one trial, failing to produce parseable JSON across an entire retry budget in another.
- **Static exploit heuristics produce false negatives on safe code** — an early version of the exploit-detection stub flagged any f-string prefix as unsafe regardless of whether it actually interpolated anything, incorrectly failing genuinely safe patches. This is direct empirical support for dynamic, sandbox-based exploit validation over static pattern matching in the full AutoPatch-Sec architecture.

Full write-up: see `refactoring_agent_report.md`.

## Status / limitations

- Functional and exploit validation gates here are **local stand-ins** (Python `compile()` and a static pattern check) for the real Validation Agent sandbox.
- Current test corpus is small (2 CWE-89 cases) — sufficient to validate the pipeline works end-to-end, not yet sufficient for statistically meaningful model comparison.
- Latency is currently *instrumented* (TTFT, tokens/sec) but not yet *optimized*.

## Part of

AutoPatch-Sec — Autonomous DevSecOps Multi-Agent Framework for Sandboxed Vulnerability Remediation (MTech project).
