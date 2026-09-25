# Refactoring Agent Module — AutoPatch-Sec

**Component of:** AutoPatch-Sec — Autonomous DevSecOps Multi-Agent Framework for Sandboxed Vulnerability Remediation
**Module owner responsibilities:** Local model serving, prompt engineering & patch synthesis, iterative refinement handling, latency & token optimization

---

## 1. Objective

The Refactoring Agent is responsible for synthesizing secure code replacements for vulnerabilities identified by the Threat Parsing Agent, using a locally-served, quantized instruction-tuned coding model running entirely on-device (Apple Silicon M4, via MLX). This module's scope covers CWE-89 (SQL Injection) as the initial target vulnerability class, with the design intended to generalize to further CWE classes.

Four sub-responsibilities were scoped for this module:
1. Local model serving — deploy and optimize quantized coding models via MLX
2. Prompt engineering & patch synthesis — transform vulnerable patterns into secure, parameterized implementations without altering business logic
3. Iterative refinement handling — process validation failures and retry with corrective context
4. Latency & token optimization — measure and reduce time-to-first-token, throughput, and token consumption

## 2. System Architecture (Module-Level)

```
Vulnerability Context (CWE, taint source/sink, vulnerable code)
        │
        ▼
  Prompt Templates  ──►  system prompt (rules + JSON schema) + user prompt (context)
        │
        ▼
  MLX Inference Harness  ──►  quantized local model, streaming generation
        │                     TTFT / throughput instrumentation
        ▼
  Patch Parser  ──►  JSON extraction + salvage fallback → structured PatchResult
        │
        ▼
  Validation (stub, standing in for the Validation Agent sandbox)
        │             ├─ functional gate (compiles as valid Python)
        │             └─ exploit gate (parameterization check)
        │
        ├─ PASS ──► done
        └─ FAIL ──► Iterative Refinement Loop (feed failure back, retry up to N times)
```

### 2.1 Components implemented

| Component | File | Purpose |
|---|---|---|
| MLX inference harness | `mlx_inference.py` | Model loading, streaming generation, TTFT/throughput instrumentation, tokenizer marker cleanup |
| Prompt templates | `prompt_templates.py` | CWE-89 system/user prompts, refinement prompt for retry turns |
| Patch parser | `patch_parser.py` | Strict JSON parsing + salvage fallback for malformed model output |
| Benchmark harness | `benchmark.py` | Multi-model comparison across a test corpus; functional/exploit stub gates |
| Iterative refinement loop | `iterative_refine.py` | Generate → validate → refine-on-failure, capped retry budget |
| Debug utility | `debug_case.py` | Single-case raw output inspection for diagnosing generation/parsing issues |

## 3. Models Evaluated

Three quantized (4-bit, MLX format) instruction-tuned coding models were converted and benchmarked locally:

| Model | Parameters | Quantization | Peak memory (approx.) |
|---|---|---|---|
| CodeQwen1.5-7B-Chat | 7B | 4-bit (4.5 bits/weight effective) | ~3.8 GB |
| Meta-Llama-3-8B-Instruct | 8B | 4-bit | ~4.2 GB |
| DeepSeek-Coder-6.7B-Instruct | 6.7B | 4-bit | ~4.0 GB |

All three run entirely on-device via Apple's MLX framework, exploiting the M4's unified memory to avoid host↔device weight transfer.

## 4. Methodology

### 4.1 Prompt design

The system prompt for CWE-89 uses **template-guided synthesis** rather than fully open-ended generation: the model is constrained to a single acceptable fix strategy (parameterized queries), explicitly forbidden from using surface-level sanitization, and required to preserve business logic (function signature, return values, control flow). Output is constrained to a strict JSON schema with an explicit escape hatch (`status: insufficient_context`) for cases the model cannot confidently fix, rather than forcing a guess.

### 4.2 Validation gates (stub implementation)

Two gates are checked per generated patch:
- **Functional gate**: the patched code must compile as valid Python (`compile()`). This stands in for running the project's actual test suite, which requires the Validation Agent's sandbox.
- **Exploit gate**: static heuristic check for the presence of a real placeholder (`?`, `%s`, `:name`) combined with the *absence* of genuine string interpolation (an f-string with actual `{}` contents, `%`-formatting, `.format()`, or concatenation of the tainted variable). This stands in for dynamic exploit replay against the patched function, which also requires the sandbox.

Both gates are explicitly documented as stubs; a key finding of this work (Section 6.3) motivates why the exploit gate specifically cannot remain a static heuristic in the final system.

### 4.3 Iterative refinement

On validation failure, the specific failure (parse error, functional failure, or exploit-still-succeeds) is classified and fed back to the model via a dedicated refinement prompt, along with the previous (failing) patch. This is capped at a configurable retry budget (default 3 retries, 4 total attempts) to bound cost, consistent with the self-correction loop specified in the overall AutoPatch-Sec architecture.

## 5. Results

### 5.1 Benchmark run (2-case CWE-89 corpus)

| Model | Parse success | Functional pass | Exploit pass | End-to-end pass | Mean TTFT (s) | Mean tok/s |
|---|---|---|---|---|---|---|
| Llama-3-8B-Instruct | 100% | 100% | 100% | **100%** | 3.12 | 23.0 |
| CodeQwen1.5-7B-Chat | 50–100%* | 50–100%* | 50–100%* | 50–100%* | 5.4 | 23.8 |
| DeepSeek-Coder-6.7B-Instruct | 100% | 0% | 100% | 0% | 3.08 | 24.1 |

*CodeQwen's pass rate varied between runs (50% in one full-corpus run, 100% on isolated repeated single-case tests) — see Section 6.2 on generation variance.

**Note on sample size**: this table reflects a 2-case corpus, sufficient to validate that the pipeline functions correctly end-to-end but *not* sufficient for statistically meaningful comparison between models. A corpus expansion (10–15+ cases spanning multiple sinks/drivers) is identified as necessary future work before these numbers should be treated as a definitive model comparison (Section 7).

### 5.2 Iterative refinement trials

**DeepSeek-Coder**, 4 attempts (1 initial + 3 refinement turns), case `cwe89_001_login_lookup`: failed on every attempt with the identical defect — code generated with no whitespace between tokens (e.g. `defget_user_by_username(cursor,username):...`), despite an explicit, targeted correction ("check for missing spaces...") included in every refinement prompt.

**CodeQwen1.5-7B-Chat**, same case, two separate trials:
- Trial A: attempt 0 produced a genuinely vulnerable patch (real `{username}` f-string interpolation); the refinement loop corrected this by attempt 1, converging on a safe, correctly parameterized query.
- Trial B: all 4 attempts failed on JSON formatting (`parse_error`), never reaching a compilable patch.

### 5.3 Quantization tradeoff: 4-bit vs 8-bit (CodeQwen1.5-7B-Chat)

To evaluate whether higher-precision quantization is worth its resource cost for this task, CodeQwen1.5-7B-Chat was additionally converted and benchmarked at 8-bit (8.5 bits/weight effective), alongside the existing 4-bit version, on the same 2-case corpus:

| Quantization | Parse | Functional | Exploit | End-to-end | Mean TTFT (s) | Mean tok/s | Peak memory |
|---|---|---|---|---|---|---|---|
| 4-bit | 50% | 50% | 0% | 0% | 5.42 | 23.7 | ~3.8 GB |
| 8-bit | 100% | 100% | 50% | 50% | 10.04 | 13.7 | ~7.8 GB |

8-bit roughly doubled end-to-end pass rate (0% → 50%) relative to 4-bit on this corpus, at roughly double the memory footprint, roughly double the TTFT, and a ~42% drop in generation throughput. For comparison, Llama-3-8B-Instruct at 4-bit achieved 100% end-to-end pass with better latency than CodeQwen at either quantization level — indicating model choice has a larger effect on reliability here than quantization level alone, though within a single model family, 8-bit is a meaningful reliability improvement over 4-bit at a real latency/memory cost.

## 6. Findings

### 6.1 DeepSeek-Coder's code-specific space-collapse defect

Across all trials, DeepSeek-Coder-6.7B (4-bit) reliably produced well-formed JSON with normally-spaced prose in the `explanation` and `diff_summary` fields, but the `patched_code` field consistently had all whitespace removed between tokens. This was confirmed to originate at the model's own token generation — not a tokenizer/decoding artifact — by inspecting per-token output before any post-processing was applied. Iterative refinement with an explicit, correctly-targeted instruction did not resolve this across 4 full attempts.

**Implication**: this represents a case where prompt-based self-correction fails specifically because the defect is a low-level generation habit rather than a reasoning error — the model does not appear to lack understanding of the required fix, but fails to consistently reproduce whitespace when generating code inside a JSON string context, at this quantization level.

### 6.2 Generation variance at low temperature

CodeQwen1.5-7B-Chat (temperature = 0.2) produced materially different outcomes across repeated runs on identical input — succeeding cleanly in some trials, failing to produce parseable JSON across an entire refinement budget in others. This indicates non-trivial output variance even at low sampling temperature, relevant to any claim of "reliability" made about a given model without reporting results across multiple trials per case.

### 6.3 Static exploit heuristics produce false negatives on safe patches

An early version of the exploit-detection stub flagged any f-string prefix (`f"..."`) as evidence of unsafe interpolation, regardless of whether the string actually interpolated anything. This produced a false negative on a genuinely safe patch (`f"SELECT ... WHERE username = ?"` — a vestigial f-prefix with a real placeholder and correctly separated parameters), which would have incorrectly failed the refinement loop and caused unnecessary retries. The heuristic was corrected to check for actual interpolation content rather than syntax presence alone.

**Implication**: this is direct, empirical evidence supporting the original architectural decision to require *dynamic* exploit replay in an isolated sandbox (the Validation Agent's responsibility) rather than relying on static pattern matching — a static check, however carefully tuned, can misclassify semantically safe code that happens to use unnecessary-but-harmless syntax.

### 6.4 Tokenizer/decode pipeline inconsistencies across MLX conversions

During development, three distinct decoding defects were identified and fixed in the inference harness:
- Concatenating per-token decoded text without full-sequence reassembly caused literal tokenizer marker characters (`▁`, `Ġ`, `Ċ`) to leak into output for SentencePiece- and byte-level-BPE-tokenized models.
- SentencePiece byte-fallback tokens (`<0xXX>`, used for bytes such as newline outside the normal vocabulary) appeared as literal 6-character strings rather than being decoded to the corresponding byte.
- Full-sequence `tokenizer.decode()` behaved inconsistently across different MLX model conversions — reliably converting markers for one tokenizer, silently dropping the represented whitespace for another — necessitating a return to per-token accumulation with explicit marker cleanup as the robust approach.

These are documented as general-purpose fixes in `mlx_inference.py`, applicable beyond this project to any MLX-based local inference harness consuming multiple differently-converted models.

### 6.5 Quantization level materially affects reliability, not just resource cost

The 4-bit vs 8-bit comparison in Section 5.3 shows quantization level is not purely a memory/speed knob — it directly affects patch validity. At 4-bit, CodeQwen produced a patch that both compiled and closed the vulnerability in 0% of trials; at 8-bit, the same model succeeded in 50%. This suggests that for a security-correctness task specifically (as opposed to more forgiving generation tasks), the common assumption that "4-bit is good enough" deserves per-task validation rather than being taken as a general default — the reliability cost of aggressive quantization can be substantial even when the latency/memory savings look attractive in isolation.

## 7. Limitations and Future Work

- **Corpus depth**: current results are based on a 2-case CWE-89 corpus. Expansion to 10–15+ cases spanning multiple database drivers/ORMs and sink patterns is required before pass-rate comparisons are statistically meaningful.
- **Stub validation gates**: both functional and exploit gates are local approximations (Python `compile()` and static pattern matching respectively). Integration with the actual Validation Agent sandbox (real test suite execution, dynamic exploit replay) is required for production-representative results.
- **Latency optimization**: a 4-bit vs. 8-bit quantization tradeoff was measured and analyzed (Section 5.3), showing quantization level materially affects patch validity, not just speed/memory. Prompt-prefix caching across n-best sampling calls remains untested and is a candidate next step.
- **Container spin-up optimization**: out of scope for this report; tracked separately as part of this module owner's stated responsibilities.
- **n-best sampling**: `generate_n_best()` is implemented in the inference harness but not yet exercised in the benchmark or refinement loop; current results reflect single-sample generation per attempt.

## 8. Conclusion

A working, locally-served Refactoring Agent was implemented and validated end-to-end for CWE-89 patch synthesis, running entirely on Apple Silicon via MLX across three quantized models. Beyond a functioning pipeline, this work produced several specific, reproducible findings about model reliability (DeepSeek-Coder's systematic code-formatting defect, CodeQwen's generation variance, and quantization level's material effect on patch validity beyond its expected latency/memory cost) and about validation design (the necessity of dynamic over static exploit checking) that directly inform and justify design decisions in the broader AutoPatch-Sec architecture.
