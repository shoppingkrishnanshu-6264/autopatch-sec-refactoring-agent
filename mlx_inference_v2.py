"""
mlx_inference.py
-----------------
Local MLX inference harness for the AutoPatch-Sec Refactoring Agent.

Responsibilities covered:
  - Load a quantized instruction-tuned coding model via mlx-lm
    (CodeQwen-7B, Llama-3-8B-Instruct, etc., in MLX 4-bit/8-bit format)
  - Generate a patch for a given prompt with streaming
  - Instrument latency: time-to-first-token (TTFT), tokens/sec, total tokens
  - Expose a simple `generate_patch()` call the rest of the pipeline can use

Requirements:
    pip install mlx-lm

Model prep (one-time, outside this script):
    # Convert + quantize a HF model to MLX format, e.g. 4-bit:
    python -m mlx_lm.convert \
        --hf-path Qwen/CodeQwen1.5-7B-Chat \
        --mlx-path ./models/codeqwen-7b-mlx-4bit \
        -q --q-bits 4

Usage:
    python mlx_inference.py --model ./models/codeqwen-7b-mlx-4bit --prompt-file prompt.txt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


@dataclass
class GenerationMetrics:
    """Latency/throughput metrics for one generation call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_seconds: float = 0.0          # time to first token
    total_seconds: float = 0.0         # wall clock for full generation
    tokens_per_second: float = 0.0     # completion throughput, post-TTFT

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_seconds": round(self.ttft_seconds, 4),
            "total_seconds": round(self.total_seconds, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


@dataclass
class PatchGenerationResult:
    raw_text: str
    metrics: GenerationMetrics
    candidate_index: int = 0


class RefactoringAgentModel:
    """
    Thin wrapper around mlx-lm for repeated, instrumented calls from the
    Refactoring Agent. Loads the model once; reuses across n-best sampling
    so the prompt-processing (prefill) cost is not paid from cold state
    each call, and KV structure stays warm for shared-prefix prompts.
    """

    def __init__(self, model_path: str, max_tokens: int = 1024):
        self.model_path = model_path
        self.max_tokens = max_tokens
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_path)
        self.load_seconds = time.perf_counter() - t0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        seed: Optional[int] = None,
    ) -> PatchGenerationResult:
        """
        Single generation call with TTFT / throughput instrumentation.
        Low temperature by default: patch synthesis wants determinism,
        not creativity. Raise temperature only for n-best diversity sampling.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampler = make_sampler(temp=temperature, top_p=top_p)

        # Accumulate TOKEN IDS, not per-chunk .text. Concatenating each
        # token's individually-decoded text skips the byte-level/SentencePiece
        # reassembly step, so space/newline markers (Ġ, ▁, Ċ) leak through
        # literally instead of becoming real whitespace. Decoding the full
        # id sequence at the end avoids that.
        token_ids: list[int] = []
        first_token_time: Optional[float] = None

        start = time.perf_counter()
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_ids.append(response.token)

        end = time.perf_counter()
        ttft = (first_token_time - start) if first_token_time else (end - start)
        total = end - start
        completion_tokens = len(token_ids)
        post_ttft = max(total - ttft, 1e-6)
        tps = completion_tokens / post_ttft if completion_tokens else 0.0

        full_text = self.tokenizer.decode(token_ids) if token_ids else ""

        metrics = GenerationMetrics(
            prompt_tokens=len(self.tokenizer.encode(prompt)),
            completion_tokens=completion_tokens,
            ttft_seconds=ttft,
            total_seconds=total,
            tokens_per_second=tps,
        )
        return PatchGenerationResult(raw_text=full_text, metrics=metrics)

    def generate_n_best(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int = 3,
        temperature: float = 0.4,
    ) -> list[PatchGenerationResult]:
        """
        Generate n candidate patches for downstream sandbox validation.
        Same shared prefix (system + user prompt) across calls — if you
        move to a server-based setup (mlx_lm.server) rather than this
        in-process loop, enable prompt-caching there to avoid re-prefilling
        the shared prefix n times.
        """
        results = []
        for i in range(n):
            r = self.generate(system_prompt, user_prompt, temperature=temperature)
            r.candidate_index = i
            results.append(r)
        return results


def main():
    parser = argparse.ArgumentParser(description="MLX Refactoring Agent inference")
    parser.add_argument("--model", required=True, help="Path to local MLX model dir")
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--n-best", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    system_prompt = (
        open(args.system_prompt_file).read()
        if args.system_prompt_file
        else "You are a secure code refactoring assistant."
    )
    user_prompt = open(args.prompt_file).read()

    agent = RefactoringAgentModel(args.model, max_tokens=args.max_tokens)
    print(f"[load] model ready in {agent.load_seconds:.2f}s\n")

    if args.n_best > 1:
        results = agent.generate_n_best(system_prompt, user_prompt, n=args.n_best)
    else:
        results = [agent.generate(system_prompt, user_prompt)]

    for r in results:
        print(f"--- candidate {r.candidate_index} ---")
        print(r.raw_text)
        print(json.dumps(r.metrics.to_dict()))
        print()


if __name__ == "__main__":
    main()
