"""Compare base and best-GRPO pass@N on GSM8K with vLLM."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B"
ADAPTER = ROOT / "checkpoints/02_lora_rank/grpo_lora_r8_seed0"
MERGED = ROOT / "checkpoints/02_lora_rank/grpo_lora_r8_seed0_merged"
RESULTS = ROOT / "results/04_pass_at_n"


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with (ROOT / "ACTIVITY_LOG.md").open("a") as handle:
        handle.write(f"- {timestamp} — Experiment 4 pass@N: {message}\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def ground_truth(example: dict) -> str:
    return example["answer"].rstrip().splitlines()[-1].removeprefix("####").strip()


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Unbiased probability that k samples contain at least one of c successes."""
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if not 0 <= correct <= n:
        raise ValueError(f"correct must be in [0, {n}], got {correct}")
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def curve(records: list[dict], n_samples: int) -> list[float]:
    return [
        sum(pass_at_k(n_samples, row["n_correct"], k) for row in records) / len(records)
        for k in range(1, n_samples + 1)
    ]


def make_sampling_params(n_samples: int, max_tokens: int, seed: int, n: int):
    from vllm import SamplingParams

    return [
        SamplingParams(
            n=n_samples,
            temperature=1.0,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=seed + i,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        for i in range(n)
    ]


def score_requests(requests, examples: list[dict]) -> list[dict]:
    records = []
    for request, example in zip(requests, examples, strict=True):
        samples = []
        for output in request.outputs:
            scores = r1_zero_reward_fn(output.text, ground_truth(example))
            samples.append({
                "response": output.text,
                "length": len(output.token_ids),
                **scores,
            })
        records.append({
            "question": example["question"],
            "ground_truth": ground_truth(example),
            "n_correct": int(sum(sample["reward"] for sample in samples)),
            "n_formatted": int(sum(sample["format_reward"] for sample in samples)),
            "samples": samples,
        })
    return records


def save_condition(name: str, records: list[dict], config: dict, elapsed: float) -> Path:
    payload = {
        "condition": name,
        "config": config,
        "n_examples": len(records),
        "pass_at_n": curve(records, config["n_samples"]),
        "sample_accuracy": sum(row["n_correct"] for row in records)
        / (len(records) * config["n_samples"]),
        "format_rate": sum(row["n_formatted"] for row in records)
        / (len(records) * config["n_samples"]),
        "elapsed_seconds": elapsed,
        "records": records,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("base", "grpo"), required=True)
    parser.add_argument("--n-examples", type=int, default=1319)
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.15)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1; physical GPU 1 must be the only visible GPU")
    if args.condition == "grpo" and not MERGED.is_dir():
        raise FileNotFoundError(f"Run scripts/merge_best_grpo.py first: {MERGED}")

    from vllm import LLM

    examples = load_jsonl(ROOT / "data/gsm8k/test.jsonl")[: args.n_examples]
    template = (ROOT / "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt").read_text()
    prompts = [template.format(question=row["question"]) for row in examples]
    config = {
        "base_model": MODEL,
        "grpo_adapter": str(ADAPTER.relative_to(ROOT)),
        "grpo_inference_checkpoint": str(MERGED.relative_to(ROOT)),
        "grpo_selection": "highest final 256-example validation reward among saved GRPO runs",
        "dataset": "GSM8K test",
        "prompt": "r1_zero_three_shot",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "n_samples": args.n_samples,
        "seed_policy": "paired per-example seeds",
        "seed": args.seed,
        "physical_gpu": 1,
    }
    log(f"starting {args.condition}: {config}, n_examples={len(examples)}")
    engine = LLM(
        model=MODEL if args.condition == "base" else str(MERGED),
        dtype="bfloat16",
        max_model_len=768,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=64,
        enforce_eager=True,
    )
    params = make_sampling_params(args.n_samples, args.max_tokens, args.seed, len(prompts))
    started = time.time()
    outputs = engine.generate(prompts, params)
    records = score_requests(outputs, examples)
    elapsed = time.time() - started
    path = save_condition(args.condition, records, config, elapsed)
    values = curve(records, args.n_samples)
    log(
        f"finished {args.condition}: pass@1={values[0]:.4f}, "
        f"pass@{args.n_samples}={values[-1]:.4f}, elapsed={elapsed / 60:.1f} min; "
        f"saved {path.relative_to(ROOT)}"
    )
    print(
        f"{args.condition}: pass@1={values[0]:.2%}, "
        f"pass@{args.n_samples}={values[-1]:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
