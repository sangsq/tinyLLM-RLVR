"""Compare Qwen and OLMo before/after GRPO with GSM8K pass@N."""

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
RESULTS = ROOT / "results/04_pass_at_n"
CONDITIONS = {
    "base": {
        "family": "Qwen2.5-0.5B",
        "base_model": "Qwen/Qwen2.5-0.5B",
        "inference_model": "Qwen/Qwen2.5-0.5B",
        "training_result": None,
    },
    "grpo": {
        "family": "Qwen2.5-0.5B",
        "base_model": "Qwen/Qwen2.5-0.5B",
        "inference_model": str(ROOT / "checkpoints/02_lora_rank/grpo_lora_r8_seed0_merged"),
        "training_result": "results/02_lora_rank/grpo_lora_r8_seed0.json",
    },
    "olmo_base": {
        "family": "OLMo-2-0425-1B",
        "base_model": "allenai/OLMo-2-0425-1B",
        "inference_model": "allenai/OLMo-2-0425-1B",
        "training_result": None,
    },
    "olmo_grpo": {
        "family": "OLMo-2-0425-1B",
        "base_model": "allenai/OLMo-2-0425-1B",
        "inference_model": str(ROOT / "checkpoints/05_olmo_grpo/olmo_grpo_lora_r8_seed0_merged"),
        "training_result": "results/05_olmo_grpo/olmo_grpo_lora_r8_seed0.json",
    },
}


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


def result_name(condition: str, temperature: float) -> str:
    if temperature == 1.0:
        return condition
    return f"{condition}_t{str(temperature).replace('.', 'p')}"


def make_sampling_params(
    n_samples: int, max_tokens: int, temperature: float, seed: int, start: int, n: int
):
    from vllm import SamplingParams

    return [
        SamplingParams(
            n=n_samples,
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=seed + start + i,
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
            "mean_length": sum(sample["length"] for sample in samples) / len(samples),
            "representative_samples": samples[:3],
        })
    return records


def save_condition(
    condition: str, output_name: str, records: list[dict], config: dict, elapsed: float
) -> Path:
    payload = {
        "condition": condition,
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
    path = RESULTS / f"{output_name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--n-examples", type=int, default=1319)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, choices=(0.5, 1.0, 1.5), default=1.0)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--request-batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1; physical GPU 1 must be the only visible GPU")
    condition = CONDITIONS[args.condition]
    inference_model = condition["inference_model"]
    if args.condition.endswith("grpo") and not Path(inference_model).is_dir():
        raise FileNotFoundError(f"Merged checkpoint not found: {inference_model}")

    from vllm import LLM

    examples = load_jsonl(ROOT / "data/gsm8k/test.jsonl")[: args.n_examples]
    template = (ROOT / "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt").read_text()
    prompts = [template.format(question=row["question"]) for row in examples]
    config = {
        "model_family": condition["family"],
        "base_model": condition["base_model"],
        "inference_model": (
            str(Path(inference_model).relative_to(ROOT))
            if Path(inference_model).is_absolute()
            else inference_model
        ),
        "training_result": condition["training_result"],
        "adapter": "all-linear LoRA r=8, alpha=16" if args.condition.endswith("grpo") else None,
        "dataset": "GSM8K test",
        "n_examples": len(examples),
        "prompt": "r1_zero_three_shot",
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "n_samples": args.n_samples,
        "seed_policy": "paired per-example seeds",
        "seed": args.seed,
        "physical_gpu": 1,
        "vllm": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "enforce_eager": True,
        },
    }
    output_name = result_name(args.condition, args.temperature)
    result_path = RESULTS / f"{output_name}.json"
    records, previous_elapsed = [], 0.0
    if result_path.is_file() and not args.overwrite:
        previous = json.loads(result_path.read_text())
        if previous.get("config") == config:
            records = previous["records"]
            previous_elapsed = previous["elapsed_seconds"]
    if len(records) == len(examples):
        print(f"{output_name}: already complete", flush=True)
        return
    log(
        f"starting {args.condition}: {config}, n_examples={len(examples)}, "
        f"resuming_at={len(records)}"
    )
    engine = LLM(
        model=inference_model,
        dtype="bfloat16",
        max_model_len=768,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
    )
    started = time.time()
    for start in range(len(records), len(prompts), args.request_batch_size):
        stop = min(start + args.request_batch_size, len(prompts))
        params = make_sampling_params(
            args.n_samples, args.max_tokens, args.temperature, args.seed, start, stop - start
        )
        outputs = engine.generate(prompts[start:stop], params, use_tqdm=False)
        records.extend(score_requests(outputs, examples[start:stop]))
        elapsed = previous_elapsed + time.time() - started
        path = save_condition(args.condition, output_name, records, config, elapsed)
        print(f"{output_name}: {stop}/{len(prompts)} questions", flush=True)
    values = curve(records, args.n_samples)
    log(
        f"finished {output_name}: pass@1={values[0]:.4f}, "
        f"pass@{args.n_samples}={values[-1]:.4f}, elapsed={elapsed / 60:.1f} min; "
        f"saved {path.relative_to(ROOT)}"
    )
    print(
        f"{output_name}: pass@1={values[0]:.2%}, "
        f"pass@{args.n_samples}={values[-1]:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
