"""Shared, small-scale on-policy GRPO experiment loop."""

from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from cs336_alignment.checkpoint import (
    get_model_and_tokenizer,
    parameter_counts,
    save_model_and_tokenizer,
)
from cs336_alignment.components import grpo_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.hf_utils import HFCompletionServer


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B"
PROMPTS = {
    "r1_zero": "r1_zero.prompt",
    "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
}
DATA = ROOT / "data" / "gsm8k"


@dataclass
class RunConfig:
    experiment: str
    name: str
    seed: int
    learning_rate: float
    adapter: str = "full"
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_target_modules: str = "all-linear"
    prompt: str = "r1_zero_three_shot"
    baseline: str = "mean"
    advantage_normalizer: str = "std"
    loss_normalization: str = "sequence"
    rollout_steps: int = 40
    questions_per_rollout: int = 16
    group_size: int = 16
    max_tokens: int = 128
    n_val: int = 256
    rollout_generation_batch_size: int = 1
    evaluation_batch_size: int = 8

    def __post_init__(self) -> None:
        if self.adapter not in {"full", "lora"}:
            raise ValueError(f"Unknown adapter mode: {self.adapter}")
        if self.adapter == "lora" and self.lora_rank is None:
            raise ValueError("lora_rank is required for LoRA runs")


def log_activity(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with (ROOT / "ACTIVITY_LOG.md").open("a") as handle:
        handle.write(f"- {timestamp} — Revised GRPO runner: {message}\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def ground_truth(example: dict) -> str:
    return example["answer"].rstrip().splitlines()[-1].removeprefix("####").strip()


def repeat(items: list[str], n: int) -> list[str]:
    return [item for item in items for _ in range(n)]


def score_outputs(outputs, examples) -> tuple[dict, list[dict]]:
    rows = []
    for output, example in zip(outputs, examples):
        scores = r1_zero_reward_fn(output.text, ground_truth(example))
        rows.append({
            "question": example["question"],
            "ground_truth": ground_truth(example),
            "response": output.text,
            "length": len(output.token_ids),
            **scores,
        })
    return {
        "reward": sum(row["reward"] for row in rows) / len(rows),
        "format_reward": sum(row["format_reward"] for row in rows) / len(rows),
        "response_length": sum(row["length"] for row in rows) / len(rows),
    }, rows


def evaluate(generator, examples, prompt_fn, seed, max_tokens, batch_size):
    prompts = [prompt_fn(example["question"]) for example in examples]
    outputs = generator.generate_completions(
        prompts,
        {
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "max_tokens": max_tokens,
            "seed": seed,
            "stop": ["</answer>"],
        },
        batch_size=batch_size,
    )
    return score_outputs(outputs, examples)


def save_result(result: dict) -> Path:
    output_dir = ROOT / "results" / result["config"]["experiment"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{result["config"]["name"]}.json'
    path.write_text(json.dumps(result, indent=2))
    return path

def is_complete(config: RunConfig) -> bool:
    path = ROOT / "results" / config.experiment / f"{config.name}.json"
    if not path.is_file():
        return False
    result = json.loads(path.read_text())
    checkpoint = result.get("checkpoint", {}).get("path")
    return (
        len(result.get("train", [])) == config.rollout_steps
        and checkpoint is not None
        and (ROOT / checkpoint).is_dir()
    )


def run(config: RunConfig, model_id: str = MODEL) -> Path:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1; physical GPU 1 must be the only visible GPU")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    train = load_jsonl(DATA / "train.jsonl")[:6400]
    val = load_jsonl(DATA / "test.jsonl")[: config.n_val]
    random.Random(config.seed).shuffle(train)
    template = (ROOT / "cs336_alignment" / "prompts" / PROMPTS[config.prompt]).read_text()
    prompt_fn = lambda question: template.format(question=question)
    long_prompt = config.prompt == "r1_zero_three_shot"

    log_activity(f"started {config.name}: {asdict(config)}")
    started = time.time()
    model, tokenizer = get_model_and_tokenizer(
        model_id,
        "cuda:0",
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_target_modules=config.lora_target_modules,
    )
    counts = parameter_counts(model)
    generator = HFCompletionServer(model, tokenizer)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "n": config.group_size,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "stop": ["</answer>"],
    }
    result = {
        "config": {
            **asdict(config),
            "model": model_id,
            "physical_gpu": 1,
            **counts,
        },
        "hardware": torch.cuda.get_device_name(0),
        "train": [],
        "eval": [],
        "rollouts": [],
    }

    initial, rows = evaluate(
        generator,
        val,
        prompt_fn,
        config.seed,
        config.max_tokens,
        batch_size=config.evaluation_batch_size if long_prompt else 32,
    )
    result["eval"].append({"rollout_step": 0, "update": 0, **initial})
    result["rollouts"].append({"stage": "before", "examples": rows[:3]})

    for step in range(1, config.rollout_steps + 1):
        start = ((step - 1) * config.questions_per_rollout) % len(train)
        examples = train[start : start + config.questions_per_rollout]
        prompts = [prompt_fn(example["question"]) for example in examples]
        repeated_prompts = repeat(prompts, config.group_size)
        answers = repeat([ground_truth(example) for example in examples], config.group_size)
        sampling["seed"] = config.seed * 10_000 + step
        responses = [
            output.text
            for output in generator.generate_completions(
                prompts,
                sampling,
                batch_size=config.rollout_generation_batch_size if long_prompt else len(prompts),
            )
        ]
        batch_size = len(responses)
        loss, stats = grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=max(1, batch_size // (4 if long_prompt else 16)),
            max_grad_norm=1.0,
            reward_fn=r1_zero_reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=responses,
            repeated_ground_truths=answers,
            group_size=config.group_size,
            baseline=config.baseline,
            advantage_normalizer=config.advantage_normalizer,
            loss_normalization=config.loss_normalization,
            normalization_constant=(
                config.questions_per_rollout * config.group_size * config.max_tokens
                if config.loss_normalization == "constant"
                else None
            ),
        )
        result["train"].append({
            "rollout_step": step,
            "update": step,
            "loss": float(loss.detach().cpu()),
            **{
                key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
                for key, value in stats.items()
            },
        })

        if step % max(1, config.rollout_steps // 5) == 0 or step == config.rollout_steps:
            summary, rows = evaluate(
                generator,
                val,
                prompt_fn,
                config.seed + step,
                config.max_tokens,
                batch_size=config.evaluation_batch_size if long_prompt else 32,
            )
            result["eval"].append({"rollout_step": step, "update": step, **summary})
            if step == config.rollout_steps:
                result["rollouts"].append({"stage": "after", "examples": rows[:3]})
        result["elapsed_seconds"] = time.time() - started
        save_result(result)
        print(
            f"{config.name}: {step}/{config.rollout_steps}, "
            f"rollout reward={result['train'][-1]['mean_reward']:.3f}",
            flush=True,
        )

    checkpoint = ROOT / "checkpoints" / config.experiment / config.name
    result["checkpoint"] = {
        "path": str(checkpoint.relative_to(ROOT)),
        "kind": save_model_and_tokenizer(model, tokenizer, checkpoint),
    }
    result["elapsed_seconds"] = time.time() - started
    path = save_result(result)
    log_activity(
        f"finished {config.name} in {result['elapsed_seconds'] / 60:.1f} min; "
        f"final reward={result['eval'][-1]['reward']:.4f}; saved {path.relative_to(ROOT)} "
        f"and {result['checkpoint']['kind']} at {result['checkpoint']['path']}"
    )
    del optimizer, generator, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return path
