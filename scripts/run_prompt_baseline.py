"""Experiment 0: evaluate three prompts on the full GSM8K test set."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "00_prompt_baseline"
MODEL = "Qwen/Qwen2.5-0.5B"


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with (ROOT / "ACTIVITY_LOG.md").open("a") as handle:
        handle.write(f"- {timestamp} — Experiment 0: {message}\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def answer(example: dict) -> str:
    return example["answer"].rstrip().splitlines()[-1].removeprefix("####").strip()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1; physical GPU 1 must be the only visible GPU")
    RESULTS.mkdir(parents=True, exist_ok=True)
    examples = load_jsonl(ROOT / "data" / "gsm8k" / "test.jsonl")
    prompts = {
        "question_only": ("question_only.prompt", question_only_reward_fn, None),
        "r1_zero": ("r1_zero.prompt", r1_zero_reward_fn, ["</answer>"]),
        "r1_zero_three_shot": (
            "r1_zero_three_shot_gsm8k.prompt",
            r1_zero_reward_fn,
            ["</answer>"],
        ),
    }

    log(f"started {MODEL} on {len(examples)} GSM8K test questions using physical GPU 1")
    server = VLLMServer(MODEL, gpu=1, gpu_memory_utilization=0.8, logging_level="WARNING")
    server.start()
    try:
        for name, (filename, reward_fn, stop) in prompts.items():
            template = (ROOT / "cs336_alignment" / "prompts" / filename).read_text()
            sampling = {
                "temperature": 1.0,
                "n": 1,
                "max_tokens": 512,
                "seed": 0,
                "stop": stop,
                "include_stop_str_in_output": True,
            }
            outputs = server.generate_completions(
                [template.format(question=row["question"]) for row in examples], sampling
            )
            records = []
            for example, output in zip(examples, outputs):
                scores = reward_fn(output.text, answer(example))
                records.append({
                    "question": example["question"],
                    "ground_truth": answer(example),
                    "response": output.text,
                    "length": len(output.token_ids),
                    **scores,
                })
            payload = {
                "model": MODEL,
                "physical_gpu": 1,
                "prompt": name,
                "sampling": sampling,
                "n_examples": len(records),
                "accuracy": sum(row["reward"] for row in records) / len(records),
                "format_rate": sum(row["format_reward"] for row in records) / len(records),
                "mean_response_length": sum(row["length"] for row in records) / len(records),
                "records": records,
            }
            path = RESULTS / f"{name}.json"
            path.write_text(json.dumps(payload, indent=2))
            log(f"finished {name}: accuracy={payload['accuracy']:.4f}; saved {path.relative_to(ROOT)}")
            print(f"{name}: accuracy={payload['accuracy']:.2%}", flush=True)
    finally:
        server.stop()
    log("finished all prompt baselines")


if __name__ == "__main__":
    main()
