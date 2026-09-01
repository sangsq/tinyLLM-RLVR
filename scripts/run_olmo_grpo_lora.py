"""Experiment 5: matched rank-8 LoRA GRPO on OLMo-2-0425-1B."""

import argparse
import traceback

from cs336_alignment.experiments import RunConfig, is_complete, log_activity, run


MODEL = "allenai/OLMo-2-0425-1B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    config = RunConfig(
        "05_olmo_grpo",
        "olmo_grpo_lora_r8_seed0",
        0,
        learning_rate=1e-4,
        adapter="lora",
        lora_rank=8,
        lora_alpha=16,
        rollout_generation_batch_size=4,
    )
    if args.skip_completed and is_complete(config):
        print(f"{config.name}: already complete", flush=True)
        return
    try:
        run(config, model_id=MODEL)
    except Exception:
        log_activity(f"FAILED {config.name}: {traceback.format_exc().replace(chr(10), ' | ')}")
        raise


if __name__ == "__main__":
    main()
