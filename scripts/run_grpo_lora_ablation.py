"""Experiment 2: all-linear LoRA rank ablation under standard GRPO."""

import argparse
import traceback

from cs336_alignment.experiments import RunConfig, is_complete, log_activity, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(2, 4, 8, 16))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    ranks = (args.rank,) if args.rank is not None else (2, 4, 8, 16)
    seeds = (args.seed,) if args.seed is not None else (0, 1, 2)
    for rank in ranks:
        for seed in seeds:
            config = RunConfig(
                "02_lora_rank",
                f"grpo_lora_r{rank}_seed{seed}",
                seed,
                learning_rate=1e-4,
                adapter="lora",
                lora_rank=rank,
                lora_alpha=2 * rank,
                rollout_generation_batch_size=4,
            )
            if args.skip_completed and is_complete(config):
                print(f"{config.name}: already complete", flush=True)
                continue
            try:
                run(config)
            except Exception:
                log_activity(f"FAILED {config.name}: {traceback.format_exc().replace(chr(10), ' | ')}")
                raise


if __name__ == "__main__":
    main()
