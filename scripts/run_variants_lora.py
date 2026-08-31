"""Experiment 3: on-policy estimator variants using all-linear LoRA."""

import argparse
import traceback

from cs336_alignment.experiments import RunConfig, is_complete, log_activity, run


VARIANTS = {
    "grpo_constant": ("mean", "std"),
    "dr_grpo": ("mean", "none"),
    "rft": ("none", "none"),
    "maxrl": ("mean", "mean"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    variants = (args.variant,) if args.variant else tuple(VARIANTS)
    seeds = (args.seed,) if args.seed is not None else (0, 1, 2)
    for name in variants:
        baseline, normalizer = VARIANTS[name]
        for seed in seeds:
            config = RunConfig(
                "03_lora_variants",
                f"{name}_lora_r{args.rank}_seed{seed}",
                seed,
                learning_rate=1e-4,
                adapter="lora",
                lora_rank=args.rank,
                lora_alpha=2 * args.rank,
                baseline=baseline,
                advantage_normalizer=normalizer,
                loss_normalization="constant",
                rollout_steps=30,
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
