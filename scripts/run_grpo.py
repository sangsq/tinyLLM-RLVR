"""Experiment 1: standard on-policy GRPO with full-parameter updates."""

import argparse
import traceback

from cs336_alignment.experiments import RunConfig, log_activity, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    for seed in (args.seed,) if args.seed is not None else (0, 1, 2):
        config = RunConfig("01_grpo", f"grpo_full_seed{seed}", seed, learning_rate=1e-5)
        try:
            run(config)
        except Exception:
            log_activity(f"FAILED {config.name}: {traceback.format_exc().replace(chr(10), ' | ')}")
            raise


if __name__ == "__main__":
    main()
