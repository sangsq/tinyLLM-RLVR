# tinyLLM-RLVR

Small-scale reasoning RL on GSM8K with Qwen2.5-0.5B.

The project implements GRPO with LoRA; studies LoRA rank ablation, and several other on-policy RL estimators with LoRA.

Based on and follows closely the [assignment 5 of Stanford CS336 (Spring 2026)](https://github.com/stanford-cs336/assignment5-alignment).



## Setup

As in previous assignments, we use `uv` to manage dependencies.

Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```
