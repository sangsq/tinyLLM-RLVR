from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_lora_adapter(
    model,
    rank: int,
    alpha: int | None = None,
    target_modules: str | list[str] = "all-linear",
):
    """Freeze a causal LM and attach one LoRA adapter to its linear blocks."""
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        r=rank,
        lora_alpha=alpha or 2 * rank,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(model, config)


def get_model_and_tokenizer(
    model_id_or_dir: str,
    device: str,
    *,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
    lora_target_modules: str | list[str] = "all-linear",
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="eager" if device == "cpu" else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    if lora_rank is not None:
        model = add_lora_adapter(model, lora_rank, lora_alpha, lora_target_modules)
    return model, tokenizer


def parameter_counts(model) -> dict[str, int | float]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
    }


def save_model_and_tokenizer(model, tokenizer, output_dir: str | Path) -> str:
    """Save full weights for full FT, or adapter weights for a PEFT model."""
    from peft import PeftModel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return "adapter" if isinstance(model, PeftModel) else "full_model"
