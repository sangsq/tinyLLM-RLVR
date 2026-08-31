from cs336_alignment.checkpoint import (
    add_lora_adapter,
    parameter_counts,
    save_model_and_tokenizer,
)


def test_lora_freezes_base_and_saves_adapter(tiny_train_model, tokenizer, tmp_path):
    model = add_lora_adapter(tiny_train_model, rank=2)
    counts = parameter_counts(model)

    assert 0 < counts["trainable_parameters"] < counts["total_parameters"]
    assert all(
        parameter.requires_grad == ("lora_" in name)
        for name, parameter in model.named_parameters()
    )

    assert save_model_and_tokenizer(model, tokenizer, tmp_path) == "adapter"
    assert (tmp_path / "adapter_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert (tmp_path / "tokenizer.json").is_file()
