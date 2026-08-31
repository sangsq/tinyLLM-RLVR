from dataclasses import dataclass
import torch


@dataclass
class VLLMCompletion:
    text: str
    token_ids: list[int]
    finish_reason: str | None


class HFCompletionServer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_completions(
        self,
        prompts: list[str],
        sampling_params: dict,
        batch_size: int | None = None,
    ) -> list[VLLMCompletion]:
        n = sampling_params["n"]
        temperature = sampling_params["temperature"]
        do_sample = temperature > 0

        gen_kwargs = {
            "max_new_tokens": sampling_params["max_tokens"],
            "num_return_sequences": n,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = sampling_params.get("top_p", 1.0)
            if "top_k" in sampling_params:
                gen_kwargs["top_k"] = sampling_params["top_k"]

        if "seed" in sampling_params:
            torch.manual_seed(sampling_params["seed"])
            torch.cuda.manual_seed_all(sampling_params["seed"])

        if sampling_params.get("stop") is not None:
            gen_kwargs["stop_strings"] = sampling_params["stop"]
            gen_kwargs["tokenizer"] = self.tokenizer

        batches = [prompts]
        if batch_size is not None:
            batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

        self.model.eval()
        device = next(self.model.parameters()).device
        completions = []

        with torch.inference_mode():
            for batch in batches:
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(device)
                outputs = self.model.generate(**inputs, **gen_kwargs)

                prompt_len = inputs["input_ids"].shape[1]
                new_ids = outputs[:, prompt_len:]

                completions.extend(
                    VLLMCompletion(
                        text=self.tokenizer.decode(ids, skip_special_tokens=True),
                        token_ids=ids.tolist(),
                        finish_reason=None,
                    )
                    for ids in new_ids
                )

        return completions