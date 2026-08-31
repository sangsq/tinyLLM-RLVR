from __future__ import annotations
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer


def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizer,
        ) -> dict[str, torch.Tensor]:
    
    def _output_mask(input, output, max_len):
        x = len(input)
        y = len(input) + len(output)
        return [True if x <= i < y else False for i in range(max_len)]
    
    def _padding(ids, max_len):
        l = len(ids)
        ids.extend([0] * (max_len-l))
        return ids
    
    prompt_ids = [tokenizer.encode(s) for s in prompt_strs]
    output_ids = [tokenizer.encode(s) for s in output_strs]
    max_len = max([len(a)+len(b) for a,b in zip(prompt_ids, output_ids)])


    concat_ids = torch.tensor([_padding(a+b, max_len) for a,b in zip(prompt_ids, output_ids)])
    mask = torch.tensor([_output_mask(a, b, max_len) for a,b in zip(prompt_ids, output_ids)])

    result = {}
    result["input_ids"] = concat_ids[:, :-1]
    result["labels"] = concat_ids[:, 1:]
    result["response_mask"] = mask[:, 1:]

    return result



def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:

    logits = model(input_ids)["logits"]
    log_prob = F.log_softmax(logits, dim=-1)

    result = {}
    result["log_probs"] = log_prob.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        result["token_entropy"] = -(log_prob * log_prob.exp()).sum(dim=-1)

    return result
    


def compute_rollout_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        ) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = [reward_fn(x, y) for x,y in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([r["reward"] for r in rewards])
    metadata = dict(
        mean_reward=raw_rewards.mean().tolist(),
        mean_format_rewards=torch.tensor([r["format_reward"] for r in rewards]).mean().tolist()
                    )
    return raw_rewards, metadata



def compute_group_normalized_rewards(
        raw_rewards: torch.Tensor,
        group_size: int,
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std",
        ):
    grouped_rewards = raw_rewards.reshape(-1, group_size)
    mu = grouped_rewards.mean(dim=1, keepdim=True)
    sigma = grouped_rewards.std(dim=1, keepdim=True)

    if baseline=="mean":
        b = mu
    elif baseline=="none":
        b = torch.zeros_like(mu)
    
    if advantage_normalizer=="std":
        s = sigma + advantage_eps
    elif advantage_normalizer=="mean":
        s = mu + advantage_eps
    elif advantage_normalizer=="none":
        s = torch.ones_like(sigma)
    
    advantages = ((grouped_rewards - b) / s).reshape(-1)
    metadata = {}
    return advantages, raw_rewards, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = raw_rewards_or_advantages.reshape(-1, 1)

    def masked_mean(tensor: torch.Tensor) -> torch.Tensor:
        if response_mask is None:
            return tensor.mean()
        mask = response_mask.to(tensor.dtype)
        return (tensor * mask).sum() / mask.sum().clamp_min(1)
    if importance_reweighting_method == "none":
        return -advantages * policy_log_probs, {}

    if old_log_probs is None:
        raise ValueError("old_log_probs is required for off-policy losses")
    if old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("old_log_probs and policy_log_probs must have the same shape")

    log_ratio = policy_log_probs - old_log_probs
    if importance_reweighting_method == "noclip":
        ratio = log_ratio.exp()
        return -advantages * ratio, {"importance_ratio": masked_mean(ratio.detach())}

    if cliprange is None:
        raise ValueError(f"cliprange is required for {importance_reweighting_method}")

    if importance_reweighting_method == "grpo":
        ratio = log_ratio.exp()
        unclipped = advantages * ratio
        clipped = advantages * ratio.clamp(1 - cliprange, 1 + cliprange)
        objective = torch.minimum(unclipped, clipped)
        clipped_tokens = objective != unclipped
        return -objective, {
            "importance_ratio": masked_mean(ratio.detach()),
            "clip_fraction": masked_mean(clipped_tokens.detach().float()),
        }

    if importance_reweighting_method == "gspo":
        if response_mask is None:
            raise ValueError("response_mask is required for GSPO")
        mask = response_mask.to(log_ratio.dtype)
        response_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1)
        sequence_ratio = ((log_ratio * mask).sum(dim=1, keepdim=True) / response_lengths).exp()
        unclipped = advantages * sequence_ratio
        clipped = advantages * sequence_ratio.clamp(1 - cliprange, 1 + cliprange)
        objective = torch.minimum(unclipped, clipped).expand_as(policy_log_probs)
        clipped_sequences = unclipped != torch.minimum(unclipped, clipped)
        return -objective, {
            "importance_ratio": sequence_ratio.detach().mean(),
            "clip_fraction": clipped_sequences.detach().float().mean(),
        }

    raise ValueError(f"Unknown importance reweighting method: {importance_reweighting_method}")
    

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    ) -> torch.Tensor:
    sequence_loss = (per_token_policy_gradient_loss * mask).sum(dim=1)
    if loss_normalization == "sequence":
        return (sequence_loss / mask.sum(dim=1).clamp_min(1)).mean()
    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant is required for constant normalization")
        return sequence_loss.sum() / normalization_constant
    raise ValueError(f"Unknown loss normalization: {loss_normalization}")



def grpo_train_step(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: Optimizer,
        gradient_accumulation_steps: int,
        max_grad_norm: float | None,
        reward_fn: Callable[[str, str], dict[str, float]],
        repeated_prompts: list[str],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        # Reward normalization
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std",
        # Importance reweighting and clipping
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        # Loss normalization
        loss_normalization: Literal["sequence", "constant"] = "sequence",
        normalization_constant: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    raw_rewards, reward_stats = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, _, advantage_stats = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)

    # Zero-advantage samples have exactly zero gradient. Pruning them is especially
    # helpful for RFT and homogeneous GRPO groups.
    active = advantages != 0
    if not active.any():
        optimizer.zero_grad(set_to_none=True)
        zero = torch.tensor(0.0, device=model.device)
        return zero, {
            **reward_stats,
            **advantage_stats,
            "avg_rewards": reward_stats["mean_reward"],
            "gradient_norm": 0.0,
            "token_entropy": 0.0,
            "active_fraction": 0.0,
        }

    device = model.device
    input_ids = batch["input_ids"][active].to(device)
    labels = batch["labels"][active].to(device)
    response_mask = batch["response_mask"][active].to(device)
    advantages = advantages[active].to(device)
    active_old_log_probs = None
    if old_log_probs is not None:
        active_old_log_probs = old_log_probs[active].to(device)

    optimizer.zero_grad(set_to_none=True)
    model.train()
    n_total = len(repeated_prompts)
    n_active = len(input_ids)
    n_microbatches = min(gradient_accumulation_steps, n_active)
    microbatch_size = (n_active + n_microbatches - 1) // n_microbatches
    total_loss = torch.tensor(0.0, device=device)
    entropy_sum = torch.tensor(0.0, device=device)
    token_count = torch.tensor(0.0, device=device)
    loss_stats: dict[str, list[torch.Tensor]] = {}

    for start in range(0, n_active, microbatch_size):
        stop = min(start + microbatch_size, n_active)
        b_mask = response_mask[start:stop]
        outputs = get_response_log_probs(
            model,
            input_ids[start:stop],
            labels[start:stop],
            return_token_entropy=True,
        )
        b_old_log_probs = (
            None if active_old_log_probs is None else active_old_log_probs[start:stop]
        )
        per_token_loss, micro_stats = compute_policy_gradient_loss(
            advantages[start:stop],
            outputs["log_probs"],
            importance_reweighting_method,
            b_old_log_probs,
            cliprange,
            b_mask,
        )
        loss = aggregate_loss_across_microbatch(
            per_token_loss, b_mask, loss_normalization, normalization_constant
        )
        if loss_normalization == "sequence":
            # Preserve the original full-batch mean: pruned samples contribute
            # exactly zero, but they still belong in the denominator.
            loss = loss * ((stop - start) / n_total)
        loss.backward()
        total_loss += loss.detach()
        entropy_sum += (outputs["token_entropy"].detach() * b_mask).sum()
        token_count += b_mask.sum()
        for key, value in micro_stats.items():
            loss_stats.setdefault(key, []).append(value.detach())

    if max_grad_norm is None:
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().square().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
    else:
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata: dict[str, torch.Tensor | float] = {
        **reward_stats,
        **advantage_stats,
        "avg_rewards": reward_stats["mean_reward"],
        "gradient_norm": float(gradient_norm),
        "token_entropy": float(entropy_sum / token_count.clamp_min(1)),
        "active_fraction": float(active.float().mean()),
    }
    metadata.update({key: torch.stack(values).mean() for key, values in loss_stats.items()})
    return total_loss, metadata
