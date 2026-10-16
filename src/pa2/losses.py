"""Response-only log-probabilities, the SFT loss, and the DPO loss.

SFT reduction (documented in the report): mean negative log-likelihood over all
unmasked response tokens in the batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .batching import IGNORE_INDEX, build_dpo_batch, build_sft_batch


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position log p(label) with the causal shift, plus the response mask.

    Position ``t`` of the logits predicts token ``t + 1``. Returns tensors of shape
    ``(batch, seq_len - 1)``; masked positions have log-prob 0.
    """

    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    safe_targets = targets.masked_fill(~mask, 0)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        safe_targets.reshape(-1),
        reduction="none",
    ).view_as(safe_targets)
    return -nll * mask, mask


def sequence_logprobs(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed response log-prob and response-token count for each row."""

    device = model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logps, mask = token_logprobs(logits, labels)
    return logps.sum(-1), mask.sum(-1)


def sft_loss_from_batch(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean NLL over all response tokens in the batch."""

    summed, counts = sequence_logprobs(model, batch)
    return -summed.sum() / counts.sum().clamp_min(1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    summed, _ = sequence_logprobs(model, batch)
    return summed


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return sft_loss_from_batch(model, batch)


def dpo_logits(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
) -> torch.Tensor:
    """(policy gap) - (reference gap), before scaling by beta."""

    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    return policy_gap - reference_gap.to(policy_gap.device)


def dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
) -> torch.Tensor:
    """Batch mean of -log sigmoid(beta * (Delta_theta - Delta_ref))."""

    logits = dpo_logits(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
    )
    return -F.logsigmoid(beta * logits).mean()


def dpo_rewards(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implicit rewards r_w and r_l (eqs. 7-8), detached."""

    r_w = beta * (policy_chosen_logps.detach() - reference_chosen_logps.to(policy_chosen_logps.device))
    r_l = beta * (policy_rejected_logps.detach() - reference_rejected_logps.to(policy_rejected_logps.device))
    return r_w, r_l


def dpo_forward(
    policy_model,
    reference_model,
    batch: dict[str, dict[str, torch.Tensor]],
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DPO loss plus per-row diagnostics for one chosen/rejected batch.

    The reference model runs under ``torch.no_grad`` and may live on another device.
    """

    policy_chosen, _ = sequence_logprobs(policy_model, batch["chosen"])
    policy_rejected, _ = sequence_logprobs(policy_model, batch["rejected"])
    with torch.no_grad():
        reference_chosen, _ = sequence_logprobs(reference_model, batch["chosen"])
        reference_rejected, _ = sequence_logprobs(reference_model, batch["rejected"])
    reference_chosen = reference_chosen.float().to(policy_chosen.device)
    reference_rejected = reference_rejected.float().to(policy_chosen.device)

    loss = dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    r_w, r_l = dpo_rewards(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    stats = {
        "reward_chosen": r_w,
        "reward_rejected": r_l,
        "reward_margin": r_w - r_l,
        "reward_correct": (r_w > r_l).float(),
        "policy_chosen_logps": policy_chosen.detach(),
        "policy_rejected_logps": policy_rejected.detach(),
    }
    return loss, stats


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    loss, _ = dpo_forward(policy_model, reference_model, batch, beta)
    return loss
