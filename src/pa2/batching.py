"""Tokenization, response-only label masking, and padding for SFT and DPO.

Every example is built as ``prompt_ids + response_ids + [eos]`` where prompt and
response are tokenized separately (no special tokens added), so the prompt /
response boundary is exact. EOS is appended before truncation to
``max_length``, so a long response can lose its EOS. Prompt and padding
positions get label ``-100``; response positions get their own token id.
Sequences are right-padded to the longest row in the batch and rows keep their
input order.
"""

from __future__ import annotations

from typing import Sequence

import torch

IGNORE_INDEX = -100


def encode_text(tokenizer, text: str) -> list[int]:
    """Token ids for ``text`` without BOS/EOS or other added special tokens."""

    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


def pad_token_id(tokenizer) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    return 0


def tokenize_example(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    max_prompt_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, labels)`` for one prompt/response pair, unpadded.

    ``max_prompt_length`` (used by DPO) keeps the *last* prompt tokens, which
    preserves the assistant header at the end of the formatted prompt.
    """

    prompt_ids = encode_text(tokenizer, prompt)
    response_ids = encode_text(tokenizer, response)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        response_ids = response_ids + [int(eos_id)]

    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[len(prompt_ids) - max_prompt_length :] if max_prompt_length > 0 else []

    input_ids = (prompt_ids + response_ids)[:max_length]
    n_prompt = min(len(prompt_ids), len(input_ids))
    labels = [IGNORE_INDEX] * n_prompt + input_ids[n_prompt:]
    return input_ids, labels


def collate(rows: Sequence[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad ``(input_ids, labels)`` rows into LM batch tensors."""

    width = max((len(ids) for ids, _ in rows), default=0)
    width = max(width, 1)
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in rows:
        n_pad = width - len(ids)
        input_ids.append(ids + [pad_id] * n_pad)
        attention_mask.append([1] * len(ids) + [0] * n_pad)
        labels.append(labs + [IGNORE_INDEX] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    max_prompt_length: int | None = None,
) -> dict[str, torch.Tensor]:
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    rows = [
        tokenize_example(tokenizer, p, r, max_length, max_prompt_length)
        for p, r in zip(prompts, responses)
    ]
    return collate(rows, pad_token_id(tokenizer))


def build_sft_batch(tokenizer, prompts, responses, max_length):
    return build_lm_batch(tokenizer, list(prompts), list(responses), max_length)


def build_dpo_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length,
    max_prompt_length: int | None = None,
):
    """Separate chosen and rejected LM batches built from identical prompts."""

    prompts = list(prompts)
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses and rejected_responses must align")
    return {
        "chosen": build_lm_batch(tokenizer, prompts, list(chosen_responses), max_length, max_prompt_length),
        "rejected": build_lm_batch(tokenizer, prompts, list(rejected_responses), max_length, max_prompt_length),
    }
