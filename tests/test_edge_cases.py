"""Extra checks beyond the public smoke tests (truncation, ordering, numerics)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


def manual_logprob(model, tokenizer, prompt, response):
    """Reference log p(response + eos | prompt) computed token by token."""

    ids = tokenizer.encode(prompt) + tokenizer.encode(response) + [tokenizer.eos_token_id]
    logits = model(torch.tensor([ids])).logits[0]
    logp = F.log_softmax(logits.float(), -1)
    n_prompt = len(tokenizer.encode(prompt))
    return sum(logp[t - 1, ids[t]].item() for t in range(n_prompt, len(ids)))


def test_eos_is_dropped_when_response_is_truncated():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab"], ["cdefgh"], max_length=5)
    assert batch["input_ids"].tolist() == [tok.encode("abcde")]
    assert batch["labels"].tolist() == [[-100, -100] + tok.encode("cde")]
    assert tok.eos_token_id not in batch["labels"][0].tolist()


def test_eos_kept_at_exact_boundary():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab"], ["cd"], max_length=5)
    assert batch["labels"][0, -1].item() == tok.eos_token_id
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]


def test_prompt_longer_than_max_length_gives_no_response_tokens():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tok, ["abcdefgh", "a"], ["xy", "xy"], max_length=4)
    assert batch["labels"][0].eq(-100).all()
    logps = adapter.compute_response_logprobs(model, tok, ["abcdefgh", "a"], ["xy", "xy"], max_length=4)
    assert logps[0].item() == 0.0
    loss = adapter.compute_sft_loss(model, tok, ["abcdefgh"], ["xy"], max_length=4)
    assert torch.isfinite(loss)


def test_mixed_lengths_preserve_row_order_and_right_padding():
    tok = ToyTokenizer()
    prompts = ["long prompt here: ", "q: ", "mid: "]
    responses = ["r", "a much longer response", "ok"]
    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=64)
    for row, (p, r) in enumerate(zip(prompts, responses)):
        ids = tok.encode(p) + tok.encode(r) + [tok.eos_token_id]
        n = len(ids)
        assert batch["input_ids"][row, :n].tolist() == ids
        assert batch["attention_mask"][row, :n].eq(1).all()
        assert batch["attention_mask"][row, n:].eq(0).all()
        assert batch["labels"][row, n:].eq(-100).all()


def test_tokenizer_without_pad_or_eos():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["a", "abc"], ["b", "d"], max_length=8)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["input_ids"][0, 3:].tolist() == [tok.eos_token_id, tok.eos_token_id]  # falls back to EOS as pad
    tok2 = NoEosTokenizer()
    batch2 = adapter.build_sft_batch(tok2, ["a"], ["bc"], max_length=8)
    assert batch2["labels"].tolist() == [[-100] + tok2.encode("bc")]


def test_response_logprobs_match_manual_computation():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.3)
    prompts = ["Q: hi ", "x"]
    responses = ["hello there", "y"]
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=64)
    for row, (p, r) in enumerate(zip(prompts, responses)):
        assert math.isclose(logps[row].item(), manual_logprob(model, tok, p, r), rel_tol=1e-5, abs_tol=1e-5)


def test_sft_loss_is_token_mean():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.9)
    prompts, responses = ["a: ", "bb: "], ["x", "longer one"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=64)
    total = sum(manual_logprob(model, tok, p, r) for p, r in zip(prompts, responses))
    n_tokens = sum(len(tok.encode(r)) + 1 for r in responses)
    assert math.isclose(loss.item(), -total / n_tokens, rel_tol=1e-5)


def test_dpo_loss_from_logps_matches_formula():
    pc = torch.tensor([-3.0, -1.0, -7.5])
    pr = torch.tensor([-4.0, -0.5, -2.0])
    rc = torch.tensor([-3.5, -1.2, -6.0])
    rr = torch.tensor([-3.0, -0.4, -2.5])
    beta = 0.1
    expected = torch.stack([
        -F.logsigmoid(torch.tensor(beta * ((a - b) - (c - d))))
        for a, b, c, d in zip(pc.tolist(), pr.tolist(), rc.tolist(), rr.tolist())
    ]).mean()
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    torch.testing.assert_close(loss, expected)


def test_dpo_loss_matches_logps_path_and_reference_has_no_graph():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.8)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=-0.4)
    prompts, chosen, rejected = ["p: ", "q: "], ["good", "yes"], ["bad", "nope"]
    loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.5, max_length=32)
    pc = adapter.compute_response_logprobs(policy, tok, prompts, chosen, 32)
    pr = adapter.compute_response_logprobs(policy, tok, prompts, rejected, 32)
    with torch.no_grad():
        rc = adapter.compute_response_logprobs(reference, tok, prompts, chosen, 32)
        rr = adapter.compute_response_logprobs(reference, tok, prompts, rejected, 32)
    torch.testing.assert_close(loss, adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, 0.5))
    loss.backward()
    assert reference.logit_bias.grad is None


def test_dpo_batch_shares_prompt_masking():
    tok = ToyTokenizer()
    batch = adapter.build_dpo_batch(tok, ["abc: "], ["x"], ["much longer"], max_length=32)
    n_prompt = len(tok.encode("abc: "))
    for side in ("chosen", "rejected"):
        assert batch[side]["labels"][0, :n_prompt].eq(-100).all()
        assert batch[side]["input_ids"][0, :n_prompt].tolist() == tok.encode("abc: ")


def test_small_optimization_steps_reduce_losses():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    first = adapter.compute_sft_loss(model, tok, ["q: "], ["abc"], 16)
    for _ in range(5):
        opt.zero_grad()
        adapter.compute_sft_loss(model, tok, ["q: "], ["abc"], 16).backward()
        opt.step()
    assert adapter.compute_sft_loss(model, tok, ["q: "], ["abc"], 16).item() < first.item()

    policy, reference = ToyCausalLM(tok.vocab_size), ToyCausalLM(tok.vocab_size)
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    start = adapter.compute_dpo_loss(policy, reference, tok, ["q: "], ["a"], ["b"], 0.1, 16)
    assert math.isclose(start.item(), math.log(2), rel_tol=1e-6)
    for _ in range(5):
        opt.zero_grad()
        adapter.compute_dpo_loss(policy, reference, tok, ["q: "], ["a"], ["b"], 0.1, 16).backward()
        opt.step()
    assert adapter.compute_dpo_loss(policy, reference, tok, ["q: "], ["a"], ["b"], 0.1, 16).item() < start.item()
