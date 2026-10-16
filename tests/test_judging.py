"""Offline tests for the pairwise judging pipeline (no Kaggle access needed)."""

from __future__ import annotations

import json

import pytest

from pa2.judging import build_judge_prompt, parse_judge_response, run_judging, summarize
from pa2.utils import read_jsonl


def make_pairs(n=7):
    pairs = []
    for i in range(n):
        cand_a = i % 2 == 0
        pairs.append({
            "pair_id": f"p{i}", "instruction": f"question {i}", "category": "c1" if i < 4 else "c2",
            "dataset": "alpaca_eval",
            "response_a": "candidate answer" if cand_a else "base",
            "response_b": "base" if cand_a else "candidate answer",
            "hidden": {"A": "dpo_secret" if cand_a else "sft_secret", "B": "sft_secret" if cand_a else "dpo_secret",
                       "candidate": "dpo_secret", "baseline": "sft_secret"},
        })
    return pairs


def prefer_candidate(prompt, batch):
    return json.dumps([
        {"item": i, "winner": "A" if p["hidden"]["A"] == "dpo_secret" else "B", "reason": "ok"}
        for i, p in enumerate(batch, start=1)
    ])


def test_prompt_is_blinded():
    prompt = build_judge_prompt(make_pairs(3), "safety")
    assert "secret" not in prompt and "dpo" not in prompt.lower() and "sft" not in prompt.lower()
    assert "[Response A]" in prompt and "[Response B]" in prompt
    assert "refusal" in prompt


def test_parser_handles_fences_case_and_missing_items():
    text = 'Sure!\n```json\n[{"item": 1, "winner": "a"}, {"item": 3, "winner": "Tie"}]\n```'
    v = parse_judge_response(text, 3, "general")
    assert [x["winner"] for x in v] == ["A", "unknown", "tie"]
    with pytest.raises(ValueError):
        parse_judge_response("I cannot decide.", 2, "general")


def test_end_to_end_counts_and_translation(tmp_path):
    pairs = make_pairs(7)
    out, fail = tmp_path / "j.jsonl", tmp_path / "f.jsonl"
    sleeps = []
    run_judging(pairs, prefer_candidate, str(out), str(fail), None, "judge", "general",
                batch_size=5, sleep_seconds=5, sleep_fn=sleeps.append)
    assert sleeps == [5]  # two batches -> one sleep between requests
    s = summarize(pairs, str(out), str(fail), "general")
    assert (s["winner"]["wins"], s["winner"]["losses"], s["winner"]["ties"]) == (7, 0, 0)
    assert s["winner"]["win_rate_without_ties"] == 1.0
    assert s["winner"]["judge_position_counts"] == {"A": 4, "B": 3}


def test_retries_then_logs_failures_and_resume_skips_done(tmp_path):
    pairs = make_pairs(7)
    out, fail = tmp_path / "j.jsonl", tmp_path / "f.jsonl"
    calls = {"n": 0}

    def flaky(prompt, batch):
        calls["n"] += 1
        if "question 5" in prompt:  # second batch always fails
            raise RuntimeError("request denied")
        return prefer_candidate(prompt, batch)

    stats = run_judging(pairs, flaky, str(out), str(fail), None, "judge", "general",
                        batch_size=5, max_retries=3, sleep_fn=lambda s: None)
    assert calls["n"] == 1 + 3
    assert stats["failed_pairs"] == 2
    assert {r["pair_id"] for r in read_jsonl(str(fail))} == {"p5", "p6"}
    s = summarize(pairs, str(out), str(fail), "general")
    assert s["winner"]["failures"] == 2 and s["winner"]["wins"] == 5

    calls["n"] = 0

    def counting(prompt, batch):
        calls["n"] += 1
        return prefer_candidate(prompt, batch)

    run_judging(pairs, counting, str(out), str(fail), None, "judge", "general",
                batch_size=5, reuse_judgments=True, sleep_fn=lambda s: None)
    assert calls["n"] == 1  # only the 2 previously failed pairs are re-judged
    assert summarize(pairs, str(out), str(fail), "general")["winner"]["wins"] == 7


def test_persona_mode_has_two_criteria(tmp_path):
    pairs = make_pairs(2)

    def judge(prompt, batch):
        return json.dumps([{"item": i, "style": "A", "quality": "tie"} for i in range(1, len(batch) + 1)])

    out, fail = tmp_path / "j.jsonl", tmp_path / "f.jsonl"
    run_judging(pairs, judge, str(out), str(fail), None, "judge", "persona", sleep_fn=lambda s: None)
    s = summarize(pairs, str(out), str(fail), "persona")
    assert s["style"]["wins"] == 1 and s["style"]["losses"] == 1
    assert s["quality"]["ties"] == 2
