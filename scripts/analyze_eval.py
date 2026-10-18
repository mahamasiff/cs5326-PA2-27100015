"""Secondary analysis of generations and judgments for the report (section 14.6).

Computes, for each comparison: repetition-loop and length statistics, win rate split by
whether either response loops, win rate split by relative length, refusal-phrase rates
on the HH set, judge position counts, and verdict consistency on repeated HH prompts.

  python scripts/analyze_eval.py --gen_dir outputs/generations --eval_dir outputs/eval \
      --output outputs/eval/analysis.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pa2.utils import read_json, read_jsonl, write_json  # noqa: E402

REFUSAL = re.compile(
    r"\b(I'?m sorry|I am sorry|I cannot|I can'?t|I can not|I won'?t|I'?m not able|I am not able|"
    r"unable to (help|assist|provide)|not appropriate|I do not|I don'?t (think|feel comfortable)|"
    r"cannot (help|assist|provide))",
    re.I,
)

COMPARISONS = {
    "dpo_uf_vs_sft": ("alpaca_dpo_uf", "alpaca_sft", "winner"),
    "dpo_hh_vs_sft": ("hh_dpo_hh", "hh_sft", "winner"),
    "lora_vs_base": ("persona_lora", "persona_base", "quality"),
}


def loops(text: str, n: int = 8, min_repeats: int = 4) -> bool:
    """True if some n-word sequence occurs at least ``min_repeats`` times."""

    words = text.split()
    if len(words) < 40:
        return False
    grams = collections.Counter(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return grams.most_common(1)[0][1] >= min_repeats


def rate(c: collections.Counter) -> dict:
    n = c["win"] + c["tie"] + c["loss"]
    return {"n": n, **{k: c[k] for k in ("win", "tie", "loss")},
            "win_rate_with_ties": round((c["win"] + 0.5 * c["tie"]) / n, 3) if n else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", default="outputs/generations")
    p.add_argument("--eval_dir", default="outputs/eval")
    p.add_argument("--output", default="outputs/eval/analysis.json")
    args = p.parse_args()

    gens = {}
    for name in ("alpaca_base", "alpaca_sft", "alpaca_dpo_uf", "hh_sft", "hh_dpo_hh", "persona_base", "persona_lora"):
        rows = read_json(os.path.join(args.gen_dir, f"{name}.json"))
        gens[name] = {r["id"]: r for r in rows}

    out = {"generation_stats": {}, "comparisons": {}}
    for name, rows in gens.items():
        outs = [r["output"] for r in rows.values()]
        out["generation_stats"][name] = {
            "n": len(outs),
            "looping": sum(loops(o) for o in outs),
            "hit_max_new_tokens": sum(r["hit_max_new_tokens"] for r in rows.values()),
            "mean_chars": round(sum(map(len, outs)) / len(outs), 1),
        }

    for comp, (cand, base, crit) in COMPARISONS.items():
        judged = {r["pair_id"]: r for r in read_jsonl(os.path.join(args.eval_dir, f"{comp}_judged.jsonl"))}
        by_loop, by_len = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
        positions = collections.Counter()
        for pid, r in judged.items():
            v = r["verdicts"][crit]
            positions[v["position"]] += 1
            outcome = {r["hidden"]["candidate"]: "win", r["hidden"]["baseline"]: "loss"}.get(v["model"], "tie")
            c_out, b_out = gens[cand][pid]["output"], gens[base][pid]["output"]
            key = f"candidate_{'loops' if loops(c_out) else 'ok'}/baseline_{'loops' if loops(b_out) else 'ok'}"
            by_loop[key][outcome] += 1
            by_len["candidate_longer" if len(c_out) > len(b_out) else "candidate_not_longer"][outcome] += 1
        out["comparisons"][comp] = {
            "criterion": crit,
            "judge_position_counts": dict(positions),
            "by_loop_status": {k: rate(v) for k, v in sorted(by_loop.items())},
            "by_relative_length": {k: rate(v) for k, v in sorted(by_len.items())},
        }

    # Refusal phrases in the first 300 characters, per HH category.
    refusals = collections.defaultdict(lambda: {"sft": 0, "dpo_hh": 0, "n": 0})
    for pid, r in gens["hh_sft"].items():
        cat = r.get("category", "all")
        refusals[cat]["n"] += 1
        refusals[cat]["sft"] += bool(REFUSAL.search(r["output"][:300]))
        refusals[cat]["dpo_hh"] += bool(REFUSAL.search(gens["hh_dpo_hh"][pid]["output"][:300]))
    out["hh_refusal_phrase_counts"] = dict(sorted(refusals.items()))

    # Judge consistency on prompts that appear more than once in the HH set. Batched fp16 generation can
    # give slightly different outputs for repeats, so only groups where both models' answers are identical
    # across all repeats are used: the judge then saw the same pair (possibly in a different A/B order).
    judged = {r["pair_id"]: r for r in read_jsonl(os.path.join(args.eval_dir, "dpo_hh_vs_sft_judged.jsonl"))}
    groups = collections.defaultdict(list)
    for pid, r in gens["hh_sft"].items():
        groups[r["instruction"]].append(pid)
    repeated = [ids for ids in groups.values() if len(ids) > 1]
    identical = [ids for ids in repeated
                 if len({gens["hh_sft"][i]["output"] for i in ids}) == 1
                 and len({gens["hh_dpo_hh"][i]["output"] for i in ids}) == 1]
    out["hh_judge_consistency"] = {
        "unique_prompts": len(groups),
        "repeated_prompts": len(repeated),
        "repeated_with_identical_outputs": len(identical),
        "identical_with_inconsistent_verdicts": sum(
            len({judged[i]["verdicts"]["winner"]["model"] for i in ids}) > 1 for ids in identical),
    }

    write_json(args.output, out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
