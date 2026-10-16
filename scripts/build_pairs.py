"""Build a blinded pairwise-comparison JSONL (candidate vs baseline).

Response order is randomized per pair with a fixed seed. The mapping from
position (A/B) to model identity is kept only under ``hidden`` and is never put
into the judge prompt.

  python scripts/build_pairs.py \
      --candidate outputs/generations/alpaca_dpo_uf.json --candidate_name dpo_uf \
      --baseline outputs/generations/alpaca_sft.json --baseline_name sft \
      --output outputs/eval/dpo_uf_vs_sft_pairs.jsonl
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pa2.utils import read_json, write_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, help="generation JSON of the model under test (e.g. DPO)")
    p.add_argument("--candidate_name", required=True)
    p.add_argument("--baseline", required=True, help="generation JSON of the baseline (e.g. SFT)")
    p.add_argument("--baseline_name", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cand = {r["id"]: r for r in read_json(args.candidate)}
    base = {r["id"]: r for r in read_json(args.baseline)}
    if set(cand) != set(base):
        missing = sorted(set(cand) ^ set(base))[:10]
        raise SystemExit(f"generation files cover different prompts, e.g. {missing}")

    rng = random.Random(args.seed)
    rows = []
    for pid in base:  # baseline file order == eval-set order
        c, b = cand[pid], base[pid]
        if c["instruction"] != b["instruction"]:
            raise SystemExit(f"instruction mismatch for {pid}")
        candidate_is_a = rng.random() < 0.5
        a, bb = (c, b) if candidate_is_a else (b, c)
        rows.append({
            "pair_id": pid,
            "instruction": b["instruction"],
            "category": b.get("category"),
            "dataset": b.get("dataset"),
            "response_a": a["output"],
            "response_b": bb["output"],
            "hidden": {
                "A": args.candidate_name if candidate_is_a else args.baseline_name,
                "B": args.baseline_name if candidate_is_a else args.candidate_name,
                "candidate": args.candidate_name,
                "baseline": args.baseline_name,
            },
        })
    write_jsonl(args.output, rows)
    n_a = sum(r["hidden"]["A"] == args.candidate_name for r in rows)
    print(f"wrote {len(rows)} pairs to {args.output} (candidate in position A for {n_a})")


if __name__ == "__main__":
    main()
