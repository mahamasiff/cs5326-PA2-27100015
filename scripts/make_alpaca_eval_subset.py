"""Create the fixed AlpacaEval evaluation subset (300 prompts, seed 42).

Downloads tatsu-lab/alpaca_eval (805 prompts), samples 300 indices with
``random.Random(42).sample`` and writes them in sampled order to
eval_sets/alpaca_eval_300_seed42.json with fields {id, instruction, dataset, source_dataset}.
"""

from __future__ import annotations

import argparse
import json
import os
import random


def main():
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval_sets/alpaca_eval_300_seed42.json")
    args = parser.parse_args()

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        full = json.load(f)
    indices = random.Random(args.seed).sample(range(len(full)), args.n)
    subset = [
        {
            "id": f"alpaca-eval-{idx:03d}",
            "instruction": full[idx]["instruction"],
            "dataset": "alpaca_eval",
            "source_dataset": full[idx].get("dataset"),
        }
        for idx in indices
    ]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(subset)} of {len(full)} prompts to {args.output}")


if __name__ == "__main__":
    main()
