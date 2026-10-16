"""Build the fixed SFT and DPO train/val subsets.

Outputs (JSONL, raw un-templated text; the ChatML template is applied at train time):
  <out_dir>/sft_{train,val}.jsonl        {"id", "prompt", "response"}
  <out_dir>/hh_{train,val}.jsonl         {"id", "prompt", "chosen", "rejected"}
  <out_dir>/uf_{train,val}.jsonl         {"id", "prompt", "chosen", "rejected"}
and small ID manifests + filter statistics in <manifest_dir>/ for auditing.

Usage:
  python scripts/prepare_data.py --out_dir data/processed --which sft hh uf
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pa2.utils import write_json, write_jsonl  # noqa: E402

SEED = 42


def take_valid(dataset, convert, n: int, seed: int, stats: dict) -> list[dict]:
    """Shuffle with ``seed`` and keep the first ``n`` rows that ``convert`` accepts."""

    out = []
    for idx, example in enumerate(dataset.shuffle(seed=seed)):
        row, reason = convert(example)
        if row is None:
            stats[reason] = stats.get(reason, 0) + 1
            continue
        out.append(row)
        if len(out) >= n:
            break
    stats["scanned"] = idx + 1
    stats["kept"] = len(out)
    if len(out) < n:
        print(f"WARNING: only {len(out)} valid rows (wanted {n})")
    return out


# ---------------------------------------------------------------- UltraChat SFT
def ultrachat_convert(split):
    def convert(ex):
        msgs = ex["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            return None, "bad_roles"
        prompt, response = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if not prompt or not response:
            return None, "empty"
        return {"id": f"ultrachat-{split}-{ex['prompt_id']}", "prompt": prompt, "response": response}, None

    return convert


# ---------------------------------------------------------------- Anthropic HH-RLHF
HUMAN, ASSISTANT = "\n\nHuman:", "\n\nAssistant:"


def split_hh(text: str):
    """Return (prompt, response) for a single-turn HH transcript, else None."""

    if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
        return None
    head, _, response = text.partition(ASSISTANT)
    _, _, prompt = head.partition(HUMAN)
    return prompt.strip(), response.strip()


def hh_convert(split):
    counter = {"i": 0}

    def convert(ex):
        counter["i"] += 1
        c, r = split_hh(ex["chosen"]), split_hh(ex["rejected"])
        if c is None or r is None:
            return None, "multi_turn"
        if c[0] != r[0]:
            return None, "prompt_mismatch"
        if not c[0] or not c[1] or not r[1]:
            return None, "empty"
        if c[1] == r[1]:
            return None, "identical_responses"
        row = {"id": f"hh-{split}-{counter['i']:06d}", "prompt": c[0], "chosen": c[1], "rejected": r[1]}
        return row, None

    return convert


# ---------------------------------------------------------------- UltraFeedback
def uf_convert(split):
    def convert(ex):
        chosen, rejected = ex["chosen"], ex["rejected"]
        if len(chosen) != 2 or len(rejected) != 2:
            return None, "multi_turn"
        if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
            return None, "bad_roles"
        if ex["score_chosen"] <= ex["score_rejected"]:
            return None, "tied_or_inverted_scores"
        prompt = ex["prompt"].strip()
        c, r = chosen[-1]["content"].strip(), rejected[-1]["content"].strip()
        if not prompt or not c or not r:
            return None, "empty"
        if c == r:
            return None, "identical_responses"
        return {"id": f"uf-{split}-{ex['prompt_id']}", "prompt": prompt, "chosen": c, "rejected": r}, None

    return convert


TRACKS = {
    "sft": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", "test_sft", ultrachat_convert, 30000, 1000),
    "hh": ("Anthropic/hh-rlhf", None, "train", "test", hh_convert, 15000, 1000),
    "uf": ("HuggingFaceH4/ultrafeedback_binarized", None, "train_prefs", "test_prefs", uf_convert, 15000, 1000),
}


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--manifest_dir", default="data/manifests")
    parser.add_argument("--which", nargs="+", default=list(TRACKS), choices=list(TRACKS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in args.which:
        repo, config, train_split, val_split, make_convert, n_train, n_val = TRACKS[name]
        paths = {s: os.path.join(args.out_dir, f"{name}_{s}.jsonl") for s in ("train", "val")}
        if all(os.path.exists(p) for p in paths.values()) and not args.overwrite:
            print(f"[{name}] already prepared; skipping")
            continue
        stats = {"source": repo, "seed": args.seed}
        for split_key, hf_split, n in (("train", train_split, n_train), ("val", val_split, n_val)):
            ds = load_dataset(repo, config, split=hf_split)
            split_stats = {"hf_split": hf_split, "available": len(ds), "requested": n}
            rows = take_valid(ds, make_convert(split_key), n, args.seed, split_stats)
            write_jsonl(paths[split_key], rows)
            write_json(os.path.join(args.manifest_dir, f"{name}_{split_key}_ids.json"), [r["id"] for r in rows])
            stats[split_key] = split_stats
            print(f"[{name}] {split_key}: {split_stats}")
        write_json(os.path.join(args.manifest_dir, f"{name}_stats.json"), stats)


if __name__ == "__main__":
    main()
