"""Judge saved pairwise JSONL files with a Kaggle model-proxy judge.

Run inside a Kaggle notebook configured as a utility benchmark (kaggle_benchmarks
available). Never regenerates model outputs; reads only the pairs file.

  python scripts/judge_pairwise_kaggle.py \
    --pairs_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl \
    --output_path outputs/eval/dpo_uf_vs_sft_judged.jsonl \
    --judge_model google/gemini-3.6-flash \
    --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

``--mode`` picks the rubric (general / safety / persona); by default it is
inferred from the pairs' dataset field. ``--backend dummy`` runs offline with a
fake judge for testing the pipeline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pa2.judging import CRITERIA, run_judging, save_summary, summarize  # noqa: E402
from pa2.utils import read_jsonl  # noqa: E402

DATASET_MODES = {"alpaca_eval": "general", "hh_rlhf_style_eval": "safety", "persona_style_eval": "persona"}


def kaggle_backend(judge_model: str):
    import kaggle_benchmarks as kbench

    llm = None
    for name in (judge_model, judge_model.split("/", 1)[-1]):
        try:
            llm = kbench.llms[name]
            break
        except KeyError:
            continue
    if llm is None:
        raise SystemExit(f"judge model {judge_model!r} not available. Options: {sorted(kbench.llms)}")
    counter = itertools.count()

    def call(prompt: str, batch: list[dict]) -> str:
        # A fresh chat per request so batches never share conversation history.
        with kbench.chats.new(f"pairwise_judge_{next(counter)}"):
            return str(llm.prompt(prompt, temperature=0))

    return call


def dummy_backend(mode: str):
    """Offline stand-in: prefers the longer response (for pipeline tests only)."""

    def call(prompt: str, batch: list[dict]) -> str:
        out = []
        for i, p in enumerate(batch, start=1):
            la, lb = len(p["response_a"]), len(p["response_b"])
            pick = "tie" if la == lb else ("A" if la > lb else "B")
            out.append({"item": i, **{k: pick for k in CRITERIA[mode]}, "reason": "longer response (dummy judge)"})
        return json.dumps(out)

    return call


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--judge_model", default="google/gemini-3.6-flash")
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--sleep_seconds", type=float, default=5.0)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--reuse_judgments", action="store_true")
    p.add_argument("--mode", choices=sorted(CRITERIA), default=None)
    p.add_argument("--failures_path", default=None)
    p.add_argument("--summary_path", default=None)
    p.add_argument("--backend", choices=["kaggle", "dummy"], default="kaggle")
    p.add_argument("--limit", type=int, default=None, help="judge only the first N pairs (smoke test)")
    args = p.parse_args()

    if args.sleep_seconds < 5:
        raise SystemExit("--sleep_seconds must be at least 5 (manual section 13.2)")
    stem = args.output_path[: -len(".jsonl")] if args.output_path.endswith(".jsonl") else args.output_path
    failures_path = args.failures_path or f"{stem}_failures.jsonl"
    summary_path = args.summary_path or f"{stem}_summary.json"
    raw_path = f"{stem}_raw.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    pairs = read_jsonl(args.pairs_path)[: args.limit]
    mode = args.mode or DATASET_MODES.get(pairs[0].get("dataset"), "general")
    call = kaggle_backend(args.judge_model) if args.backend == "kaggle" else dummy_backend(mode)
    print(f"[judge] mode={mode} judge={args.judge_model} backend={args.backend}")

    stats = run_judging(
        pairs, call, args.output_path, failures_path, raw_path, args.judge_model, mode,
        batch_size=args.batch_size, sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries, reuse_judgments=args.reuse_judgments,
    )
    summary = summarize(pairs, args.output_path, failures_path, mode, stats)
    save_summary(summary, summary_path)
    print(json.dumps({k: v for k, v in summary.items() if k in CRITERIA[mode] or k in ("candidate", "baseline", "n_pairs")}, indent=2))


if __name__ == "__main__":
    main()
