"""Collect judge summaries into one markdown results table (manual section 14.5).

  python scripts/collect_results.py --eval_dir outputs/eval --output outputs/eval/results.md

Each evaluation set gets its own rows; unlike sets are never merged into one score.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pa2.judging import CRITERIA  # noqa: E402
from pa2.utils import read_json  # noqa: E402


def fmt(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="outputs/eval")
    p.add_argument("--output", default="outputs/eval/results.md")
    args = p.parse_args()

    lines = [
        "| Comparison | Criterion | Judge | N | Wins | Ties | Losses | Unknown | Failures | Win rate (ties=0.5) | Win rate (no ties) | Retries |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    cat_lines = []
    for path in sorted(glob.glob(os.path.join(args.eval_dir, "*_summary.json"))):
        s = read_json(path)
        label = f"{s.get('candidate')} vs {s.get('baseline')} ({os.path.basename(path).replace('_judged_summary.json', '')})"
        for key in CRITERIA[s["mode"]]:
            r = s[key]
            lines.append(
                f"| {label} | {key} | {s.get('judge_model')} | {s['n_pairs']} | {r['wins']} | {r['ties']} | {r['losses']} | "
                f"{r['unknown']} | {r['failures']} | {fmt(r['win_rate_with_ties'])} | {fmt(r['win_rate_without_ties'])} | "
                f"{s.get('run_stats', {}).get('retries', 0)} |"
            )
            for cat, c in r["by_category"].items():
                cat_lines.append(f"| {label} | {key} | {cat} | {c['wins']} | {c['ties']} | {c['losses']} | {fmt(c['win_rate_with_ties'])} |")

    text = "## DPO vs SFT (candidate perspective)\n\n" + "\n".join(lines)
    text += "\n\n## By category\n\n| Comparison | Criterion | Category | Wins | Ties | Losses | Win rate (ties=0.5) |\n|---|---|---|---|---|---|---|\n"
    text += "\n".join(cat_lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
