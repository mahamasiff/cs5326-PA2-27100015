"""Training plots for the report (manual section 14.4).

  python scripts/plot_metrics.py --run sft=outputs/sft --run dpo_hh=outputs/dpo_hh \
      --run dpo_uf=outputs/dpo_uf --run lora_dpo=outputs/lora_dpo --out_dir outputs/plots

Run type is inferred from metrics.jsonl. Train curves are drawn raw (faint) and
with an exponential moving average (solid; alpha stated in the legend); the
selected ``best`` checkpoint is marked with a dashed vertical line.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pa2.utils import read_json, read_jsonl  # noqa: E402

EMA = 0.9


def ema(values, alpha=EMA):
    out, avg = [], None
    for v in values:
        avg = v if avg is None else alpha * avg + (1 - alpha) * v
        out.append(avg)
    return out


def series(rows, split, key):
    pts = [(r["step"], r[key]) for r in rows if r.get("split") == split and r.get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def best_step(run_dir):
    path = os.path.join(run_dir, "best", "checkpoint_info.json")
    return read_json(path).get("step") if os.path.exists(path) else None


def mark_best(ax, step):
    if step is not None:
        ax.axvline(step, color="gray", linestyle="--", linewidth=1, label=f"best ckpt (step {step})")


def plot_train_val(ax, rows, key, title, ylabel, best):
    xs, ys = series(rows, "train", key)
    if xs:
        ax.plot(xs, ys, color="tab:blue", alpha=0.25, linewidth=1)
        ax.plot(xs, ema(ys), color="tab:blue", label=f"train (EMA {EMA})")
    xs, ys = series(rows, "val", key)
    if xs:
        ax.plot(xs, ys, "o-", color="tab:orange", label="validation")
    mark_best(ax, best)
    ax.set_title(title)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_sft(name, run_dir, rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_train_val(ax, rows, "loss", f"{name}: response-token NLL", "loss (mean NLL per token)", best_step(run_dir))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_loss.png"), dpi=150)
    plt.close(fig)


def plot_dpo(name, run_dir, rows, out_dir):
    best = best_step(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    plot_train_val(axes[0, 0], rows, "loss", "DPO loss", "-log sigmoid(beta * logit gap)", best)
    plot_train_val(axes[0, 1], rows, "reward_accuracy", "Reward accuracy (r_w > r_l)", "fraction of pairs", best)
    plot_train_val(axes[1, 0], rows, "reward_margin", "Reward margin (r_w - r_l)", "mean margin", best)
    ax = axes[1, 1]
    for key, color in (("reward_chosen", "tab:green"), ("reward_rejected", "tab:red")):
        xs, ys = series(rows, "val", key)
        ax.plot(xs, ys, "o-", color=color, label=f"val {key.split('_')[1]}")
    mark_best(ax, best)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Validation implicit rewards")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("beta * (log pi - log pi_ref)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"{name} (validation every eval step, train smoothed)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_metrics.png"), dpi=150)
    plt.close(fig)


def plot_lora(name, rows, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    specs = [("loss", "eval_loss", "DPO loss"), ("rewards/accuracies", "eval_rewards/accuracies", "Reward accuracy"),
             ("rewards/margins", "eval_rewards/margins", "Reward margin")]
    for ax, (train_key, eval_key, title) in zip(axes, specs):
        t = [(r["step"], r[train_key]) for r in rows if train_key in r and "step" in r]
        e = [(r["step"], r[eval_key]) for r in rows if eval_key in r and "step" in r]
        if t:
            ax.plot([p[0] for p in t], [p[1] for p in t], color="tab:blue", label="train")
        if e:
            ax.plot([p[0] for p in e], [p[1] for p in e], "o-", color="tab:orange", label="validation")
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{name} (TRL DPOTrainer + LoRA)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_metrics.png"), dpi=150)
    plt.close(fig)


def plot_dpo_overlay(dpo_runs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in dpo_runs.items():
        for ax, key in zip(axes, ("reward_accuracy", "reward_margin")):
            xs, ys = series(rows, "val", key)
            ax.plot(xs, ys, "o-", label=name)
    for ax, title in zip(axes, ("Validation reward accuracy", "Validation reward margin")):
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dpo_runs_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--out_dir", default="outputs/plots")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dpo_runs = {}
    for spec in args.run:
        name, _, run_dir = spec.partition("=")
        rows = read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
        if not rows:
            print(f"skip {name}: no metrics.jsonl in {run_dir}")
            continue
        if "split" not in rows[0]:
            plot_lora(name, rows, args.out_dir)
        elif any("reward_accuracy" in r for r in rows):
            plot_dpo(name, run_dir, rows, args.out_dir)
            dpo_runs[name] = rows
        else:
            plot_sft(name, run_dir, rows, args.out_dir)
        print(f"plotted {name}")
    if len(dpo_runs) > 1:
        plot_dpo_overlay(dpo_runs, args.out_dir)


if __name__ == "__main__":
    main()
