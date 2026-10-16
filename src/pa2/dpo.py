"""From-scratch DPO loop (no TRL / HF Trainer), shared by the HH-RLHF and UltraFeedback runs.

Policy: SFT best checkpoint in fp32 (trainable) on ``policy_device``.
Reference: frozen copy of the same checkpoint in fp16 on ``reference_device``,
evaluated under ``torch.no_grad``. Best checkpoint = highest validation reward
accuracy (ties broken by reward margin).
"""

from __future__ import annotations

import math
import os

import torch
from torch.utils.data import DataLoader

from .batching import build_dpo_batch
from .losses import dpo_forward
from .prompts import format_prompt, load_tokenizer
from .sft import autocast_ctx
from .utils import (
    MetricsLogger,
    build_optimizer,
    build_scheduler,
    describe_environment,
    read_jsonl,
    save_checkpoint,
    set_seed,
    write_json,
)

DEFAULTS = {
    "max_length": 512,
    "max_prompt_length": 256,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 1e-6,
    "scheduler": "constant",
    "warmup_steps": 0,
    "beta": 0.1,
    "optimizer": "rmsprop",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "epochs": 1,
    "fp16": True,
    "eval_every": 50,
    "eval_at_start": True,
    "eval_batch_size": 4,
    "log_every": 5,
    "seed": 42,
    "gradient_checkpointing": False,
    "policy_device": "cuda:0",
    "reference_device": "cuda:1",
    "reference_dtype": "float16",
    "max_train_examples": None,
    "max_val_examples": None,
    "max_steps": None,
    "num_workers": 2,
    "overwrite": False,
}

STAT_KEYS = ("reward_chosen", "reward_rejected", "reward_margin", "reward_correct")


class DPOCollator:
    def __init__(self, tokenizer, max_length: int, max_prompt_length: int | None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __call__(self, rows):
        prompts = [format_prompt(r["prompt"]) for r in rows]
        return build_dpo_batch(
            self.tokenizer, prompts, [r["chosen"] for r in rows], [r["rejected"] for r in rows],
            self.max_length, self.max_prompt_length,
        )


def pick_devices(cfg) -> tuple[torch.device, torch.device]:
    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cpu")
    policy = torch.device(cfg["policy_device"])
    reference = torch.device(cfg["reference_device"])
    if reference.index is not None and reference.index >= torch.cuda.device_count():
        print(f"[dpo] {reference} not available; placing reference on {policy}")
        reference = policy
    return policy, reference


@torch.no_grad()
def evaluate_dpo(policy, reference, loader, beta, device, fp16) -> dict:
    policy.eval()
    total_loss, n = 0.0, 0
    sums = {k: 0.0 for k in STAT_KEYS}
    for batch in loader:
        with autocast_ctx(device, fp16):
            loss, stats = dpo_forward(policy, reference, batch, beta)
        rows = stats["reward_margin"].numel()
        total_loss += loss.item() * rows
        n += rows
        for k in STAT_KEYS:
            sums[k] += stats[k].float().sum().item()
    policy.train()
    out = {"loss": total_loss / max(1, n), "pairs": n}
    out.update({k: v / max(1, n) for k, v in sums.items()})
    out["reward_accuracy"] = out.pop("reward_correct")
    return out


def train_dpo(cfg: dict) -> dict:
    from transformers import AutoModelForCausalLM

    cfg = {**DEFAULTS, **cfg}
    run_dir = cfg["output_dir"]
    final_dir = os.path.join(run_dir, "final")
    if os.path.exists(os.path.join(final_dir, "checkpoint_info.json")) and not cfg["overwrite"]:
        print(f"[dpo] {final_dir} already exists; skipping (set overwrite=true to retrain)")
        return {"skipped": True}

    os.makedirs(run_dir, exist_ok=True)
    set_seed(cfg["seed"])
    policy_device, reference_device = pick_devices(cfg)
    fp16 = bool(cfg["fp16"]) and policy_device.type == "cuda"
    write_json(os.path.join(run_dir, "config.json"), {"config": cfg, "environment": describe_environment()})

    sft_path = cfg["sft_checkpoint"]
    tokenizer = load_tokenizer(sft_path)
    policy = AutoModelForCausalLM.from_pretrained(sft_path, torch_dtype=torch.float32).to(policy_device)
    policy.config.use_cache = False
    if cfg["gradient_checkpointing"]:
        policy.gradient_checkpointing_enable()
    ref_dtype = torch.float16 if cfg["reference_dtype"] == "float16" and reference_device.type == "cuda" else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(sft_path, torch_dtype=ref_dtype).to(reference_device)
    reference.eval()
    reference.requires_grad_(False)

    train_rows = read_jsonl(cfg["train_path"])[: cfg["max_train_examples"]]
    val_rows = read_jsonl(cfg["val_path"])[: cfg["max_val_examples"]]
    collator = DPOCollator(tokenizer, cfg["max_length"], cfg["max_prompt_length"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_rows, batch_size=cfg["per_device_batch_size"], shuffle=True, generator=generator,
        collate_fn=collator, num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_rows, batch_size=cfg["eval_batch_size"], shuffle=False,
        collate_fn=collator, num_workers=cfg["num_workers"],
    )

    accum = cfg["gradient_accumulation_steps"]
    micro_total = len(train_loader) * cfg["epochs"]
    total_steps = math.ceil(micro_total / accum)
    if cfg["max_steps"]:
        total_steps = min(total_steps, cfg["max_steps"])

    optimizer = build_optimizer(policy, cfg["optimizer"], cfg["learning_rate"], cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], total_steps, cfg["warmup_steps"])
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    logger = MetricsLogger(run_dir)
    beta = cfg["beta"]
    print(f"[dpo] train={len(train_rows)} val={len(val_rows)} optimizer_steps={total_steps} "
          f"policy={policy_device} reference={reference_device}")

    best = (-1.0, -float("inf"))
    step, micro, skipped = 0, 0, 0
    window = {"loss": 0.0, "n": 0, **{k: 0.0 for k in STAT_KEYS}}

    def run_eval():
        nonlocal best
        m = evaluate_dpo(policy, reference, val_loader, beta, policy_device, fp16)
        key = (m["reward_accuracy"], m["reward_margin"])
        is_best = step > 0 and key > best
        logger.log("val", step, **m, is_best=is_best)
        if is_best:
            best = key
            save_checkpoint(policy, tokenizer, os.path.join(run_dir, "best"),
                            {"step": step, "reference_checkpoint": sft_path, "beta": beta, **m})

    if cfg["eval_at_start"]:
        run_eval()

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    done = False
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            micro += 1
            with autocast_ctx(policy_device, fp16):
                loss, stats = dpo_forward(policy, reference, batch, beta)
            scaler.scale(loss / accum).backward()
            rows = stats["reward_margin"].numel()
            window["loss"] += loss.item() * rows
            window["n"] += rows
            for k in STAT_KEYS:
                window[k] += stats[k].float().sum().item()

            if micro % accum != 0 and micro != micro_total:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % cfg["log_every"] == 0 or step == 1:
                n = max(1, window["n"])
                logger.log(
                    "train", step, loss=window["loss"] / n, lr=scheduler.get_last_lr()[0],
                    grad_norm=float(grad_norm), epoch=epoch, skipped_steps=skipped,
                    reward_chosen=window["reward_chosen"] / n, reward_rejected=window["reward_rejected"] / n,
                    reward_margin=window["reward_margin"] / n, reward_accuracy=window["reward_correct"] / n,
                )
                window = {"loss": 0.0, "n": 0, **{k: 0.0 for k in STAT_KEYS}}
            if step % cfg["eval_every"] == 0:
                run_eval()
            if step >= total_steps:
                done = True
                break
        if done:
            break

    if step % cfg["eval_every"] != 0:
        run_eval()
    save_checkpoint(policy, tokenizer, final_dir, {"step": step, "reference_checkpoint": sft_path, "beta": beta})
    summary = {
        "optimizer_steps": step,
        "best_val_reward_accuracy": best[0],
        "best_val_reward_margin": best[1],
        "skipped_fp16_steps": skipped,
    }
    write_json(os.path.join(run_dir, "summary.json"), summary)
    return summary
