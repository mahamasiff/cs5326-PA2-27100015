"""From-scratch supervised fine-tuning loop (no HF Trainer).

fp32 master weights + fp16 autocast + GradScaler, gradient accumulation,
gradient clipping, warmup + cosine schedule, periodic validation loss, and only
``best`` (lowest validation loss) and ``final`` checkpoints.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from .batching import build_sft_batch
from .losses import sequence_logprobs, sft_loss_from_batch
from .prompts import format_prompt, load_tokenizer
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
    "model_name": "Qwen/Qwen2.5-0.5B",
    "max_length": 512,
    "per_device_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.03,
    "scheduler": "cosine",
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "optimizer": "adamw",
    "epochs": 1,
    "fp16": True,
    "eval_every": 100,
    "eval_batch_size": 8,
    "log_every": 10,
    "seed": 42,
    "gradient_checkpointing": False,
    "max_train_examples": None,
    "max_val_examples": None,
    "max_steps": None,
    "device": "cuda:0",
    "num_workers": 2,
    "overwrite": False,
}


class SFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        prompts = [format_prompt(r["prompt"]) for r in rows]
        responses = [r["response"] for r in rows]
        return build_sft_batch(self.tokenizer, prompts, responses, self.max_length)


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate_sft(model, loader, device, fp16: bool) -> dict:
    """Token-mean validation NLL over the whole validation set."""

    model.eval()
    total_logp, total_tokens = 0.0, 0
    for batch in loader:
        with autocast_ctx(device, fp16):
            summed, counts = sequence_logprobs(model, batch)
        total_logp += summed.float().sum().item()
        total_tokens += int(counts.sum().item())
    model.train()
    loss = -total_logp / max(1, total_tokens)
    return {"loss": loss, "ppl": math.exp(min(loss, 50)), "tokens": total_tokens}


def train_sft(cfg: dict) -> dict:
    from transformers import AutoModelForCausalLM

    cfg = {**DEFAULTS, **cfg}
    run_dir = cfg["output_dir"]
    final_dir = os.path.join(run_dir, "final")
    if os.path.exists(os.path.join(final_dir, "checkpoint_info.json")) and not cfg["overwrite"]:
        print(f"[sft] {final_dir} already exists; skipping (set overwrite=true to retrain)")
        return {"skipped": True}

    os.makedirs(run_dir, exist_ok=True)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    fp16 = bool(cfg["fp16"]) and device.type == "cuda"
    write_json(os.path.join(run_dir, "config.json"), {"config": cfg, "environment": describe_environment()})

    tokenizer = load_tokenizer(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    train_rows = read_jsonl(cfg["train_path"])[: cfg["max_train_examples"]]
    val_rows = read_jsonl(cfg["val_path"])[: cfg["max_val_examples"]]
    collator = SFTCollator(tokenizer, cfg["max_length"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_rows, batch_size=cfg["per_device_batch_size"], shuffle=True, generator=generator,
        collate_fn=collator, num_workers=cfg["num_workers"], drop_last=False,
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
    warmup_steps = int(round(cfg["warmup_ratio"] * total_steps))

    optimizer = build_optimizer(model, cfg["optimizer"], cfg["learning_rate"], cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], total_steps, warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    logger = MetricsLogger(run_dir)
    print(f"[sft] train={len(train_rows)} val={len(val_rows)} optimizer_steps={total_steps} warmup={warmup_steps}")

    best_val = float("inf")
    step, micro = 0, 0
    window_loss, window_n, skipped = 0.0, 0, 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    def run_eval():
        nonlocal best_val
        metrics = evaluate_sft(model, val_loader, device, fp16)
        is_best = metrics["loss"] < best_val
        logger.log("val", step, loss=metrics["loss"], ppl=metrics["ppl"], is_best=is_best)
        if is_best:
            best_val = metrics["loss"]
            save_checkpoint(model, tokenizer, os.path.join(run_dir, "best"),
                            {"step": step, "val_loss": best_val, "base_model": cfg["model_name"]})

    done = False
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            micro += 1
            with autocast_ctx(device, fp16):
                loss = sft_loss_from_batch(model, batch)
            scaler.scale(loss / accum).backward()
            window_loss += loss.item()
            window_n += 1

            if micro % accum != 0 and micro != micro_total:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % cfg["log_every"] == 0 or step == 1:
                logger.log("train", step, loss=window_loss / max(1, window_n), lr=scheduler.get_last_lr()[0],
                           grad_norm=float(grad_norm), epoch=epoch, skipped_steps=skipped)
                window_loss, window_n = 0.0, 0
            if step % cfg["eval_every"] == 0:
                run_eval()
            if step >= total_steps:
                done = True
                break
        if done:
            break

    if step % cfg["eval_every"] != 0:
        run_eval()
    save_checkpoint(model, tokenizer, final_dir, {"step": step, "base_model": cfg["model_name"]})
    summary = {"optimizer_steps": step, "best_val_loss": best_val, "skipped_fp16_steps": skipped}
    write_json(os.path.join(run_dir, "summary.json"), summary)
    return summary
