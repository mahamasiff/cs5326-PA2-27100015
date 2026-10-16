"""Seeding, file I/O, config loading, metric logging, and LR schedules."""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Iterable

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    """Write atomically so an interrupted notebook never leaves a half-written file."""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply ``key=value`` overrides (values parsed as YAML)."""

    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        cfg[key.strip()] = yaml.safe_load(value)
    return cfg


class MetricsLogger:
    """Appends one JSON line per event to ``<run_dir>/metrics.jsonl``."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "metrics.jsonl")
        self.start = time.time()

    def log(self, split: str, step: int, **values) -> None:
        row = {"split": split, "step": step, "elapsed_s": round(time.time() - self.start, 1)}
        row.update({k: (float(v) if isinstance(v, (int, float, torch.Tensor)) else v) for k, v in values.items()})
        append_jsonl(self.path, row)
        print(json.dumps(row), flush=True)


def warmup_cosine_lambda(total_steps: int, warmup_steps: int):
    """Linear warmup to the peak LR, then cosine decay to 0."""

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def constant_lambda(total_steps: int, warmup_steps: int):
    """Constant LR, with optional linear warmup."""

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    return fn


def build_scheduler(optimizer, kind: str, total_steps: int, warmup_steps: int):
    fns = {"cosine": warmup_cosine_lambda, "constant": constant_lambda}
    if kind not in fns:
        raise ValueError(f"unknown scheduler {kind!r}; expected one of {sorted(fns)}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fns[kind](total_steps, warmup_steps))


def build_optimizer(model, name: str, lr: float, weight_decay: float):
    """AdamW or RMSprop; weight decay skips biases and norm weights."""

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n.lower() else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=lr)
    raise ValueError(f"unknown optimizer {name!r}")


def save_checkpoint(model, tokenizer, path: str, extra: dict | None = None) -> None:
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    if extra is not None:
        write_json(os.path.join(path, "checkpoint_info.json"), extra)


def describe_environment() -> dict:
    import transformers

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    try:
        import datasets

        info["datasets"] = datasets.__version__
    except ImportError:
        pass
    return info
