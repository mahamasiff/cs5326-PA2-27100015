"""Task 5: persona DPO with TRL DPOTrainer + PEFT LoRA on SmolLM2-360M-Instruct.

Rows {prompt, chosen, rejected} are converted to TRL's conversational format:
  {"prompt": [{"role": "user", ...}], "chosen": [{"role": "assistant", ...}],
   "rejected": [{"role": "assistant", ...}]}
With ``peft_config`` and no ``ref_model``, TRL uses the frozen base (adapter
disabled) as the DPO reference. Saves the LoRA adapter to <output_dir>/final and
the trainer log history to <output_dir>/metrics.jsonl.

  python scripts/train_lora_dpo.py --config configs/lora_dpo.yaml [--set key=value ...]
"""

from __future__ import annotations

import os

# Single GPU: HF Trainer would otherwise wrap the PEFT model in DataParallel on T4 x2.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

from pa2.utils import describe_environment, load_config, read_json, set_seed, write_json, write_jsonl  # noqa: E402


def to_conversational(row: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def main():
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_dpo.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"])

    rows = read_json(cfg["data_path"])
    order = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(order)
    n_val = max(1, int(round(cfg["val_fraction"] * len(rows))))
    val_idx, train_idx = sorted(order[:n_val]), sorted(order[n_val:])
    train_ds = Dataset.from_list([to_conversational(rows[i]) for i in train_idx])
    val_ds = Dataset.from_list([to_conversational(rows[i]) for i in val_idx])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.float32)

    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    use_fp16 = bool(cfg["fp16"]) and torch.cuda.is_available()
    wanted = {
        "output_dir": out_dir,
        "beta": cfg["beta"],
        "learning_rate": cfg["learning_rate"],
        "lr_scheduler_type": cfg["lr_scheduler_type"],
        "warmup_ratio": cfg["warmup_ratio"],
        "num_train_epochs": cfg["num_train_epochs"],
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "max_length": cfg["max_length"],
        "fp16": use_fp16,
        "bf16": False,
        "logging_steps": cfg["logging_steps"],
        "eval_strategy": "steps",
        "eval_steps": cfg["eval_steps"],
        "save_strategy": "no",
        "report_to": "none",
        "seed": cfg["seed"],
        "max_steps": cfg.get("max_steps", -1),
    }
    fields = {f.name for f in dataclasses.fields(DPOConfig)}
    dropped = sorted(k for k in wanted if k not in fields)
    if dropped:
        print(f"[lora] this TRL version has no DPOConfig fields {dropped}; ignoring them")
    dpo_args = DPOConfig(**{k: v for k, v in wanted.items() if k in fields})

    trainer = DPOTrainer(
        model=model, ref_model=None, args=dpo_args, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, peft_config=lora,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"[lora] train={len(train_ds)} val={len(val_ds)} trainable params={trainable:,} / {total:,}")
    write_json(os.path.join(out_dir, "config.json"), {
        "config": cfg, "dpo_args_used": {k: v for k, v in wanted.items() if k in fields},
        "trainable_params": trainable, "total_params": total,
        "train_rows": len(train_ds), "val_rows": len(val_ds), "val_indices": val_idx,
        "environment": {**describe_environment(), "trl": __import__("trl").__version__,
                        "peft": __import__("peft").__version__},
    })

    initial_eval = trainer.evaluate()
    trainer.train()
    final_eval = trainer.evaluate()
    trainer.save_model(os.path.join(out_dir, "final"))
    tokenizer.save_pretrained(os.path.join(out_dir, "final"))
    write_jsonl(os.path.join(out_dir, "metrics.jsonl"), trainer.state.log_history)
    write_json(os.path.join(out_dir, "summary.json"), {"initial_eval": initial_eval, "final_eval": final_eval})
    print({"initial_eval": initial_eval, "final_eval": final_eval})


if __name__ == "__main__":
    main()
