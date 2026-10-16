"""End-to-end pipeline smoke test with a tiny randomly initialised Qwen2 model.

Builds a 2-layer Qwen2 model that uses the real Qwen2.5 tokenizer, runs a few
SFT optimizer steps, then a few DPO steps from that SFT checkpoint, and checks
that best/final checkpoints and metrics.jsonl are written. Runs on CPU in about
a minute; on Kaggle it also exercises the two-GPU placement.

  python scripts/smoke_train.py --work_dir /tmp/pa2_smoke
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pa2.dpo import train_dpo  # noqa: E402
from pa2.prompts import load_tokenizer  # noqa: E402
from pa2.sft import train_sft  # noqa: E402
from pa2.utils import read_jsonl, write_jsonl  # noqa: E402


def main():
    from transformers import AutoConfig, AutoModelForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()
    work = args.work_dir
    os.makedirs(work, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)
    config = AutoConfig.from_pretrained(args.tokenizer)
    config.update({"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                   "num_attention_heads": 4, "num_key_value_heads": 2, "max_window_layers": 2})
    if getattr(config, "layer_types", None):
        config.layer_types = config.layer_types[:2]
    tiny_dir = os.path.join(work, "tiny_base")
    AutoModelForCausalLM.from_config(config).save_pretrained(tiny_dir)
    tokenizer.save_pretrained(tiny_dir)

    sft_rows = [{"id": f"s{i}", "prompt": f"Say the number {i}.", "response": f"The number is {i}."} for i in range(24)]
    dpo_rows = [{"id": f"d{i}", "prompt": f"Is {i} even?", "chosen": "Let me check carefully.", "rejected": "No idea."}
                for i in range(24)]
    for name, rows in (("sft", sft_rows), ("dpo", dpo_rows)):
        write_jsonl(os.path.join(work, f"{name}_train.jsonl"), rows[:16])
        write_jsonl(os.path.join(work, f"{name}_val.jsonl"), rows[16:])

    common = {"num_workers": 0, "overwrite": True, "log_every": 1, "eval_every": 2}
    print(train_sft({
        **common, "model_name": tiny_dir, "output_dir": os.path.join(work, "sft"),
        "train_path": os.path.join(work, "sft_train.jsonl"), "val_path": os.path.join(work, "sft_val.jsonl"),
        "per_device_batch_size": 2, "gradient_accumulation_steps": 2, "learning_rate": 1e-3, "max_length": 64,
    }))
    print(train_dpo({
        **common, "sft_checkpoint": os.path.join(work, "sft", "best"), "output_dir": os.path.join(work, "dpo"),
        "train_path": os.path.join(work, "dpo_train.jsonl"), "val_path": os.path.join(work, "dpo_val.jsonl"),
        "gradient_accumulation_steps": 2, "learning_rate": 1e-3, "max_length": 64, "max_prompt_length": 32,
    }))

    for run in ("sft", "dpo"):
        for sub in ("best", "final"):
            path = os.path.join(work, run, sub, "checkpoint_info.json")
            assert os.path.exists(path), f"missing {path}"
        metrics = read_jsonl(os.path.join(work, run, "metrics.jsonl"))
        assert any(m["split"] == "val" for m in metrics), f"no validation metrics for {run}"
    dpo_val = [m for m in read_jsonl(os.path.join(work, "dpo", "metrics.jsonl")) if m["split"] == "val"]
    assert dpo_val[0]["reward_accuracy"] == 0.0, "policy == reference at step 0, so strict accuracy must be 0"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
