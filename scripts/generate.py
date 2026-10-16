"""Generate responses for a fixed evaluation set.

Writes a JSON list of {id, instruction, output, generator, dataset, ...} and a
sibling ``.meta.json`` recording the exact decoding settings. The file is
rewritten after every batch, so a rerun resumes and skips finished prompts.

Examples:
  # 2-5 prompt sanity check before the full run (manual section 12)
  python scripts/generate.py --model_path outputs/sft/best --generator sft \
      --eval_set eval_sets/alpaca_eval_300_seed42.json --output outputs/generations/sanity_sft.json --limit 3

  python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base \
      --eval_set eval_sets/alpaca_eval_300_seed42.json --output outputs/generations/alpaca_base.json

  # SmolLM2 persona model: its own chat template, optional LoRA adapter
  python scripts/generate.py --model_path HuggingFaceTB/SmolLM2-360M-Instruct --adapter_path outputs/lora_dpo/final \
      --prompt_format chat_template --generator smollm2-lora-dpo \
      --eval_set eval_sets/persona_style_eval.json --output outputs/generations/persona_lora.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

from pa2.prompts import format_prompt, load_tokenizer, stop_token_ids  # noqa: E402
from pa2.utils import read_json, set_seed, write_json  # noqa: E402

KEEP_FIELDS = ("category", "expected_behavior", "source_dataset")


def build_prompt(tokenizer, instruction: str, prompt_format: str) -> str:
    if prompt_format == "chatml":
        return format_prompt(instruction)
    if prompt_format == "chat_template":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
        )
    raise ValueError(prompt_format)


def main():
    from transformers import AutoModelForCausalLM

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", default=None, help="optional PEFT/LoRA adapter on top of model_path")
    p.add_argument("--eval_set", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--generator", required=True, help="name stored in each record, e.g. sft, dpo_uf")
    p.add_argument("--prompt_format", choices=["chatml", "chat_template"], default="chatml")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity check)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    items = read_json(args.eval_set)[: args.limit]
    done = {r["id"]: r for r in read_json(args.output)} if os.path.exists(args.output) else {}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} prompts, {len(done)} already generated, {len(todo)} to go")

    decoding = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    meta = {**vars(args), "decoding": decoding}
    write_json(args.output.replace(".json", "") + ".meta.json", meta)
    if not todo:
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32
    if args.prompt_format == "chatml":
        tokenizer = load_tokenizer(args.model_path)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model.eval()

    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": stop_token_ids(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    stops = set(gen_kwargs["eos_token_id"])
    order = {it["id"]: i for i, it in enumerate(items)}
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start : start + args.batch_size]
        prompts = [build_prompt(tokenizer, it["instruction"], args.prompt_format) for it in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        for it, toks in zip(chunk, new_tokens):
            text = tokenizer.decode(toks, skip_special_tokens=True).strip()
            record = {
                "id": it["id"],
                "instruction": it["instruction"],
                "output": text,
                "generator": args.generator,
                "dataset": it.get("dataset", os.path.basename(args.eval_set)),
                "hit_max_new_tokens": not any(int(t) in stops for t in toks),
            }
            record.update({k: it[k] for k in KEEP_FIELDS if k in it})
            done[it["id"]] = record
        write_json(args.output, sorted(done.values(), key=lambda r: order.get(r["id"], 1 << 30)))
        print(f"  {min(start + args.batch_size, len(todo))}/{len(todo)}", flush=True)

    if args.limit:
        for r in list(done.values())[: args.limit]:
            print("=" * 80, "\nINSTRUCTION:", r["instruction"][:300], "\n--- OUTPUT:\n", r["output"][:1500])


if __name__ == "__main__":
    main()
