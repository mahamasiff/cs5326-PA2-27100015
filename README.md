# CS5326 PA2 — Instruction Tuning and DPO on a Small Language Model

From-scratch SFT and DPO for `Qwen/Qwen2.5-0.5B`, plus TRL `DPOTrainer` + LoRA persona tuning of
`HuggingFaceTB/SmolLM2-360M-Instruct`, trained on Kaggle T4 x2 and evaluated with blinded pairwise
judgments from a Kaggle model-proxy judge. Built on the course
[starter repository](https://github.com/ahmad903501/small-model-alignment-starter); the assignment manual
is `docs/assignment_manual.pdf`.

## Where each task lives

| Manual task | Implementation |
|---|---|
| Adapter contract (section 4.1) | `submission_adapter.py` → `src/pa2/batching.py`, `src/pa2/losses.py` |
| Task 1: setup | `notebooks/01_setup_baseline.ipynb` |
| Task 2: baseline generation | `scripts/make_alpaca_eval_subset.py`, `scripts/generate.py`, notebook 01 |
| Task 3: SFT | `src/pa2/sft.py`, `scripts/train_sft.py`, `configs/sft.yaml`, notebook 02 |
| Task 4: two DPO runs | `src/pa2/dpo.py`, `scripts/train_dpo.py`, `configs/dpo_{hh,uf}.yaml`, notebooks 03–04 |
| Task 5: LoRA persona DPO | `scripts/train_lora_dpo.py`, `configs/lora_dpo.yaml`, notebook 05 |
| Task 6: evaluation outputs | `scripts/generate.py`, `scripts/build_pairs.py`, notebook 06 |
| Task 7: Kaggle judging | `src/pa2/judging.py`, `scripts/judge_pairwise_kaggle.py`, notebook 07 |
| Task 8: report material | `scripts/plot_metrics.py`, `scripts/collect_results.py` |

SFT and both from-scratch DPO runs use only Transformers (model/tokenizer loading) and PyTorch
(DataLoader, optimizers, `LambdaLR`, AMP). No HF `Trainer`, no TRL. TRL/PEFT are used only for Task 5.

## Implementation decisions (state these in the report)

- **Prompt template** (`src/pa2/prompts.py`), shared by SFT, DPO, and generation:
  ```
  <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n
  ```
  For Qwen, `eos_token` is set to `<|im_end|>` (appended after every response, and the generation stop token)
  and `pad_token` to `<|endoftext|>`. The SmolLM2 persona task uses that model's own chat template.
- **Batching**: prompt and response are tokenized separately without special tokens and concatenated; EOS is
  appended before truncation to `max_length`; prompt and padding labels are `-100`; right padding to the
  longest row in the batch; row order is preserved.
- **SFT loss reduction**: mean negative log-likelihood over all unmasked response tokens in the batch.
- **DPO**: summed response-token log-probs; loss `-log σ(β[(πw−πl) − (refw−refl)])` averaged over the batch;
  reference model frozen, fp16, `cuda:1`, under `no_grad`; policy fp32 on `cuda:0` with fp16 autocast and
  `GradScaler`. Prompts longer than 256 tokens keep their **last** 256 tokens (preserving the assistant header).
  Validation runs at step 0 (reward accuracy is exactly 0 there because policy = reference) and every 50
  optimizer steps; `best` = highest validation reward accuracy, ties broken by reward margin.
- **Data** (`scripts/prepare_data.py`, seed 42): UltraChat-200k first turn (30k/1k), Anthropic HH-RLHF
  single-turn only (15k/1k), UltraFeedback-binarized single-turn with strictly higher chosen score (15k/1k).
  Selected IDs and filter counts are written to `data/manifests/`.
- **Evaluation sets**: `eval_sets/alpaca_eval_300_seed42.json` (`random.Random(42).sample` of the 805
  AlpacaEval prompts), the provided `eval_sets/hh_rlhf_eval_300_seed42.json`, and
  `eval_sets/persona_style_eval.json` (40 hand-written prompts: cooking, coding, general help, safety).
- **Judging**: blinded A/B order (seed 42), identities only in `hidden`; general rubric for AlpacaEval,
  safety rubric for the HH set, and separate *style* and *quality* verdicts for the persona set.

## Running on Kaggle

Following section 5.4 of the manual: one training notebook, a separate generation notebook that takes the training
outputs as an input dataset, and a separate evaluation notebook that takes the generation outputs as an
input dataset. Use **GPU T4 x2** with Internet on (the evaluation notebook needs no GPU). Start each with
**Save Version → Save & Run All (Commit)**, then attach its saved output to the next notebook via
**Add Input**. Every notebook clones this repo into `/kaggle/working/cs5326-PA2-27100015` and writes
results under `outputs/`.

| Notebook | Tasks | Input | Produces |
|---|---|---|---|
| `01_training` | 1, 3, 4 (both DPO runs), 5 | — | `outputs/{sft,dpo_hh,dpo_uf,lora_dpo}/`, `outputs/plots/`, `data/manifests/` |
| `02_generation` | 2, 6 | saved `01_training` | `outputs/generations/*.json`, `outputs/eval/*_pairs.jsonl` |
| `03_evaluation` (utility benchmark) | 7 | saved `02_generation` | `outputs/eval/*_judged.jsonl`, `*_summary.json`, `results.md` |

The executed copies of all three notebooks, with every cell output from the reported runs, are in
`notebooks/executed/`.

Equivalent commands from the repository root:

```bash
python scripts/prepare_data.py --out_dir data/processed --which sft hh uf
python scripts/train_sft.py --config configs/sft.yaml
python scripts/train_dpo.py --config configs/dpo_hh.yaml --set sft_checkpoint=outputs/sft/best
python scripts/train_dpo.py --config configs/dpo_uf.yaml --set sft_checkpoint=outputs/sft/best
python scripts/train_lora_dpo.py --config configs/lora_dpo.yaml
python scripts/generate.py --model_path outputs/sft/best --generator sft \
    --eval_set eval_sets/alpaca_eval_300_seed42.json --output outputs/generations/alpaca_sft.json
python scripts/build_pairs.py --candidate outputs/generations/alpaca_dpo_uf.json --candidate_name dpo_uf \
    --baseline outputs/generations/alpaca_sft.json --baseline_name sft --output outputs/eval/dpo_uf_vs_sft_pairs.jsonl
python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl \
    --output_path outputs/eval/dpo_uf_vs_sft_judged.jsonl --judge_model google/gemini-3.6-flash \
    --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
```

Any config value can be overridden with `--set key=value` (e.g. `--set max_steps=20` for a pilot run).

## Tests

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py   # public smoke tests
PYTHONPATH=. python -m pytest -q tests/                               # + edge cases and judging
python scripts/smoke_train.py --work_dir /tmp/pa2_smoke               # tiny-model SFT -> DPO pipeline
```

`tests/test_edge_cases.py` covers truncation at the EOS boundary, prompts longer than `max_length`,
tokenizers without pad/EOS tokens, mixed lengths and row order, exact log-prob and loss values, the
reference model receiving no gradients, and small optimization steps. `tests/test_judging.py` covers
blinding, parsing, retries, failure logging, and resume.

## Outputs and reproducibility

| Artifact | Path |
|---|---|
| Run configs + environment | `outputs/<run>/config.json` |
| Training/validation metrics | `outputs/<run>/metrics.jsonl`, `outputs/<run>/summary.json` |
| Checkpoints (Kaggle outputs, not in git) | `outputs/<run>/{best,final}/` |
| Data subsets | `data/manifests/*_ids.json`, `data/manifests/*_stats.json` |
| Generations + decoding settings | `outputs/generations/*.json`, `*.meta.json` |
| Secondary analysis (loops, length, refusals, judge consistency) | `outputs/eval/analysis.json` (`scripts/analyze_eval.py`) |
| Pairwise files, judgments, summaries | `outputs/eval/` |
| Plots | `outputs/plots/` |

Seeds: 42 everywhere (data selection, shuffling, AlpacaEval subset, A/B order, LoRA split).
