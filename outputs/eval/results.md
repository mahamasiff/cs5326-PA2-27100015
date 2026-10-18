## DPO vs SFT (candidate perspective)

| Comparison | Criterion | Judge | N | Wins | Ties | Losses | Unknown | Failures | Win rate (ties=0.5) | Win rate (no ties) | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | google/gemini-3.6-flash | 300 | 139 | 57 | 104 | 0 | 0 | 55.8% | 57.2% | 0 |
| dpo_uf vs sft (dpo_uf_vs_sft) | winner | google/gemini-3.6-flash | 300 | 122 | 43 | 135 | 0 | 0 | 47.8% | 47.5% | 0 |
| lora_dpo vs smollm2_instruct (lora_vs_base) | style | google/gemini-3.6-flash | 40 | 21 | 18 | 1 | 0 | 0 | 75.0% | 95.5% | 0 |
| lora_dpo vs smollm2_instruct (lora_vs_base) | quality | google/gemini-3.6-flash | 40 | 12 | 2 | 26 | 0 | 0 | 32.5% | 31.6% | 0 |

## By category

| Comparison | Criterion | Category | Wins | Ties | Losses | Win rate (ties=0.5) |
|---|---|---|---|---|---|---|
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | benign_helpfulness | 15 | 1 | 14 | 51.7% |
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | calibration_and_uncertainty | 11 | 9 | 10 | 51.7% |
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | cyber_abuse | 15 | 7 | 8 | 61.7% |
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | harmful_instructions | 18 | 5 | 7 | 68.3% |
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | hate_or_harassment | 13 | 8 | 9 | 56.7% |
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | legal_financial_safety | 13 | 3 | 14 | 48.3% |
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | medical_safety | 12 | 8 | 10 | 53.3% |
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | privacy_and_doxxing | 13 | 8 | 9 | 56.7% |
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | safe_transformation | 17 | 8 | 5 | 70.0% |
| dpo_hh vs sft (dpo_hh_vs_sft) | winner | self_harm_support | 12 | 0 | 18 | 40.0% |
| dpo_uf vs sft (dpo_uf_vs_sft) | winner | all | 122 | 43 | 135 | 47.8% |
| lora_dpo vs smollm2_instruct (lora_vs_base) | style | coding | 7 | 3 | 0 | 85.0% |
| lora_dpo vs smollm2_instruct (lora_vs_base) | style | cooking | 7 | 2 | 1 | 80.0% |
| lora_dpo vs smollm2_instruct (lora_vs_base) | style | general_help | 1 | 9 | 0 | 55.0% |
| lora_dpo vs smollm2_instruct (lora_vs_base) | style | safety_sensitive | 6 | 4 | 0 | 80.0% |
| lora_dpo vs smollm2_instruct (lora_vs_base) | quality | coding | 2 | 2 | 6 | 30.0% |
| lora_dpo vs smollm2_instruct (lora_vs_base) | quality | cooking | 2 | 0 | 8 | 20.0% |
| lora_dpo vs smollm2_instruct (lora_vs_base) | quality | general_help | 4 | 0 | 6 | 40.0% |
| lora_dpo vs smollm2_instruct (lora_vs_base) | quality | safety_sensitive | 4 | 0 | 6 | 40.0% |
