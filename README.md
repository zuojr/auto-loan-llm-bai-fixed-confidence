# Auto-loan LLM Gen-CV-BAI Fixed-confidence Experiment

This repository contains code and curated outputs for a fixed-confidence Gen-CV-BAI experiment on the auto-loan data.

The main comparison uses four methods:

1. `baseline`: reward-only fixed-confidence BAI.
2. `qwen_local_icl`: local open-weight `Qwen/Qwen2.5-7B-Instruct` in-context scoring, no fine-tuning.
3. `qwen_lora_cls_raw`: the same `Qwen/Qwen2.5-7B-Instruct` base checkpoint with LoRA fine-tuning, using the raw yes/no classification proxy.
4. `gpt_api`: GPT API in-context scoring, no fine-tuning.

The structured boosted proxy is not reported as a method. The main metric is fixed-confidence stopping time, not fixed-budget accuracy.

## Repository map

```text
api_scoring/      Prompt templates and generated scoring inputs
configs/          Configuration placeholders
data/             Raw data location; raw data is not tracked
figures/          Main fixed-confidence figures
predictions/      Generated LLM predictions; not tracked
processed/        Generated replay pools; not tracked
qwen_finetune/    Generated Qwen training data and adapters; not tracked
results/          Curated fixed-confidence result summaries
scripts/          End-to-end experiment scripts
src/              Shared utilities and replay simulator
tests/            Unit tests
```

## Included results

The paper-ready table is:

```text
results/fixed_confidence_paper_main_table_abs_rho.csv
```

It reports absolute reward-proxy correlations as proxy-strength diagnostics because negative correlation can still be exploited with a signed control variate.

Main paper figures:

```text
figures/fixed_confidence_stopping_cdf_baseline.png
figures/fixed_confidence_stopping_cdf_qwen_local_icl.png
figures/fixed_confidence_stopping_cdf_qwen_lora_cls_raw.png
figures/fixed_confidence_stopping_cdf_gpt_api.png
```

## Install CPU environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_cpu.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_cpu.txt
```

## Data

Raw data and generated replay pools are intentionally not tracked in git.

To regenerate the full experiment inputs, place the raw auto-loan CSV at:

```text
data/raw/CPRM_AutoLOan_OnlineAutoLoanData.csv
```

Then run:

```bash
python scripts/01_prepare_replay_environment.py
```

This writes the processed replay pool, LLM scoring rows, and Qwen fine-tuning JSONL files.

LLM prompts must not expose replay-environment columns such as `prob_accept`, `expected_reward_env`, or `observed_apply`.

## Run the fixed-confidence experiment

Baseline:

```bash
bash scripts/run_baseline_fixed_confidence.sh
```

Qwen local ICL on AutoDL:

```bash
pip install -r requirements_autodl_qwen.txt
export HF_ENDPOINT=https://hf-mirror.com
export QWEN_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LOCAL_ICL_ROWS_PER_CALL=6
export LOCAL_ICL_N_EXAMPLES=12
export REPS=3000
export DELTA=0.05
bash scripts/run_qwen_local_icl_autodl_fixed_confidence.sh
```

Qwen LoRA raw classification proxy on AutoDL:

```bash
pip install -r requirements_autodl_qwen.txt
export HF_ENDPOINT=https://hf-mirror.com
export QWEN_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export QWEN_CLS_LORA_DIR="qwen_finetune/qwen_lora_cls_auto_loan"
export EPOCHS=1
export TRAIN_BATCH_SIZE=1
export GRAD_ACCUM=16
export CLS_PRED_BATCH_SIZE=16
export REPS=3000
export DELTA=0.05
bash scripts/run_qwen_lora_cls_raw_autodl_fixed_confidence.sh
```

GPT API ICL:

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export OPENAI_MODEL="<YOUR_GPT_MODEL>"
export ROWS_PER_CALL=50
export REPS=3000
export DELTA=0.05
bash scripts/run_gpt_api_pipeline_fixed_confidence.sh
```

Set API keys only in the shell. Do not write keys to files or commit them.

## Collect final table

After the four main methods finish:

```bash
python scripts/06_collect_fixed_confidence_results.py \
  --methods baseline,qwen_local_icl,qwen_lora_cls_raw,gpt_api
```

This writes:

```text
results/fixed_confidence_final_comparison_table.csv
```

For paper presentation, use:

```text
results/fixed_confidence_paper_main_table_abs_rho.csv
```

## Tests

```bash
python -m pytest tests -q
python -m py_compile scripts/02_score_openai_gpt_api.py scripts/04_merge_predictions.py scripts/05_run_fixed_confidence_replay.py scripts/06_collect_fixed_confidence_results.py
```

## Notes

- `qwen_local_icl` and `qwen_lora_cls_raw` must use the same `QWEN_BASE_MODEL`.
- The fixed-confidence replay is implemented in `src/fixed_confidence_replay.py`.
- The old fixed-budget script is retained only for diagnostics and is not the main experiment.
- Full prediction CSVs, processed replay pools, LoRA adapters, and raw data are generated artifacts and are excluded from git.
