# Reproducibility Guide

This repository contains the code used for the fixed-confidence Auto-loan Gen-CV-BAI experiment.

## What is included

- Experiment scripts in `scripts/`
- Shared replay and utility code in `src/`
- Unit tests in `tests/`
- Dependency files for CPU and AutoDL/GPU runs
- Curated fixed-confidence result summaries in `results/`
- Main fixed-confidence CDF figures in `figures/`

Large generated files are intentionally excluded from git: raw data, processed replay pools, model adapters, full predictions, and repetition-level traces.

## Data

To reproduce the full pipeline from raw data, place the raw auto-loan CSV at:

```text
data/raw/CPRM_AutoLOan_OnlineAutoLoanData.csv
```

Then run:

```bash
python scripts/01_prepare_replay_environment.py
```

This generates:

```text
processed/paired_policy_eval_pool.csv
api_scoring/scoring_full_paired.csv
api_scoring/scoring_pilot_1000.csv
qwen_finetune/train_full.jsonl
qwen_finetune/val_full.jsonl
```

The LLM prompts must not include replay-environment columns such as `prob_accept`, `expected_reward_env`, or `observed_apply`.

## CPU environment

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

## Baseline

```bash
bash scripts/run_baseline_fixed_confidence.sh
```

Main outputs:

```text
results/fixed_confidence_summary_baseline.csv
results/fixed_confidence_arm_params_baseline.csv
figures/fixed_confidence_stopping_cdf_baseline.png
```

## Qwen local ICL on AutoDL

Use the same base checkpoint as the LoRA run:

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

## Qwen LoRA raw classification proxy on AutoDL

This is the reported Qwen fine-tuning method.

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

## GPT API ICL

Set the API key only in the shell. Do not store it in git or in a file.

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export OPENAI_MODEL="<YOUR_GPT_MODEL>"
export ROWS_PER_CALL=50
export REPS=3000
export DELTA=0.05
bash scripts/run_gpt_api_pipeline_fixed_confidence.sh
```

If using PowerShell:

```powershell
$env:OPENAI_API_KEY = "<YOUR_OPENAI_API_KEY>"
$env:OPENAI_MODEL = "<YOUR_GPT_MODEL>"
$env:ROWS_PER_CALL = "50"
$env:REPS = "3000"
$env:DELTA = "0.05"
bash scripts/run_gpt_api_pipeline_fixed_confidence.sh
```

## Final table

After all four main methods finish:

```bash
python scripts/06_collect_fixed_confidence_results.py \
  --methods baseline,qwen_local_icl,qwen_lora_cls_raw,gpt_api
```

The paper-ready table with absolute reward-proxy correlations is:

```text
results/fixed_confidence_paper_main_table_abs_rho.csv
```

## Main reported metric

The main reported metric is fixed-confidence stopping time, not fixed-budget accuracy.

The primary comparison uses:

- `mean_stop_pulls`
- `median_stop_pulls`
- `q90_stop_pulls`
- `empirical_correct_at_stop`
- `|rho|` for proxy strength diagnostics
- `sample_saving_vs_baseline`

The structured boosted proxy is not a reported method.

