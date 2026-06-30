# Auto-loan Generator-Augmented BAI PROBE Experiment

This repository contains code and curated outputs for the auto-loan best-arm
identification experiment using the latest PDF algorithm, **PROBE**: PRoxy OLS
for Best-arm Exploration.

The reported replay algorithm is PDF Algorithm 1:

- calibration batch per arm;
- OLS intercept and slope on fresh proxy-reward batches;
- chi-square residual-variance upper certificates;
- monotone certificate updates;
- phase elimination with `epsilon_r = 2^-r`.

The LLM/TabPFN scoring stage is unchanged. Existing prediction files can be
reused; only the replay/simulation stage changes.

## Methods

The current comparison uses:

1. `baseline`: reward-only PROBE, with no proxy.
2. `qwen_local_icl`: local `Qwen/Qwen2.5-7B-Instruct` in-context scoring.
3. `qwen_lora_cls_raw`: same Qwen base checkpoint with LoRA fine-tuning, using
   the raw yes/no classification proxy.
4. `gpt_api`: GPT API in-context scoring.
5. `tabpfn`: structured tabular proxy reference.

The main metric is fixed-confidence stopping time under PROBE.

## Repository Map

```text
api_scoring/      Prompt templates and generated scoring inputs
data/             Raw data location; raw data is not tracked
figures/          Curated PROBE figures
predictions/      Generated LLM/TabPFN predictions; not tracked
processed/        Generated replay pools; not tracked
qwen_finetune/    Generated Qwen training data and adapters; not tracked
results/          Curated PROBE result summaries
scripts/          Scoring, proxy-building, and replay scripts
src/              Shared utilities and PROBE simulator
tests/            Unit tests
```

## Included PROBE Results

Main table:

```text
results/probe_comparison_table.csv
```

Per-method summaries:

```text
results/probe_summary_baseline.csv
results/probe_summary_qwen_local_icl.csv
results/probe_summary_qwen_lora_cls_raw.csv
results/probe_summary_gpt_api.csv
results/probe_summary_tabpfn.csv
```

Per-method figures:

```text
figures/probe_stopping_cdf_baseline.png
figures/probe_stopping_cdf_qwen_local_icl.png
figures/probe_stopping_cdf_qwen_lora_cls_raw.png
figures/probe_stopping_cdf_gpt_api.png
figures/probe_stopping_cdf_tabpfn.png
```

Current 3000-repetition PROBE results:

| method | mean stop pulls | median | q90 | correct at stop | saving vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 105159.4 | 105206.5 | 106714.3 | 1.000 | 0.0% |
| qwen_local_icl | 101697.6 | 101789.0 | 103467.2 | 1.000 | 3.3% |
| qwen_lora_cls_raw | 52036.2 | 52315.5 | 54477.1 | 1.000 | 50.5% |
| gpt_api | 95590.5 | 95715.5 | 97560.4 | 1.000 | 9.1% |
| tabpfn | 40593.6 | 40778.5 | 43101.4 | 1.000 | 61.4% |

## Install CPU Environment

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

To regenerate full experiment inputs, place the raw auto-loan CSV at:

```text
data/raw/CPRM_AutoLOan_OnlineAutoLoanData.csv
```

Then run:

```bash
python scripts/01_prepare_replay_environment.py
```

This writes processed replay pools, LLM scoring rows, and Qwen fine-tuning
JSONL files. LLM prompts must not expose replay-environment columns such as
`prob_accept`, `expected_reward_env`, or `observed_apply`.

## Generate Proxy Predictions

Use the existing scoring scripts to create prediction CSVs. API keys must only
be set in the shell and must never be written to files or committed.

Qwen local ICL and Qwen LoRA must use the same base checkpoint:

```bash
export QWEN_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

After prediction files are available, create method replay pools with
`scripts/04_merge_predictions.py` or the method-specific pipeline scripts.

## Run PROBE Replay

Baseline:

```bash
python scripts/19_run_probe_replay.py \
  --method baseline \
  --reps 3000 \
  --delta 0.05 \
  --kappa 1.0
```

Qwen local ICL:

```bash
python scripts/19_run_probe_replay.py \
  --input processed/paired_policy_eval_pool_with_qwen_local_icl.csv \
  --proxy-col proxy_qwen_local_icl \
  --method qwen_local_icl \
  --reps 3000 \
  --delta 0.05 \
  --kappa 1.0
```

Qwen LoRA raw classification proxy:

```bash
python scripts/19_run_probe_replay.py \
  --input processed/paired_policy_eval_pool_with_qwen_lora_cls_raw.csv \
  --proxy-col proxy_qwen_lora_cls_raw \
  --method qwen_lora_cls_raw \
  --reps 3000 \
  --delta 0.05 \
  --kappa 1.0
```

GPT API ICL:

```bash
python scripts/19_run_probe_replay.py \
  --input processed/paired_policy_eval_pool_with_gpt_api.csv \
  --proxy-col proxy_gpt_api \
  --method gpt_api \
  --reps 3000 \
  --delta 0.05 \
  --kappa 1.0
```

TabPFN structured proxy:

```bash
python scripts/19_run_probe_replay.py \
  --input processed/paired_policy_eval_pool_with_tabpfn.csv \
  --proxy-col proxy_tabpfn \
  --method tabpfn \
  --reps 3000 \
  --delta 0.05 \
  --kappa 1.0
```

Collect the final PROBE table:

```bash
python scripts/20_collect_probe_results.py
```

This writes:

```text
results/probe_comparison_table.csv
```

## TabPFN Structured Proxy

TabPFN predicts acceptance probability from structured application-offer
features. The replay proxy is:

```text
proxy_tabpfn = offer_mp * pred_prob
```

TabPFN uses only historical-block labels and a strict structured feature
whitelist. It does not use leakage columns such as `prob_accept`,
`expected_reward_env`, `observed_apply`, realized rewards, proxy columns,
`record_id`, `base_row_id`, or `profile_text`.

## Tests

```bash
python -m pytest -q
python -m py_compile src/probe_replay.py scripts/19_run_probe_replay.py scripts/20_collect_probe_results.py
```

## Notes

- The current replay/simulation implementation is `src/probe_replay.py`.
- Full prediction CSVs, processed replay pools, LoRA adapters, repetition-level
  replay traces, and raw data are generated artifacts and are excluded from git.
- The older fixed-budget and pre-PROBE replay scripts are retained only as
  non-reported diagnostics; the reported algorithm is PROBE.
