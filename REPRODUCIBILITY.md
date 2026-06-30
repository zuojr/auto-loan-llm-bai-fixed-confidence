# Reproducibility Guide

This repository contains implementation code and curated outputs for the
current Auto-loan Gen-CV-BAI experiment using PDF Algorithm 1 PROBE.

## Included

- experiment scripts in `scripts/`
- shared replay and utility code in `src/`
- unit tests in `tests/`
- dependency files for CPU and AutoDL/GPU runs
- curated PROBE result summaries in `results/`
- curated PROBE CDF figures in `figures/`

Large generated files are intentionally excluded from git: raw data, processed
replay pools, model adapters, full prediction CSVs, API scoring CSVs, and
repetition-level replay traces.

## Data Preparation

Place the raw auto-loan CSV at:

```text
data/raw/CPRM_AutoLOan_OnlineAutoLoanData.csv
```

Then run:

```bash
python scripts/01_prepare_replay_environment.py
```

This generates the processed replay pools, API scoring rows, and Qwen training
JSONL files. LLM prompts must not include replay-environment columns such as
`prob_accept`, `expected_reward_env`, or `observed_apply`.

## CPU Environment

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

## Proxy Generation

Generate or sync prediction CSVs for:

- `qwen_local_icl`
- `qwen_lora_cls_raw`
- `gpt_api`
- `tabpfn`

Qwen local ICL and Qwen LoRA should use the same base checkpoint:

```bash
export QWEN_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

API keys must be provided only through shell environment variables and must not
be written to repository files.

## PROBE Replay

Run all replay jobs:

```bash
bash scripts/run_baseline_fixed_confidence.sh
bash scripts/run_qwen_local_icl_autodl_fixed_confidence.sh
bash scripts/run_qwen_lora_cls_raw_autodl_fixed_confidence.sh
bash scripts/run_gpt_api_pipeline_fixed_confidence.sh
bash scripts/run_tabpfn_autodl_fixed_confidence.sh
```

Collect the comparison table and draw the combined CDF figure:

```bash
python scripts/20_collect_probe_results.py
python scripts/21_plot_probe_cdf_all_methods.py
```

Main outputs:

```text
results/probe_comparison_table.csv
figures/probe_stopping_cdf_all_methods.png
```

Per-method outputs:

```text
results/probe_summary_baseline.csv
results/probe_summary_qwen_local_icl.csv
results/probe_summary_qwen_lora_cls_raw.csv
results/probe_summary_gpt_api.csv
results/probe_summary_tabpfn.csv
figures/probe_stopping_cdf_baseline.png
figures/probe_stopping_cdf_qwen_local_icl.png
figures/probe_stopping_cdf_qwen_lora_cls_raw.png
figures/probe_stopping_cdf_gpt_api.png
figures/probe_stopping_cdf_tabpfn.png
```

## Reported Metric

The main metric is fixed-confidence stopping time under PROBE, not fixed-budget
accuracy. The comparison table reports:

- `mean_stop_pulls`
- `median_stop_pulls`
- `q90_stop_pulls`
- `empirical_correct_at_stop`
- absolute reward-proxy correlations for diagnostics
- `sample_saving_vs_baseline`

The structured boosted proxy is not a reported method.

## Verification

```bash
python -m pytest -q
python -m py_compile src/probe_replay.py scripts/19_run_probe_replay.py scripts/20_collect_probe_results.py scripts/21_plot_probe_cdf_all_methods.py
```
