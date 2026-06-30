# Results Directory

This directory contains curated summaries for the current PROBE experiment.

Main result table:

- `probe_comparison_table.csv`

Per-method summaries:

- `probe_summary_baseline.csv`
- `probe_summary_qwen_local_icl.csv`
- `probe_summary_qwen_lora_cls_raw.csv`
- `probe_summary_gpt_api.csv`
- `probe_summary_tabpfn.csv`

Per-method arm/proxy diagnostics:

- `probe_arm_params_baseline.csv`
- `probe_arm_params_qwen_local_icl.csv`
- `probe_arm_params_qwen_lora_cls_raw.csv`
- `probe_arm_params_gpt_api.csv`
- `probe_arm_params_tabpfn.csv`

Repetition-level traces are generated as `probe_repetitions_*.csv` and are
excluded from git.
