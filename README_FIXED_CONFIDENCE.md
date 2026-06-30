# PROBE Fixed-confidence Workflow

Use this workflow for the current experiment. The replay algorithm is
PDF Algorithm 1 PROBE, not the earlier Gaussian-CI replay.

## Methods

- `baseline`: reward-only PROBE with no proxy.
- `qwen_local_icl`: local `Qwen/Qwen2.5-7B-Instruct` ICL proxy.
- `qwen_lora_cls_raw`: same Qwen base checkpoint with LoRA/QLoRA fine-tuning,
  using the raw yes/no classification proxy.
- `gpt_api`: GPT API ICL proxy.
- `tabpfn`: structured tabular proxy reference.

## Run Order

Generate or sync the prediction and processed proxy-pool files first. Then run:

```bash
python scripts/19_run_probe_replay.py --method baseline --reps 3000 --delta 0.05 --kappa 1.0

python scripts/19_run_probe_replay.py \
  --input processed/paired_policy_eval_pool_with_qwen_local_icl.csv \
  --proxy-col proxy_qwen_local_icl \
  --method qwen_local_icl \
  --reps 3000 --delta 0.05 --kappa 1.0

python scripts/19_run_probe_replay.py \
  --input processed/paired_policy_eval_pool_with_qwen_lora_cls_raw.csv \
  --proxy-col proxy_qwen_lora_cls_raw \
  --method qwen_lora_cls_raw \
  --reps 3000 --delta 0.05 --kappa 1.0

python scripts/19_run_probe_replay.py \
  --input processed/paired_policy_eval_pool_with_gpt_api.csv \
  --proxy-col proxy_gpt_api \
  --method gpt_api \
  --reps 3000 --delta 0.05 --kappa 1.0

python scripts/19_run_probe_replay.py \
  --input processed/paired_policy_eval_pool_with_tabpfn.csv \
  --proxy-col proxy_tabpfn \
  --method tabpfn \
  --reps 3000 --delta 0.05 --kappa 1.0

python scripts/20_collect_probe_results.py
python scripts/21_plot_probe_cdf_all_methods.py
```

## Algorithm

PROBE uses:

- calibration sample size `s_cal = 2 + ceil(64 log(8K/delta) / kappa^2)`;
- fresh-batch OLS intercepts for reward-mean estimates;
- chi-square residual-variance upper certificates;
- monotone certificate updates;
- phase elimination with `epsilon_r = 2^-r`.

## Main Output

```text
results/probe_comparison_table.csv
```

Repetition-level traces are generated as `results/probe_repetitions_*.csv` and
are excluded from git.

## Synthetic Simulations

Run the PROBE version of the Gaussian, certificate-diagnostic, and fitted-model
proxy simulations with:

```bash
python scripts/30_run_probe_simulations.py
```
