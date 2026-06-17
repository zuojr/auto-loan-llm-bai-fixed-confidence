# Fixed-confidence workflow

Use this workflow for the main paper experiment.

## Methods

- `baseline`: reward-only fixed-confidence BAI.
- `qwen_local_icl`: local `Qwen/Qwen2.5-7B-Instruct` ICL, no fine-tuning.
- `qwen_lora_cls_raw`: same base Qwen checkpoint with LoRA/QLoRA fine-tuning, using the raw yes/no classification proxy.
- `gpt_api`: GPT API ICL, no fine-tuning.

## Run order

```bash
bash scripts/run_baseline_fixed_confidence.sh
bash scripts/run_qwen_local_icl_autodl_fixed_confidence.sh   # AutoDL/GPU
bash scripts/run_qwen_lora_cls_raw_autodl_fixed_confidence.sh # AutoDL/GPU
bash scripts/run_gpt_api_pipeline_fixed_confidence.sh        # API key needed
python scripts/06_collect_fixed_confidence_results.py --methods baseline,qwen_local_icl,qwen_lora_cls_raw,gpt_api
```

## Fixed-confidence stopping rule

Each repetition samples until the lower confidence bound of the currently better arm exceeds the upper confidence bound of the other arm. Report stopping-time statistics and correctness at stop.

Main output:

```text
results/fixed_confidence_final_comparison_table.csv
```

For paper writing, use `results/fixed_confidence_paper_main_table_abs_rho.csv`, which reports absolute reward-proxy correlations as proxy-strength diagnostics.
