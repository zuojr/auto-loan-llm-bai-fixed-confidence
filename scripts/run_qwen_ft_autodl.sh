#!/usr/bin/env bash
set -euo pipefail
# Run this on AutoDL GPU machine from the project root.
pip install -r requirements_autodl_qwen.txt

# Optional: if huggingface access is slow, set HF_ENDPOINT=https://hf-mirror.com before running.
# export HF_ENDPOINT=https://hf-mirror.com

python scripts/07_train_qwen_lora_autodl.py \
  --train-jsonl qwen_finetune/train_full.jsonl \
  --eval-jsonl qwen_finetune/val_full.jsonl \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --out-dir qwen_finetune/qwen_lora_auto_loan \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 8

python scripts/08_predict_qwen_lora_autodl.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-dir qwen_finetune/qwen_lora_auto_loan \
  --scoring api_scoring/scoring_full_paired.csv \
  --out predictions/qwen_ft_predictions.csv \
  --batch-size 8

python scripts/04_merge_predictions.py --pred predictions/qwen_ft_predictions.csv --method qwen_ft
python scripts/05_run_bai_replay.py \
  --input processed/paired_policy_eval_pool_with_qwen_ft.csv \
  --proxy-col proxy_qwen_ft \
  --method qwen_ft \
  --reps 3000
