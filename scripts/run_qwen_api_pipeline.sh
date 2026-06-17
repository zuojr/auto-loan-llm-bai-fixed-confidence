#!/usr/bin/env bash
set -euo pipefail
python scripts/03_score_qwen_dashscope_api.py \
  --scoring api_scoring/scoring_pilot_1000.csv \
  --out predictions/qwen_api_pilot.csv \
  --rows-per-call 20

python scripts/03_score_qwen_dashscope_api.py \
  --scoring api_scoring/scoring_full_paired.csv \
  --out predictions/qwen_api_predictions.csv \
  --rows-per-call 20

python scripts/04_merge_predictions.py --pred predictions/qwen_api_predictions.csv --method qwen_api
python scripts/05_run_bai_replay.py \
  --input processed/paired_policy_eval_pool_with_qwen_api.csv \
  --proxy-col proxy_qwen_api \
  --method qwen_api \
  --reps 3000
