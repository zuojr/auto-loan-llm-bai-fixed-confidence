#!/usr/bin/env bash
set -euo pipefail
mkdir -p predictions
python scripts/03_score_qwen_dashscope_api.py \
  --scoring api_scoring/scoring_full_paired.csv \
  --out predictions/qwen_api_predictions.csv \
  --rows-per-call ${ROWS_PER_CALL:-10}
python scripts/04_merge_predictions.py --pred predictions/qwen_api_predictions.csv --method qwen_api
python scripts/05_run_fixed_confidence_replay.py \
  --input processed/paired_policy_eval_pool_with_qwen_api.csv \
  --proxy-col proxy_qwen_api \
  --method qwen_api \
  --reps ${REPS:-3000} \
  --delta ${DELTA:-0.05} \
  --batch-size ${BATCH_SIZE:-20}
