#!/usr/bin/env bash
set -euo pipefail
mkdir -p predictions
python scripts/02_score_openai_gpt_api.py \
  --scoring api_scoring/scoring_full_paired.csv \
  --out predictions/gpt_api_predictions.csv \
  --rows-per-call ${ROWS_PER_CALL:-10}
python scripts/04_merge_predictions.py --pred predictions/gpt_api_predictions.csv --method gpt_api
python scripts/05_run_fixed_confidence_replay.py \
  --input processed/paired_policy_eval_pool_with_gpt_api.csv \
  --proxy-col proxy_gpt_api \
  --method gpt_api \
  --reps ${REPS:-3000} \
  --delta ${DELTA:-0.05} \
  --batch-size ${BATCH_SIZE:-20}
