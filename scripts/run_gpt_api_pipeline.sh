#!/usr/bin/env bash
set -euo pipefail
# First run a pilot; inspect predictions/gpt_pilot.csv, then remove --limit and use scoring_full_paired.csv.
python scripts/02_score_openai_gpt_api.py \
  --scoring api_scoring/scoring_pilot_1000.csv \
  --out predictions/gpt_api_pilot.csv \
  --rows-per-call 20

# Full run:
python scripts/02_score_openai_gpt_api.py \
  --scoring api_scoring/scoring_full_paired.csv \
  --out predictions/gpt_api_predictions.csv \
  --rows-per-call 20

python scripts/04_merge_predictions.py --pred predictions/gpt_api_predictions.csv --method gpt_api
python scripts/05_run_bai_replay.py \
  --input processed/paired_policy_eval_pool_with_gpt_api.csv \
  --proxy-col proxy_gpt_api \
  --method gpt_api \
  --reps 3000
