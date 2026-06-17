#!/usr/bin/env bash
set -euo pipefail
mkdir -p predictions
python scripts/09_predict_qwen_local_icl_autodl.py \
  --base-model ${QWEN_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct} \
  --scoring api_scoring/scoring_full_paired.csv \
  --out predictions/qwen_local_icl_predictions.csv \
  --rows-per-call ${LOCAL_ICL_ROWS_PER_CALL:-6} \
  --n-examples ${LOCAL_ICL_N_EXAMPLES:-12} \
  --max-new-tokens ${LOCAL_ICL_MAX_NEW_TOKENS:-1536}
python scripts/04_merge_predictions.py --pred predictions/qwen_local_icl_predictions.csv --method qwen_local_icl
python scripts/05_run_fixed_confidence_replay.py \
  --input processed/paired_policy_eval_pool_with_qwen_local_icl.csv \
  --proxy-col proxy_qwen_local_icl \
  --method qwen_local_icl \
  --reps ${REPS:-3000} \
  --delta ${DELTA:-0.05} \
  --batch-size ${BATCH_SIZE:-20}
