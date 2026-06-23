#!/usr/bin/env bash
set -euo pipefail

mkdir -p predictions processed results figures

args=(
  --train "${TABPFN_TRAIN:-processed/historical_train_rows.csv}"
  --scoring "${TABPFN_SCORING:-api_scoring/scoring_full_paired.csv}"
  --out "${TABPFN_OUT:-predictions/tabpfn_predictions.csv}"
  --backend "${TABPFN_BACKEND:-tabpfn}"
  --device "${TABPFN_DEVICE:-cuda}"
)

if [[ -n "${TABPFN_TRAIN_SIZE:-}" ]]; then
  args+=(--train-size "${TABPFN_TRAIN_SIZE}")
fi

if [[ -n "${TABPFN_LIMIT:-}" ]]; then
  args+=(--limit "${TABPFN_LIMIT}")
fi

python scripts/10_predict_tabpfn_autodl.py "${args[@]}"

python scripts/04_merge_predictions.py \
  --pred "${TABPFN_OUT:-predictions/tabpfn_predictions.csv}" \
  --method tabpfn

python scripts/05_run_fixed_confidence_replay.py \
  --input processed/paired_policy_eval_pool_with_tabpfn.csv \
  --proxy-col proxy_tabpfn \
  --method tabpfn \
  --reps "${REPS:-3000}" \
  --delta "${DELTA:-0.05}" \
  --batch-size "${BATCH_SIZE:-20}"
