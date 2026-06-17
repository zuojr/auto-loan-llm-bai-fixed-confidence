#!/usr/bin/env bash
set -euo pipefail
python scripts/05_run_fixed_confidence_replay.py \
  --method baseline \
  --reps ${REPS:-3000} \
  --delta ${DELTA:-0.05} \
  --batch-size ${BATCH_SIZE:-20}
