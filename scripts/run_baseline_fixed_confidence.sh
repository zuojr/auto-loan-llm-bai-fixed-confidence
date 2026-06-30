#!/usr/bin/env bash
set -euo pipefail
python scripts/19_run_probe_replay.py \
  --method baseline \
  --reps ${REPS:-3000} \
  --delta ${DELTA:-0.05} \
  --kappa ${KAPPA:-1.0} \
  --max-pulls ${MAX_PULLS:-1000000}
