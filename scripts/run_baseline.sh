#!/usr/bin/env bash
set -euo pipefail
python scripts/01_prepare_replay_environment.py
python scripts/05_run_bai_replay.py --method baseline --reps 3000
