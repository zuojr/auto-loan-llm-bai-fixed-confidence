#!/usr/bin/env bash
set -euo pipefail
mkdir -p predictions
python scripts/07_train_qwen_lora_autodl.py \
  --train-jsonl qwen_finetune/train_full.jsonl \
  --eval-jsonl qwen_finetune/val_full.jsonl \
  --model-name ${QWEN_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct} \
  --out-dir ${QWEN_LORA_DIR:-qwen_finetune/qwen_lora_auto_loan} \
  --epochs ${EPOCHS:-1} \
  --batch-size ${TRAIN_BATCH_SIZE:-1} \
  --grad-accum ${GRAD_ACCUM:-16}
python scripts/08_predict_qwen_lora_autodl.py \
  --base-model ${QWEN_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct} \
  --adapter-dir ${QWEN_LORA_DIR:-qwen_finetune/qwen_lora_auto_loan} \
  --scoring api_scoring/scoring_full_paired.csv \
  --out predictions/qwen_lora_predictions.csv \
  --batch-size ${PRED_BATCH_SIZE:-4}
python scripts/04_merge_predictions.py --pred predictions/qwen_lora_predictions.csv --method qwen_lora
python scripts/05_run_fixed_confidence_replay.py \
  --input processed/paired_policy_eval_pool_with_qwen_lora.csv \
  --proxy-col proxy_qwen_lora \
  --method qwen_lora \
  --reps ${REPS:-3000} \
  --delta ${DELTA:-0.05} \
  --batch-size ${BATCH_SIZE:-20}
