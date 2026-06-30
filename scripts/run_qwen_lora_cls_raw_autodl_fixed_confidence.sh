#!/usr/bin/env bash
set -euo pipefail

mkdir -p predictions results processed qwen_finetune

python scripts/10_make_qwen_lora_cls_data.py \
  --train-in qwen_finetune/train_full.jsonl \
  --val-in qwen_finetune/val_full.jsonl \
  --train-out ${QWEN_CLS_TRAIN_JSONL:-qwen_finetune/train_cls_full.jsonl} \
  --val-out ${QWEN_CLS_VAL_JSONL:-qwen_finetune/val_cls_full.jsonl}

python scripts/07_train_qwen_lora_autodl.py \
  --train-jsonl ${QWEN_CLS_TRAIN_JSONL:-qwen_finetune/train_cls_full.jsonl} \
  --eval-jsonl ${QWEN_CLS_VAL_JSONL:-qwen_finetune/val_cls_full.jsonl} \
  --model-name ${QWEN_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct} \
  --out-dir ${QWEN_CLS_LORA_DIR:-qwen_finetune/qwen_lora_cls_auto_loan} \
  --epochs ${EPOCHS:-1} \
  --batch-size ${TRAIN_BATCH_SIZE:-1} \
  --grad-accum ${GRAD_ACCUM:-16} \
  --lr ${LR:-2e-4}

python scripts/11_predict_qwen_lora_cls_logits_autodl.py \
  --base-model ${QWEN_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct} \
  --adapter-dir ${QWEN_CLS_LORA_DIR:-qwen_finetune/qwen_lora_cls_auto_loan} \
  --scoring api_scoring/scoring_full_paired.csv \
  --out predictions/qwen_lora_cls_raw_predictions.csv \
  --batch-size ${CLS_PRED_BATCH_SIZE:-16}

python scripts/13_build_qwen_lora_cls_raw_proxy.py \
  --raw-pred predictions/qwen_lora_cls_raw_predictions.csv \
  --method qwen_lora_cls_raw

python scripts/19_run_probe_replay.py \
  --input processed/paired_policy_eval_pool_with_qwen_lora_cls_raw.csv \
  --proxy-col proxy_qwen_lora_cls_raw \
  --method qwen_lora_cls_raw \
  --reps ${REPS:-3000} \
  --delta ${DELTA:-0.05} \
  --kappa ${KAPPA:-1.0} \
  --max-pulls ${MAX_PULLS:-1000000}
