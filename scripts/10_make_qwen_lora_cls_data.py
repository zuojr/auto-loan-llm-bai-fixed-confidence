#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.qwen_lora_cls_utils import convert_numeric_jsonl_to_yes_no


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert numeric Qwen LoRA JSONL data to yes/no classification JSONL.")
    ap.add_argument("--train-in", default=str(ROOT / "qwen_finetune/train_full.jsonl"))
    ap.add_argument("--val-in", default=str(ROOT / "qwen_finetune/val_full.jsonl"))
    ap.add_argument("--train-out", default=str(ROOT / "qwen_finetune/train_cls_full.jsonl"))
    ap.add_argument("--val-out", default=str(ROOT / "qwen_finetune/val_cls_full.jsonl"))
    args = ap.parse_args()

    n_train = convert_numeric_jsonl_to_yes_no(args.train_in, args.train_out)
    n_val = convert_numeric_jsonl_to_yes_no(args.val_in, args.val_out)
    print(f"Wrote {n_train} rows to {args.train_out}")
    print(f"Wrote {n_val} rows to {args.val_out}")


if __name__ == "__main__":
    main()
