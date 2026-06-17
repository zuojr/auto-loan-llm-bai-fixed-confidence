#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.qwen_lora_cls_utils import merge_calibrated_predictions


def main() -> None:
    ap = argparse.ArgumentParser(description="Build replay proxy from raw yes/no-logit Qwen LoRA probabilities.")
    ap.add_argument("--raw-pred", default=str(ROOT / "predictions/qwen_lora_cls_raw_predictions.csv"))
    ap.add_argument("--paired", default=str(ROOT / "processed/paired_policy_eval_pool.csv"))
    ap.add_argument("--method", default="qwen_lora_cls_raw")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paired = pd.read_csv(args.paired)
    pred = pd.read_csv(args.raw_pred)
    merged = merge_calibrated_predictions(
        paired,
        pred,
        method=args.method,
        pred_col="raw_pred_prob",
    )
    out = Path(args.out) if args.out else ROOT / "processed" / f"paired_policy_eval_pool_with_{args.method}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(f"Proxy column: proxy_{args.method}")


if __name__ == "__main__":
    main()
