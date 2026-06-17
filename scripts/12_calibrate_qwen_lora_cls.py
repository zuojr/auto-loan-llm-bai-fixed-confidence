#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.qwen_lora_cls_utils import apply_platt_calibration, fit_platt_calibrator, merge_calibrated_predictions


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Qwen yes/no LoRA raw probabilities and build replay proxy.")
    ap.add_argument("--val-pred", default=str(ROOT / "predictions/qwen_lora_cls_val_predictions.csv"))
    ap.add_argument("--raw-pred", default=str(ROOT / "predictions/qwen_lora_cls_raw_predictions.csv"))
    ap.add_argument("--paired", default=str(ROOT / "processed/paired_policy_eval_pool.csv"))
    ap.add_argument("--method", default="qwen_lora_cls_calibrated")
    ap.add_argument("--out-pred", default=str(ROOT / "predictions/qwen_lora_cls_calibrated_predictions.csv"))
    ap.add_argument("--out-paired", default=None)
    ap.add_argument("--out-calibrator", default=str(ROOT / "results/qwen_lora_cls_platt_calibrator.json"))
    args = ap.parse_args()

    val = pd.read_csv(_resolve(args.val_pred))
    raw = pd.read_csv(_resolve(args.raw_pred))
    paired = pd.read_csv(_resolve(args.paired))

    calibrator = fit_platt_calibrator(val["raw_pred_prob"], val["y"].astype(int))
    raw = raw.copy()
    raw["calibrated_pred_prob"] = apply_platt_calibration(calibrator, raw["raw_pred_prob"])

    out_pred = _resolve(args.out_pred)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_pred, index=False)

    out_paired = _resolve(args.out_paired) if args.out_paired else ROOT / "processed" / f"paired_policy_eval_pool_with_{args.method}.csv"
    merged = merge_calibrated_predictions(paired, raw, method=args.method)
    out_paired.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_paired, index=False)

    out_cal = _resolve(args.out_calibrator)
    out_cal.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": args.method,
        "calibrator": calibrator,
        "n_val": int(len(val)),
        "val_base_rate": float(val["y"].mean()),
        "raw_pred_mean": float(raw["raw_pred_prob"].mean()),
        "calibrated_pred_mean": float(raw["calibrated_pred_prob"].mean()),
    }
    out_cal.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out_pred}")
    print(f"Wrote {out_paired}")
    print(f"Wrote {out_cal}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
