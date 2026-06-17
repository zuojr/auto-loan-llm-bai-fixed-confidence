#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="baseline,qwen_local_icl,qwen_lora,gpt_api")
    args = ap.parse_args()
    rows = []
    for method in [m.strip() for m in args.methods.split(',') if m.strip()]:
        p = ROOT / "results" / f"fixed_confidence_summary_{method}.csv"
        if p.exists():
            df = pd.read_csv(p)
            # Main row: baseline for no-proxy, lcb_cv for proxy methods.
            if method == "baseline":
                keep = df[df.estimator == "baseline"].copy()
            else:
                keep = df[df.estimator == "lcb_cv"].copy()
            rows.append(keep)
    if not rows:
        raise FileNotFoundError("No fixed-confidence summary files found.")
    out = pd.concat(rows, ignore_index=True)
    base = out.loc[out.method == "baseline", "mean_stop_pulls"]
    baseline_mean = float(base.iloc[0]) if len(base) else None
    if baseline_mean is not None:
        out["sample_saving_vs_baseline"] = 1.0 - out["mean_stop_pulls"] / baseline_mean
    out_path = ROOT / "results" / "fixed_confidence_final_comparison_table.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
