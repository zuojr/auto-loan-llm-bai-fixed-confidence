#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect PDF Algorithm 1 PROBE replay summaries.")
    ap.add_argument(
        "--methods",
        default="baseline,qwen_local_icl,qwen_lora_cls_raw,gpt_api,tabpfn",
        help="Comma-separated method names with probe_summary_<method>.csv files.",
    )
    ap.add_argument("--out", default=str(ROOT / "results/probe_comparison_table.csv"))
    args = ap.parse_args()

    rows = []
    for method in [x.strip() for x in args.methods.split(",") if x.strip()]:
        path = ROOT / "results" / f"probe_summary_{method}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if len(df) != 1:
            raise ValueError(f"Expected one row in {path}, got {len(df)}")
        rows.append(df.iloc[0].to_dict())

    out = pd.DataFrame(rows)
    baseline = out.loc[out["method"] == "baseline", "mean_stop_pulls"]
    if not baseline.empty:
        baseline_mean = float(baseline.iloc[0])
        out["sample_saving_vs_baseline"] = 1.0 - out["mean_stop_pulls"].astype(float) / baseline_mean
    out["abs_rho_min"] = out[["abs_rho_arm0", "abs_rho_arm1"]].min(axis=1)
    out["abs_rho_max"] = out[["abs_rho_arm0", "abs_rho_arm1"]].max(axis=1)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
