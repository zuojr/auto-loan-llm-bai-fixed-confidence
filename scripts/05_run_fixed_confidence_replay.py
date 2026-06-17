#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.fixed_confidence_replay import simulate_fixed_confidence, summarize_fixed_confidence


def main():
    ap = argparse.ArgumentParser(description="Run fixed-confidence two-rule BAI replay.")
    ap.add_argument("--input", default=str(ROOT / "processed/paired_policy_eval_pool.csv"))
    ap.add_argument("--proxy-col", default=None, help="Proxy revenue column. Omit for reward-only baseline.")
    ap.add_argument("--method", default="baseline", help="Method name for output files, e.g. qwen_api, qwen_ft, gpt_api.")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-pulls", type=int, default=50000)
    ap.add_argument("--estimators", default=None, help="Comma-separated estimators. Default: baseline or baseline,oracle_cv,lcb_cv.")
    args = ap.parse_args()

    paired = pd.read_csv(args.input)
    if args.proxy_col and args.proxy_col not in paired.columns:
        raise ValueError(f"Proxy column {args.proxy_col!r} not found in {args.input}")
    estimators = None if args.estimators is None else [x.strip() for x in args.estimators.split(',') if x.strip()]

    reps_df, params_df = simulate_fixed_confidence(
        paired,
        proxy_col=args.proxy_col,
        estimators=estimators,
        reps=args.reps,
        delta=args.delta,
        seed=args.seed,
        batch_size=args.batch_size,
        max_pulls=args.max_pulls,
    )
    summary = summarize_fixed_confidence(reps_df, args.method, params_df)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    reps_path = ROOT / "results" / f"fixed_confidence_repetitions_{args.method}.csv"
    summary_path = ROOT / "results" / f"fixed_confidence_summary_{args.method}.csv"
    params_path = ROOT / "results" / f"fixed_confidence_arm_params_{args.method}.csv"
    reps_df.to_csv(reps_path, index=False)
    summary.to_csv(summary_path, index=False)
    params_df.to_csv(params_path, index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    order = ["baseline", "lcb_cv", "oracle_cv"]
    labels = {"baseline": "Reward-only", "lcb_cv": "Gen-CV-LCB", "oracle_cv": "Oracle CV"}
    for est in order:
        if est in set(reps_df["estimator"]):
            vals = reps_df.loc[reps_df["estimator"] == est, "stop_pulls"].sort_values().to_numpy()
            y = [(i + 1) / len(vals) for i in range(len(vals))]
            ax.step(vals, y, where="post", label=labels.get(est, est))
    ax.set_xlabel("Stopping time: total online pulls")
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = ROOT / "figures" / f"fixed_confidence_stopping_cdf_{args.method}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(summary.to_string(index=False))
    print(params_df.to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
