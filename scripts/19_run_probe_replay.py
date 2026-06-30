#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.probe_replay import infer_reward_scale, simulate_probe, summarize_probe


def parse_reward_scale(value: str, paired: pd.DataFrame) -> float | None:
    if value.lower() == "auto":
        return infer_reward_scale(paired)
    parsed = float(value)
    if parsed <= 0:
        raise ValueError("--reward-scale must be positive or 'auto'.")
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PDF Algorithm 1 PROBE replay.")
    ap.add_argument("--input", default=str(ROOT / "processed/paired_policy_eval_pool.csv"))
    ap.add_argument("--proxy-col", default=None, help="Proxy reward column. Omit for reward-only PROBE baseline.")
    ap.add_argument("--method", default="baseline", help="Method name for output files.")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--max-pulls", type=int, default=1_000_000)
    ap.add_argument("--reward-scale", default="auto", help="Positive scale or 'auto' for pooled replay reward SD.")
    args = ap.parse_args()

    paired = pd.read_csv(args.input)
    if args.proxy_col and args.proxy_col not in paired.columns:
        raise ValueError(f"Proxy column {args.proxy_col!r} not found in {args.input}")
    reward_scale = parse_reward_scale(args.reward_scale, paired)

    reps_df, params_df = simulate_probe(
        paired,
        proxy_col=args.proxy_col,
        reps=args.reps,
        delta=args.delta,
        kappa=args.kappa,
        seed=args.seed,
        max_pulls=args.max_pulls,
        reward_scale=reward_scale,
    )
    summary = summarize_probe(reps_df, args.method, params_df)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    reps_path = ROOT / "results" / f"probe_repetitions_{args.method}.csv"
    summary_path = ROOT / "results" / f"probe_summary_{args.method}.csv"
    params_path = ROOT / "results" / f"probe_arm_params_{args.method}.csv"
    reps_df.to_csv(reps_path, index=False)
    summary.to_csv(summary_path, index=False)
    params_df.to_csv(params_path, index=False)

    vals = reps_df["stop_pulls"].sort_values().to_numpy()
    y = [(i + 1) / len(vals) for i in range(len(vals))]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.step(vals, y, where="post", label="PROBE")
    ax.set_xlabel("Stopping time: total online pulls")
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = ROOT / "figures" / f"probe_stopping_cdf_{args.method}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(summary.to_string(index=False))
    print(params_df.to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
