#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]

METHODS = [
    ("baseline", "Reward-only PROBE", "#222222", "-"),
    ("qwen_local_icl", "Qwen 7B ICL", "#1f77b4", "-"),
    ("gpt_api", "GPT API ICL", "#9467bd", "-"),
    ("qwen_lora_cls_raw", "Qwen LoRA raw CLS", "#ff7f0e", "-"),
    ("tabpfn", "TabPFN structured", "#d62728", "-"),
]


def thousands(x: float, _pos: int) -> str:
    return f"{x / 1000:.0f}k"


def main() -> None:
    comparison_path = ROOT / "results" / "probe_comparison_table.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing {comparison_path}; run scripts/20_collect_probe_results.py first.")

    comparison = pd.read_csv(comparison_path).set_index("method")

    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)

    for method, label, color, linestyle in METHODS:
        reps_path = ROOT / "results" / f"probe_repetitions_{method}.csv"
        if not reps_path.exists():
            raise FileNotFoundError(
                f"Missing {reps_path}; rerun scripts/19_run_probe_replay.py for {method} first."
            )
        reps = pd.read_csv(reps_path)
        vals = reps["stop_pulls"].sort_values().to_numpy()
        y = [(i + 1) / len(vals) for i in range(len(vals))]

        saving = comparison.loc[method, "sample_saving_vs_baseline"] * 100.0
        mean_stop = comparison.loc[method, "mean_stop_pulls"] / 1000.0
        legend = f"{label}: {mean_stop:.1f}k, {saving:.1f}%"
        ax.step(vals, y, where="post", label=legend, color=color, linestyle=linestyle, linewidth=2.8)

        median = comparison.loc[method, "median_stop_pulls"]
        ax.axvline(median, color=color, alpha=0.16, linewidth=1.8)

    ax.set_title("PROBE fixed-confidence stopping-time CDF", fontsize=15, pad=12)
    ax.set_xlabel("Stopping time: total online pulls", fontsize=12)
    ax.set_ylabel("Empirical CDF across 3000 repetitions", fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylim(-0.015, 1.015)
    ax.grid(True, which="major", alpha=0.26)
    ax.legend(
        title="Method: mean stop, saving",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.96,
        fontsize=9.5,
        title_fontsize=10.0,
        borderaxespad=0.0,
    )

    out = ROOT / "figures" / "probe_stopping_cdf_all_methods.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
