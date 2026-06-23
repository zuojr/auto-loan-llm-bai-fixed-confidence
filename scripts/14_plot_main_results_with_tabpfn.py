#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

METHOD_LABELS = {
    "baseline": "Reward-only\nbaseline",
    "qwen_local_icl": "Qwen 2.5 7B\nICL",
    "qwen_lora_cls_raw": "Qwen 2.5 7B\nLoRA raw CLS",
    "gpt_api": "GPT API\nICL",
    "tabpfn": "TabPFN\nstructured ref.",
}

COLORS = {
    "baseline": "#6B7280",
    "qwen_local_icl": "#4F46E5",
    "qwen_lora_cls_raw": "#059669",
    "gpt_api": "#2563EB",
    "tabpfn": "#DC2626",
}


def main() -> None:
    table = ROOT / "results" / "fixed_confidence_paper_table_abs_rho_with_tabpfn.csv"
    out = ROOT / "figures" / "fixed_confidence_mean_stop_with_tabpfn.png"
    df = pd.read_csv(table)
    methods = ["baseline", "qwen_local_icl", "qwen_lora_cls_raw", "gpt_api", "tabpfn"]
    df = df.set_index("method").loc[methods].reset_index()

    x = np.arange(len(df))
    mean = df["mean_stop_pulls"].to_numpy(float)
    median = df["median_stop_pulls"].to_numpy(float)
    q90 = df["q90_stop_pulls"].to_numpy(float)
    savings = df["sample_saving_pct"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(
        x,
        mean,
        color=[COLORS[m] for m in df["method"]],
        width=0.72,
        alpha=0.9,
        label="Mean stopping time",
    )
    ax.scatter(x, median, color="white", edgecolor="#111827", zorder=3, s=46, label="Median")
    ax.scatter(x, q90, color="#111827", marker="_", zorder=3, s=280, linewidths=2.2, label="90th percentile")

    for i, (bar, saving) in enumerate(zip(bars, savings)):
        label = "baseline" if i == 0 else f"{saving:.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 160,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in df["method"]], fontsize=9)
    ax.set_ylabel("Fixed-confidence stopping time (online pulls)")
    ax.set_title("Stopping-time comparison with structured TabPFN reference")
    ax.set_ylim(0, max(q90) * 1.12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.01,
        0.98,
        "Labels above bars show sample saving vs. reward-only baseline.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
