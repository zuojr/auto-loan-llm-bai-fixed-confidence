#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SERIES = [
    ("baseline", "baseline", "Reward-only baseline", "#6B7280"),
    ("qwen_local_icl", "lcb_cv", "Qwen 2.5 7B ICL", "#4F46E5"),
    ("qwen_lora_cls_raw", "lcb_cv", "Qwen 2.5 7B LoRA raw CLS", "#059669"),
    ("gpt_api", "lcb_cv", "GPT API ICL", "#2563EB"),
    ("tabpfn", "lcb_cv", "TabPFN structured ref.", "#DC2626"),
]


def cdf_xy(values: pd.Series) -> tuple[list[float], list[float]]:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    y = [(i + 1) / n for i in range(n)]
    return vals, y


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for method, estimator, label, color in SERIES:
        path = ROOT / "results" / f"fixed_confidence_repetitions_{method}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing repetition trace: {path}")
        reps = pd.read_csv(path)
        vals = reps.loc[reps["estimator"] == estimator, "stop_pulls"]
        if vals.empty:
            raise ValueError(f"No rows for {method}/{estimator} in {path}")
        x, y = cdf_xy(vals)
        ax.step(x, y, where="post", label=label, color=color, linewidth=2.0)

    ax.set_xlabel("Fixed-confidence stopping time (online pulls)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Empirical CDF of stopping times with TabPFN structured reference")
    ax.grid(True, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = ROOT / "figures" / "fixed_confidence_stopping_cdf_with_tabpfn.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
