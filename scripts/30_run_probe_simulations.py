#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.probe_simulations import (  # noqa: E402
    correlation_certificate_diagnostics,
    run_gaussian_benchmark,
    run_ml_proxy_simulation,
)


METHOD_LABELS = {
    "reward_only_probe": "Reward-only PROBE",
    "proxy_probe": "PROBE",
    "oracle_residual_probe": "Oracle residual",
}


def plot_gaussian(gaussian, figures_dir: Path) -> None:
    colors = {
        "reward_only_probe": "#222222",
        "proxy_probe": "#1f77b4",
        "oracle_residual_probe": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, group in gaussian.groupby("method", sort=False):
        ax.plot(
            group["rho"],
            group["mean_stop_pulls"] / 1000.0,
            marker="o",
            linewidth=2.5,
            label=METHOD_LABELS.get(method, method),
            color=colors.get(method),
        )
    ax.set_xlabel("Reward-proxy correlation rho")
    ax.set_ylabel("Mean stopping time (thousand pulls)")
    ax.set_title("Gaussian benchmark under PROBE")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "simulation_probe_gaussian_samples.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, group in gaussian.groupby("method", sort=False):
        if method == "reward_only_probe":
            continue
        ax.plot(
            group["rho"],
            group["sample_ratio_vs_reward_only"],
            marker="o",
            linewidth=2.5,
            label=METHOD_LABELS.get(method, method),
            color=colors.get(method),
        )
    ref = gaussian[gaussian["method"] == "reward_only_probe"]
    ax.plot(
        ref["rho"],
        ref["oracle_residual_factor"],
        linestyle="--",
        linewidth=2.0,
        color="#555555",
        label="1 - rho^2",
    )
    ax.axhline(1.0, color="#999999", linewidth=1.0)
    ax.set_xlabel("Reward-proxy correlation rho")
    ax.set_ylabel("Ratio to reward-only PROBE")
    ax.set_title("Gaussian benchmark sample ratios")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "simulation_probe_gaussian_ratio.png", dpi=220)
    plt.close(fig)


def plot_diagnostics(diag, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = 1.0 / (diag["rho"] ** 2)
    ax.scatter(x, diag["mean_certification_time"], s=52, color="#1f77b4")
    coef = np.polyfit(x, diag["mean_certification_time"], deg=1)
    grid = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(grid, coef[0] * grid + coef[1], color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("1 / rho^2")
    ax.set_ylabel("Mean samples to certify half variance reduction")
    ax.set_title("PROBE certificate learning time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "simulation_probe_learning_time.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        diag["rho"],
        diag["plugin_failure_rate"],
        marker="o",
        linewidth=2.5,
        color="#ff7f0e",
        label="Plug-in residual variance",
    )
    ax.plot(
        diag["rho"],
        diag["certificate_failure_rate"],
        marker="o",
        linewidth=2.5,
        color="#2ca02c",
        label="PROBE upper certificate",
    )
    ax.set_xlabel("Reward-proxy correlation rho")
    ax.set_ylabel("Undercoverage probability")
    ax.set_title("Residual variance undercoverage at t=100")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "simulation_probe_certificate_failure.png", dpi=220)
    plt.close(fig)


def plot_ml_proxy(ml_proxy, figures_dir: Path) -> None:
    label_map = {
        "linear": "Linear",
        "decision_tree": "Decision tree",
        "gradient_boosting": "Gradient boosting",
    }
    labels = [label_map[m] for m in ml_proxy["proxy_model"]]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - 0.18, ml_proxy["mean_corr"], width=0.36, label="Mean corr.", color="#1f77b4")
    ax.bar(x + 0.18, ml_proxy["min_corr"], width=0.36, label="Min corr.", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Armwise reward-proxy correlation")
    ax.set_title("Fitted-model proxy alignment")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "simulation_probe_ml_proxy_alignment.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x, ml_proxy["probe_ratio_vs_reward_only"], color="#d62728")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Ratio to reward-only PROBE")
    ax.set_title("Fitted-model proxy PROBE ratios")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "simulation_probe_ml_proxy_ratios.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run synthetic simulations with PDF Algorithm 1 PROBE.")
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--max-pulls", type=int, default=2_000_000)
    ap.add_argument("--gaussian-reps", type=int, default=120)
    ap.add_argument("--gaussian-pool-size", type=int, default=200_000)
    ap.add_argument("--diagnostic-learning-reps", type=int, default=500)
    ap.add_argument("--diagnostic-failure-reps", type=int, default=15_000)
    ap.add_argument("--ml-reps", type=int, default=120)
    ap.add_argument("--ml-train-size", type=int, default=12_000)
    ap.add_argument("--ml-pool-size", type=int, default=40_000)
    args = ap.parse_args()

    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    gaussian = run_gaussian_benchmark(
        reps=args.gaussian_reps,
        pool_size=args.gaussian_pool_size,
        delta=args.delta,
        kappa=args.kappa,
        seed=args.seed,
        max_pulls=args.max_pulls,
    )
    gaussian.to_csv(results_dir / "simulation_probe_gaussian_benchmark.csv", index=False)
    plot_gaussian(gaussian, figures_dir)

    diagnostics = correlation_certificate_diagnostics(
        reps_learning=args.diagnostic_learning_reps,
        reps_failure=args.diagnostic_failure_reps,
        seed=args.seed + 1,
    )
    diagnostics.to_csv(results_dir / "simulation_probe_correlation_diagnostics.csv", index=False)
    plot_diagnostics(diagnostics, figures_dir)

    ml_proxy = run_ml_proxy_simulation(
        reps=args.ml_reps,
        train_size=args.ml_train_size,
        pool_size=args.ml_pool_size,
        delta=args.delta,
        kappa=args.kappa,
        seed=args.seed + 2,
        max_pulls=args.max_pulls,
    )
    ml_proxy.to_csv(results_dir / "simulation_probe_ml_proxy.csv", index=False)
    plot_ml_proxy(ml_proxy, figures_dir)

    print("Gaussian benchmark:")
    print(gaussian.to_string(index=False))
    print("\nCorrelation diagnostics:")
    print(diagnostics.to_string(index=False))
    print("\nML proxy simulation:")
    print(ml_proxy.to_string(index=False))


if __name__ == "__main__":
    main()
