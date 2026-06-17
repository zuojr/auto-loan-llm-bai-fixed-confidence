from __future__ import annotations

import numpy as np
import pandas as pd


class ArmReplayData:
    """Vectorized replay data for one pricing rule / arm."""

    def __init__(self, g: pd.DataFrame, proxy_col: str | None = None):
        self.arm = int(g["arm"].iloc[0])
        self.prob = g["prob_accept"].to_numpy(float)
        self.offer = g["offer_mp"].to_numpy(float)
        self.n_pool = len(g)
        self.expected_reward = self.offer * self.prob
        self.mu = float(np.mean(self.expected_reward))
        self.varx = float(np.mean((self.offer ** 2) * self.prob) - self.mu ** 2)
        self.varx = max(self.varx, 1e-12)
        self.sx = float(np.sqrt(self.varx))
        if proxy_col is not None and proxy_col in g.columns:
            self.y = g[proxy_col].to_numpy(float)
            self.my = float(np.mean(self.y))
            self.vary = float(np.var(self.y))
            self.vary = max(self.vary, 1e-12)
            self.sy = float(np.sqrt(self.vary))
            self.cov = float(np.mean((self.expected_reward - self.mu) * (self.y - self.my)))
            self.signed_rho = float(np.clip(self.cov / (self.sx * self.sy), -0.999, 0.999))
            self.abs_rho = float(abs(self.signed_rho))
            self.rho = float(np.clip(self.signed_rho, 0.0, 0.999))
            self.beta_oracle = float(self.cov / self.vary)
        else:
            self.y = None
            self.my = 0.0
            self.vary = np.nan
            self.sy = 0.0
            self.cov = np.nan
            self.signed_rho = 0.0
            self.abs_rho = 0.0
            self.rho = 0.0
            self.beta_oracle = 0.0


def build_arms(paired: pd.DataFrame, proxy_col: str | None = None) -> list[ArmReplayData]:
    arms = []
    for arm in sorted(paired["arm"].unique()):
        arms.append(ArmReplayData(paired.loc[paired["arm"] == arm].reset_index(drop=True), proxy_col))
    if len(arms) != 2:
        raise ValueError("This fixed-confidence replay script is currently specialized to the two-rule auto-loan experiment.")
    return arms


def arm_parameter_frame(arms: list[ArmReplayData]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "arm": a.arm,
            "mu_reward": a.mu,
            "sd_reward": a.sx,
            "proxy_mean": a.my,
            "proxy_sd": a.sy,
            "reward_proxy_cov": a.cov,
            "reward_proxy_rho": a.rho,
            "reward_proxy_signed_rho": a.signed_rho,
            "reward_proxy_abs_rho": a.abs_rho,
            "oracle_beta": a.beta_oracle,
        }
        for a in arms
    ])


def _sample_batch(a: ArmReplayData, rng: np.random.Generator, batch_size: int):
    idx = rng.integers(0, a.n_pool, size=batch_size)
    prob = a.prob[idx]
    offer = a.offer[idx]
    x = offer * rng.binomial(1, prob)
    if a.y is None:
        y = np.zeros(batch_size)
    else:
        y = a.y[idx]
    return x, y


def run_fixed_confidence_once(
    arms: list[ArmReplayData],
    method: str = "baseline",
    delta: float = 0.05,
    seed: int | None = None,
    max_pulls: int = 50000,
    batch_size: int = 20,
    rad_const: float = 1.5,
):
    """Run a two-arm fixed-confidence replay until Gaussian CIs separate.

    method options:
      - baseline: reward-only Gaussian CI stopping.
      - oracle_cv: control variate with pool-calibrated oracle beta, diagnostic only.
      - lcb_cv: adaptive one-sided lower-bound control variate.
      - signed_oracle_cv: oracle CV that allows negative beta and uses signed rho^2.
      - signed_lcb_cv: adaptive CV using a lower bound on abs(signed rho).

    The stopping rule is a practical Gaussian fixed-confidence diagnostic for replay:
    stop when the lower confidence bound of the currently estimated best rule exceeds
    the upper confidence bound of the other rule. The confidence radius uses a simple
    time-uniform union-bound schedule over checking times.
    """
    rng = np.random.default_rng(seed)
    n_arms = len(arms)
    true_best = int(np.argmax([a.mu for a in arms]))

    sum_z = np.zeros(n_arms)
    n = np.zeros(n_arms, dtype=int)
    sum_var_bound = np.zeros(n_arms)
    # Statistics for the LCB correlation estimator.
    sum_xstd_ystd = np.zeros(n_arms)
    L = np.zeros(n_arms)
    signed_L = np.zeros(n_arms)

    pulls = 0
    checks = 0
    truncated = False

    while pulls < max_pulls:
        checks += 1
        for ai, a in enumerate(arms):
            remaining = max_pulls - pulls
            if remaining <= 0:
                break
            b = min(batch_size, remaining)

            if method == "baseline" or a.y is None:
                beta = 0.0
                L_used = 0.0
                var_obs_bound = a.varx
            elif method == "oracle_cv":
                beta = a.beta_oracle
                L_used = a.rho
                var_obs_bound = a.varx * (1.0 - a.rho ** 2)
            elif method == "signed_oracle_cv":
                beta = a.beta_oracle
                L_used = a.signed_rho
                var_obs_bound = a.varx * (1.0 - a.signed_rho ** 2)
            elif method == "lcb_cv":
                beta = L[ai] * a.sx / a.sy if a.sy > 1e-12 else 0.0
                L_used = L[ai]
                var_obs_bound = a.varx * (1.0 - L_used ** 2)
            elif method == "signed_lcb_cv":
                beta = signed_L[ai] * a.sx / a.sy if a.sy > 1e-12 else 0.0
                L_used = signed_L[ai]
                var_obs_bound = a.varx * (1.0 - L_used ** 2)
            else:
                raise ValueError(f"Unknown method: {method}")

            x, y = _sample_batch(a, rng, b)
            z = x - beta * (y - a.my)
            sum_z[ai] += float(np.sum(z))
            n[ai] += b
            sum_var_bound[ai] += b * var_obs_bound
            pulls += b

            if method == "lcb_cv" and a.y is not None and a.sy > 1e-12 and a.sx > 0:
                sum_xstd_ystd[ai] += float(np.sum(((x - a.mu) / a.sx) * ((y - a.my) / a.sy)))
                nn = max(n[ai], 1)
                # A simple summable confidence schedule over arms and checks.
                eta = delta / (8.0 * n_arms * max(checks, 1) ** 2)
                rad = rad_const * (np.sqrt(np.log(2.0 / eta) / nn) + np.log(2.0 / eta) / nn)
                rho_hat = sum_xstd_ystd[ai] / nn
                L[ai] = max(L[ai], float(np.clip(rho_hat - rad, 0.0, 0.999)))
            elif method == "signed_lcb_cv" and a.y is not None and a.sy > 1e-12 and a.sx > 0:
                sum_xstd_ystd[ai] += float(np.sum(((x - a.mu) / a.sx) * ((y - a.my) / a.sy)))
                nn = max(n[ai], 1)
                # Same summable schedule, but applied to the magnitude of signed correlation.
                eta = delta / (8.0 * n_arms * max(checks, 1) ** 2)
                rad = rad_const * (np.sqrt(np.log(2.0 / eta) / nn) + np.log(2.0 / eta) / nn)
                rho_hat = sum_xstd_ystd[ai] / nn
                mag_lcb = float(np.clip(abs(rho_hat) - rad, 0.0, 0.999))
                candidate = float(np.sign(rho_hat) * mag_lcb)
                if abs(candidate) > abs(signed_L[ai]):
                    signed_L[ai] = candidate

        if np.min(n) < 2:
            continue
        means = sum_z / n
        # Time-uniform Gaussian confidence radius for the mean of each transformed arm.
        alpha = delta / (8.0 * n_arms * max(checks, 1) ** 2)
        logterm = np.log(2.0 / alpha)
        halfwidth = np.zeros(n_arms)
        for ai in range(n_arms):
            variance_of_mean_bound = sum_var_bound[ai] / (n[ai] ** 2)
            halfwidth[ai] = np.sqrt(2.0 * variance_of_mean_bound * logterm)

        best_hat = int(np.argmax(means))
        other = 1 - best_hat
        if means[best_hat] - halfwidth[best_hat] > means[other] + halfwidth[other]:
            return {
                "stop_pulls": int(pulls),
                "recommendation": best_hat,
                "correct": int(best_hat == true_best),
                "n_arm0": int(n[0]),
                "n_arm1": int(n[1]),
                "final_L_arm0": float(L[0]),
                "final_L_arm1": float(L[1]),
                "final_signed_L_arm0": float(signed_L[0]),
                "final_signed_L_arm1": float(signed_L[1]),
                "truncated": 0,
            }

    truncated = True
    means = sum_z / np.maximum(n, 1)
    best_hat = int(np.argmax(means))
    return {
        "stop_pulls": int(pulls),
        "recommendation": best_hat,
        "correct": int(best_hat == true_best),
        "n_arm0": int(n[0]),
        "n_arm1": int(n[1]),
        "final_L_arm0": float(L[0]),
        "final_L_arm1": float(L[1]),
        "final_signed_L_arm0": float(signed_L[0]),
        "final_signed_L_arm1": float(signed_L[1]),
        "truncated": int(truncated),
    }


def simulate_fixed_confidence(
    paired: pd.DataFrame,
    proxy_col: str | None = None,
    estimators: list[str] | None = None,
    reps: int = 1000,
    delta: float = 0.05,
    seed: int = 20260610,
    max_pulls: int = 50000,
    batch_size: int = 20,
    rad_const: float = 1.5,
):
    arms = build_arms(paired, proxy_col)
    if estimators is None:
        estimators = ["baseline"] if proxy_col is None else ["baseline", "oracle_cv", "lcb_cv"]
    rows = []
    for method in estimators:
        rng = np.random.default_rng(seed + (abs(hash(method)) % 100000))
        for rep in range(reps):
            row = run_fixed_confidence_once(
                arms,
                method=method,
                delta=delta,
                seed=int(rng.integers(0, 2**31 - 1)),
                max_pulls=max_pulls,
                batch_size=batch_size,
                rad_const=rad_const,
            )
            row.update({"estimator": method, "rep": rep, "delta": delta})
            rows.append(row)
    return pd.DataFrame(rows), arm_parameter_frame(arms)


def summarize_fixed_confidence(reps_df: pd.DataFrame, method_name: str, params_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for est, g in reps_df.groupby("estimator"):
        rows.append({
            "method": method_name,
            "estimator": est,
            "delta": float(g["delta"].iloc[0]) if "delta" in g else np.nan,
            "mean_stop_pulls": float(g["stop_pulls"].mean()),
            "median_stop_pulls": float(g["stop_pulls"].median()),
            "q90_stop_pulls": float(g["stop_pulls"].quantile(0.90)),
            "empirical_correct_at_stop": float(g["correct"].mean()),
            "truncated_rate": float(g["truncated"].mean()),
            "mean_final_L_arm0": float(g["final_L_arm0"].mean()),
            "mean_final_L_arm1": float(g["final_L_arm1"].mean()),
            "mean_final_signed_L_arm0": float(g["final_signed_L_arm0"].mean()) if "final_signed_L_arm0" in g else 0.0,
            "mean_final_signed_L_arm1": float(g["final_signed_L_arm1"].mean()) if "final_signed_L_arm1" in g else 0.0,
            "rho_arm0": float(params_df.loc[params_df.arm == 0, "reward_proxy_rho"].iloc[0]) if "reward_proxy_rho" in params_df else np.nan,
            "rho_arm1": float(params_df.loc[params_df.arm == 1, "reward_proxy_rho"].iloc[0]) if "reward_proxy_rho" in params_df else np.nan,
            "signed_rho_arm0": float(params_df.loc[params_df.arm == 0, "reward_proxy_signed_rho"].iloc[0]) if "reward_proxy_signed_rho" in params_df else np.nan,
            "signed_rho_arm1": float(params_df.loc[params_df.arm == 1, "reward_proxy_signed_rho"].iloc[0]) if "reward_proxy_signed_rho" in params_df else np.nan,
            "abs_rho_arm0": float(params_df.loc[params_df.arm == 0, "reward_proxy_abs_rho"].iloc[0]) if "reward_proxy_abs_rho" in params_df else np.nan,
            "abs_rho_arm1": float(params_df.loc[params_df.arm == 1, "reward_proxy_abs_rho"].iloc[0]) if "reward_proxy_abs_rho" in params_df else np.nan,
        })
    return pd.DataFrame(rows)
