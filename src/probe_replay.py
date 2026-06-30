from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProbeArmData:
    arm: int
    prob: np.ndarray
    offer: np.ndarray
    proxy: np.ndarray | None
    proxy_mean: float
    expected_reward: np.ndarray
    reward_scale: float

    @property
    def n_pool(self) -> int:
        return int(len(self.prob))

    @property
    def mu_reward(self) -> float:
        return float(np.mean(self.expected_reward))

    @property
    def mu_scaled(self) -> float:
        return float(self.mu_reward / self.reward_scale)


def infer_reward_scale(paired: pd.DataFrame) -> float:
    prob = paired["prob_accept"].to_numpy(float)
    offer = paired["offer_mp"].to_numpy(float)
    expected = offer * prob
    second = (offer**2) * prob
    variance = float(np.mean(second) - np.mean(expected) ** 2)
    return max(math.sqrt(max(variance, 1e-12)), 1e-12)


def build_probe_arms(
    paired: pd.DataFrame,
    proxy_col: str | None = None,
    reward_scale: float | None = None,
) -> list[ProbeArmData]:
    if reward_scale is None:
        reward_scale = infer_reward_scale(paired)
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive.")

    arms = []
    for arm in sorted(paired["arm"].unique()):
        g = paired.loc[paired["arm"] == arm].reset_index(drop=True)
        proxy = None
        proxy_mean = 0.0
        if proxy_col is not None:
            if proxy_col not in g.columns:
                raise ValueError(f"Proxy column {proxy_col!r} not found.")
            proxy_values = g[proxy_col].to_numpy(float)
            if float(np.std(proxy_values)) > 1e-12:
                proxy = proxy_values
                proxy_mean = float(np.mean(proxy_values))
        prob = g["prob_accept"].to_numpy(float)
        offer = g["offer_mp"].to_numpy(float)
        arms.append(
            ProbeArmData(
                arm=int(arm),
                prob=prob,
                offer=offer,
                proxy=proxy,
                proxy_mean=proxy_mean,
                expected_reward=offer * prob,
                reward_scale=float(reward_scale),
            )
        )

    if len(arms) < 2:
        raise ValueError("PROBE requires at least two arms.")
    return arms


def calibration_size(n_arms: int, delta: float, kappa: float) -> int:
    if n_arms < 2:
        raise ValueError("n_arms must be at least 2.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if not 0 < kappa <= 1:
        raise ValueError("kappa must be in (0, 1].")
    return int(2 + math.ceil(64.0 * math.log(8.0 * n_arms / delta) / (kappa**2)))


def variance_upper_certificate(batch_size: int, residual_variance: float, eta: float) -> float:
    if batch_size < 3 or not 0 < eta < 1:
        return math.inf
    logterm = math.log(1.0 / eta)
    if batch_size - 2 <= 4.0 * logterm:
        return math.inf
    denom = 1.0 - 2.0 * math.sqrt(logterm / (batch_size - 2))
    if denom <= 0:
        return math.inf
    return max(float(residual_variance), 0.0) / denom


def fresh_batch_size(variance_certificate: float, delta_r: float, epsilon_r: float) -> int:
    if not np.isfinite(variance_certificate):
        return int(2**31 - 1)
    if not 0 < delta_r < 1:
        raise ValueError("delta_r must be in (0, 1).")
    if epsilon_r <= 0:
        raise ValueError("epsilon_r must be positive.")
    return int(
        max(
            3,
            math.ceil(1.0 + 16.0 * math.log(2.0 / delta_r)),
            math.ceil(8.0 * math.log(4.0 / delta_r)),
            math.ceil(64.0 * variance_certificate * math.log(2.0 / delta_r) * (epsilon_r**-2)),
        )
    )


def _sample_batch(arm: ProbeArmData, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray | None]:
    idx = rng.integers(0, arm.n_pool, size=size)
    reward = arm.offer[idx] * rng.binomial(1, arm.prob[idx])
    reward = reward.astype(float) / arm.reward_scale
    if arm.proxy is None:
        return reward, None
    return reward, arm.proxy[idx].astype(float) - arm.proxy_mean


def ols_intercept_slope_residual_variance(x: np.ndarray, w: np.ndarray | None) -> tuple[float, float, float]:
    n = int(len(x))
    if n < 3:
        raise ValueError("OLS requires a batch size of at least 3.")
    xbar = float(np.mean(x))
    if w is None:
        residual = x - xbar
        return xbar, 0.0, float(np.sum(residual**2) / max(n - 2, 1))

    wbar = float(np.mean(w))
    wc = w - wbar
    sww = float(np.sum(wc**2))
    if sww <= 1e-12:
        residual = x - xbar
        return xbar, 0.0, float(np.sum(residual**2) / max(n - 2, 1))

    slope = float(np.sum((x - xbar) * wc) / sww)
    intercept = float(xbar - slope * wbar)
    residual = x - intercept - slope * w
    return intercept, slope, float(np.sum(residual**2) / max(n - 2, 1))


def run_probe_once(
    arms: list[ProbeArmData],
    delta: float = 0.05,
    kappa: float = 1.0,
    seed: int | None = None,
    max_pulls: int = 1_000_000,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    n_arms = len(arms)
    true_best = int(np.argmax([arm.mu_reward for arm in arms]))
    s_cal = calibration_size(n_arms, delta, kappa)

    pulls = 0
    n_by_arm = np.zeros(n_arms, dtype=int)
    certificates = np.zeros(n_arms, dtype=float)
    last_intercepts = np.array([arm.mu_scaled for arm in arms], dtype=float)
    last_slopes = np.zeros(n_arms, dtype=float)

    for ai, arm in enumerate(arms):
        if pulls + s_cal > max_pulls:
            return {
                "stop_pulls": int(pulls),
                "recommendation": int(np.argmax(last_intercepts)),
                "correct": int(np.argmax(last_intercepts) == true_best),
                "rounds": 0,
                "truncated": 1,
                "s_cal": int(s_cal),
                "reward_scale": float(arm.reward_scale),
                "n_arm0": int(n_by_arm[0]) if n_arms > 0 else 0,
                "n_arm1": int(n_by_arm[1]) if n_arms > 1 else 0,
                "final_U_arm0": math.inf,
                "final_U_arm1": math.inf,
                "final_slope_arm0": 0.0,
                "final_slope_arm1": 0.0,
            }
        x, w = _sample_batch(arm, rng, s_cal)
        _, _, vhat = ols_intercept_slope_residual_variance(x, w)
        certificates[ai] = variance_upper_certificate(s_cal, vhat, delta / (8.0 * n_arms))
        pulls += s_cal
        n_by_arm[ai] += s_cal

    active = list(range(n_arms))
    rounds = 0
    truncated = False

    while len(active) > 1:
        rounds += 1
        delta_r = delta / (16.0 * n_arms * rounds**2)
        epsilon_r = 2.0 ** (-rounds)
        round_intercepts: dict[int, float] = {}

        for ai in list(active):
            t = fresh_batch_size(certificates[ai], delta_r, epsilon_r)
            if pulls + t > max_pulls:
                truncated = True
                break
            x, w = _sample_batch(arms[ai], rng, t)
            intercept, slope, vhat = ols_intercept_slope_residual_variance(x, w)
            cert = variance_upper_certificate(t, vhat, delta_r)
            certificates[ai] = min(certificates[ai], cert)
            last_intercepts[ai] = intercept
            last_slopes[ai] = slope
            round_intercepts[ai] = intercept
            pulls += t
            n_by_arm[ai] += t

        if truncated:
            break

        best = max(active, key=lambda idx: round_intercepts[idx])
        threshold = round_intercepts[best] - epsilon_r
        active = [idx for idx in active if idx == best or round_intercepts[idx] >= threshold]

    if active:
        recommendation = int(max(active, key=lambda idx: last_intercepts[idx]))
    else:
        recommendation = int(np.argmax(last_intercepts))

    return {
        "stop_pulls": int(pulls),
        "recommendation": recommendation,
        "correct": int(recommendation == true_best),
        "rounds": int(rounds),
        "truncated": int(truncated),
        "s_cal": int(s_cal),
        "reward_scale": float(arms[0].reward_scale),
        "n_arm0": int(n_by_arm[0]) if n_arms > 0 else 0,
        "n_arm1": int(n_by_arm[1]) if n_arms > 1 else 0,
        "final_U_arm0": float(certificates[0]) if n_arms > 0 else np.nan,
        "final_U_arm1": float(certificates[1]) if n_arms > 1 else np.nan,
        "final_slope_arm0": float(last_slopes[0]) if n_arms > 0 else np.nan,
        "final_slope_arm1": float(last_slopes[1]) if n_arms > 1 else np.nan,
    }


def probe_arm_parameter_frame(arms: list[ProbeArmData]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        reward_var = float(np.mean((arm.offer**2) * arm.prob) - arm.mu_reward**2)
        if arm.proxy is None:
            proxy_mean = 0.0
            proxy_sd = 0.0
            signed_rho = 0.0
        else:
            proxy_mean = arm.proxy_mean
            proxy_sd = float(np.std(arm.proxy))
            cov = float(np.mean((arm.expected_reward - arm.mu_reward) * (arm.proxy - proxy_mean)))
            signed_rho = cov / max(math.sqrt(max(reward_var, 1e-12)) * proxy_sd, 1e-12)
            signed_rho = float(np.clip(signed_rho, -0.999, 0.999))
        rows.append(
            {
                "arm": arm.arm,
                "mu_reward": arm.mu_reward,
                "mu_scaled": arm.mu_scaled,
                "sd_reward": math.sqrt(max(reward_var, 0.0)),
                "proxy_mean": proxy_mean,
                "proxy_sd": proxy_sd,
                "signed_rho": signed_rho,
                "abs_rho": abs(signed_rho),
                "reward_scale": arm.reward_scale,
            }
        )
    return pd.DataFrame(rows)


def simulate_probe(
    paired: pd.DataFrame,
    proxy_col: str | None = None,
    reps: int = 1000,
    delta: float = 0.05,
    kappa: float = 1.0,
    seed: int = 20260610,
    max_pulls: int = 1_000_000,
    reward_scale: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    arms = build_probe_arms(paired, proxy_col=proxy_col, reward_scale=reward_scale)
    rng = np.random.default_rng(seed + (abs(hash("probe")) % 100000))
    rows = []
    for rep in range(reps):
        row = run_probe_once(
            arms,
            delta=delta,
            kappa=kappa,
            seed=int(rng.integers(0, 2**31 - 1)),
            max_pulls=max_pulls,
        )
        row.update({"estimator": "probe", "rep": rep, "delta": delta, "kappa": kappa})
        rows.append(row)
    return pd.DataFrame(rows), probe_arm_parameter_frame(arms)


def summarize_probe(reps_df: pd.DataFrame, method_name: str, params_df: pd.DataFrame) -> pd.DataFrame:
    g = reps_df
    return pd.DataFrame(
        [
            {
                "method": method_name,
                "estimator": "probe",
                "delta": float(g["delta"].iloc[0]),
                "kappa": float(g["kappa"].iloc[0]),
                "mean_stop_pulls": float(g["stop_pulls"].mean()),
                "median_stop_pulls": float(g["stop_pulls"].median()),
                "q90_stop_pulls": float(g["stop_pulls"].quantile(0.90)),
                "empirical_correct_at_stop": float(g["correct"].mean()),
                "truncated_rate": float(g["truncated"].mean()),
                "mean_rounds": float(g["rounds"].mean()),
                "s_cal": float(g["s_cal"].iloc[0]),
                "reward_scale": float(g["reward_scale"].iloc[0]),
                "abs_rho_arm0": float(params_df.loc[params_df.arm == 0, "abs_rho"].iloc[0]),
                "abs_rho_arm1": float(params_df.loc[params_df.arm == 1, "abs_rho"].iloc[0]),
                "signed_rho_arm0": float(params_df.loc[params_df.arm == 0, "signed_rho"].iloc[0]),
                "signed_rho_arm1": float(params_df.loc[params_df.arm == 1, "signed_rho"].iloc[0]),
            }
        ]
    )
