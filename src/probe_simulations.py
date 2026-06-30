from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.probe_replay import (
    calibration_size,
    fresh_batch_size,
    ols_intercept_slope_residual_variance,
    variance_upper_certificate,
)


DEFAULT_GAUSSIAN_REWARD_MEANS = [1.00, 0.90, 0.84, 0.76, 0.70, 0.62, 0.54, 0.44]
DEFAULT_GAUSSIAN_PROXY_MEANS = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75]
DEFAULT_ML_REWARD_MEANS = [1.00, 0.90, 0.84, 0.76, 0.70]
DEFAULT_ML_PROXY_MEANS = [0.00, 0.30, 0.60, 0.90, 1.20]


@dataclass(frozen=True)
class SimulationArmData:
    arm: int
    reward: np.ndarray
    proxy: np.ndarray | None
    reward_mean: float
    proxy_mean: float
    reward_sd: float

    @property
    def n_pool(self) -> int:
        return int(len(self.reward))


def make_gaussian_arms(
    rho: float,
    pool_size: int = 200_000,
    seed: int = 20260630,
    reward_means: list[float] | None = None,
    proxy_means: list[float] | None = None,
) -> list[SimulationArmData]:
    if not -0.999 < rho < 0.999:
        raise ValueError("rho must be in (-0.999, 0.999).")
    if pool_size < 100:
        raise ValueError("pool_size must be at least 100.")

    reward_means = reward_means or DEFAULT_GAUSSIAN_REWARD_MEANS
    proxy_means = proxy_means or DEFAULT_GAUSSIAN_PROXY_MEANS
    if len(reward_means) != len(proxy_means):
        raise ValueError("reward_means and proxy_means must have the same length.")

    rng = np.random.default_rng(seed)
    arms = []
    noise_scale = math.sqrt(max(1.0 - rho**2, 0.0))
    for arm, (reward_mean, proxy_mean) in enumerate(zip(reward_means, proxy_means)):
        xi = rng.normal(size=pool_size)
        w = rng.normal(size=pool_size)
        reward = reward_mean + xi
        proxy = proxy_mean + rho * xi + noise_scale * w
        arms.append(
            SimulationArmData(
                arm=arm,
                reward=reward.astype(float),
                proxy=proxy.astype(float),
                reward_mean=float(reward_mean),
                proxy_mean=float(proxy_mean),
                reward_sd=1.0,
            )
        )
    return arms


def make_oracle_residual_arms(arms: list[SimulationArmData], rho: float) -> list[SimulationArmData]:
    residual_arms = []
    for arm in arms:
        if arm.proxy is None:
            raise ValueError("oracle residual arms require proxy samples.")
        centered_proxy = arm.proxy - arm.proxy_mean
        residualized = arm.reward - rho * centered_proxy
        residual_arms.append(
            SimulationArmData(
                arm=arm.arm,
                reward=residualized,
                proxy=None,
                reward_mean=arm.reward_mean,
                proxy_mean=0.0,
                reward_sd=math.sqrt(max(1.0 - rho**2, 1e-12)),
            )
        )
    return residual_arms


def _sample_sim_batch(
    arm: SimulationArmData,
    rng: np.random.Generator,
    size: int,
    use_proxy: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    idx = rng.integers(0, arm.n_pool, size=size)
    reward = arm.reward[idx].astype(float)
    if not use_proxy or arm.proxy is None:
        return reward, None
    return reward, arm.proxy[idx].astype(float) - arm.proxy_mean


def run_probe_once_on_sim_arms(
    arms: list[SimulationArmData],
    delta: float = 0.05,
    kappa: float = 1.0,
    seed: int | None = None,
    max_pulls: int = 2_000_000,
    use_proxy: bool = True,
) -> dict[str, float | int]:
    if len(arms) < 2:
        raise ValueError("simulation PROBE requires at least two arms.")

    rng = np.random.default_rng(seed)
    n_arms = len(arms)
    true_best = int(np.argmax([arm.reward_mean for arm in arms]))
    s_cal = calibration_size(n_arms, delta, kappa)

    pulls = 0
    n_by_arm = np.zeros(n_arms, dtype=int)
    certificates = np.zeros(n_arms, dtype=float)
    last_intercepts = np.array([arm.reward_mean for arm in arms], dtype=float)
    last_slopes = np.zeros(n_arms, dtype=float)

    for ai, arm in enumerate(arms):
        if pulls + s_cal > max_pulls:
            return _sim_result(
                pulls,
                np.argmax(last_intercepts),
                true_best,
                0,
                True,
                s_cal,
                n_by_arm,
                certificates,
                last_slopes,
            )
        reward, proxy = _sample_sim_batch(arm, rng, s_cal, use_proxy=use_proxy)
        _, _, vhat = ols_intercept_slope_residual_variance(reward, proxy)
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
            batch_size = fresh_batch_size(certificates[ai], delta_r, epsilon_r)
            if pulls + batch_size > max_pulls:
                truncated = True
                break
            reward, proxy = _sample_sim_batch(arms[ai], rng, batch_size, use_proxy=use_proxy)
            intercept, slope, vhat = ols_intercept_slope_residual_variance(reward, proxy)
            cert = variance_upper_certificate(batch_size, vhat, delta_r)
            certificates[ai] = min(certificates[ai], cert)
            last_intercepts[ai] = intercept
            last_slopes[ai] = slope
            round_intercepts[ai] = intercept
            pulls += batch_size
            n_by_arm[ai] += batch_size

        if truncated:
            break

        best = max(active, key=lambda idx: round_intercepts[idx])
        threshold = round_intercepts[best] - epsilon_r
        active = [idx for idx in active if idx == best or round_intercepts[idx] >= threshold]

    if active:
        recommendation = int(max(active, key=lambda idx: last_intercepts[idx]))
    else:
        recommendation = int(np.argmax(last_intercepts))
    return _sim_result(
        pulls,
        recommendation,
        true_best,
        rounds,
        truncated,
        s_cal,
        n_by_arm,
        certificates,
        last_slopes,
    )


def _sim_result(
    pulls: int,
    recommendation: int,
    true_best: int,
    rounds: int,
    truncated: bool,
    s_cal: int,
    n_by_arm: np.ndarray,
    certificates: np.ndarray,
    slopes: np.ndarray,
) -> dict[str, float | int]:
    return {
        "stop_pulls": int(pulls),
        "recommendation": int(recommendation),
        "correct": int(recommendation == true_best),
        "rounds": int(rounds),
        "truncated": int(truncated),
        "s_cal": int(s_cal),
        "n_arm0": int(n_by_arm[0]) if len(n_by_arm) > 0 else 0,
        "n_arm1": int(n_by_arm[1]) if len(n_by_arm) > 1 else 0,
        "final_U_arm0": float(certificates[0]) if len(certificates) > 0 else np.nan,
        "final_U_arm1": float(certificates[1]) if len(certificates) > 1 else np.nan,
        "final_slope_arm0": float(slopes[0]) if len(slopes) > 0 else np.nan,
        "final_slope_arm1": float(slopes[1]) if len(slopes) > 1 else np.nan,
    }


def run_probe_on_sim_arms(
    arms: list[SimulationArmData],
    method: str,
    reps: int = 120,
    delta: float = 0.05,
    kappa: float = 1.0,
    seed: int = 20260630,
    max_pulls: int = 2_000_000,
    use_proxy: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        row = run_probe_once_on_sim_arms(
            arms,
            delta=delta,
            kappa=kappa,
            seed=int(rng.integers(0, 2**31 - 1)),
            max_pulls=max_pulls,
            use_proxy=use_proxy,
        )
        row.update({"method": method, "rep": rep, "delta": delta, "kappa": kappa})
        rows.append(row)
    return pd.DataFrame(rows)


def simulation_summary(reps_df: pd.DataFrame, method: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": method,
                "delta": float(reps_df["delta"].iloc[0]),
                "kappa": float(reps_df["kappa"].iloc[0]),
                "mean_stop_pulls": float(reps_df["stop_pulls"].mean()),
                "median_stop_pulls": float(reps_df["stop_pulls"].median()),
                "q90_stop_pulls": float(reps_df["stop_pulls"].quantile(0.90)),
                "empirical_correct_at_stop": float(reps_df["correct"].mean()),
                "truncated_rate": float(reps_df["truncated"].mean()),
                "mean_rounds": float(reps_df["rounds"].mean()),
                "s_cal": float(reps_df["s_cal"].iloc[0]),
            }
        ]
    )


def run_gaussian_benchmark(
    rhos: list[float] | None = None,
    reps: int = 120,
    pool_size: int = 200_000,
    delta: float = 0.05,
    kappa: float = 1.0,
    seed: int = 20260630,
    max_pulls: int = 2_000_000,
) -> pd.DataFrame:
    rhos = rhos or [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    rows = []
    for ridx, rho in enumerate(rhos):
        arms = make_gaussian_arms(rho=rho, pool_size=pool_size, seed=seed + 1009 * ridx)
        method_specs = [
            ("reward_only_probe", arms, False),
            ("proxy_probe", arms, True),
            ("oracle_residual_probe", make_oracle_residual_arms(arms, rho), False),
        ]
        summaries = []
        for midx, (method, method_arms, use_proxy) in enumerate(method_specs):
            reps_df = run_probe_on_sim_arms(
                method_arms,
                method=method,
                reps=reps,
                delta=delta,
                kappa=kappa,
                seed=seed + 100_003 * ridx + 7_919 * midx,
                max_pulls=max_pulls,
                use_proxy=use_proxy,
            )
            summary = simulation_summary(reps_df, method)
            summary.insert(0, "rho", float(rho))
            summaries.append(summary)
        rho_df = pd.concat(summaries, ignore_index=True)
        baseline_mean = float(rho_df.loc[rho_df["method"] == "reward_only_probe", "mean_stop_pulls"].iloc[0])
        rho_df["sample_ratio_vs_reward_only"] = rho_df["mean_stop_pulls"] / baseline_mean
        rho_df["oracle_residual_factor"] = 1.0 - float(rho) ** 2
        rows.append(rho_df)
    return pd.concat(rows, ignore_index=True)


def _one_arm_gaussian_sample(
    rho: float,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    z = rng.normal(size=size)
    w = rng.normal(size=size)
    x = 1.0 + z
    y = rho * z + math.sqrt(max(1.0 - rho**2, 0.0)) * w
    return x, y


def _true_residual_variance_for_slope(rho: float, slope: float) -> float:
    return float(1.0 - 2.0 * slope * rho + slope**2)


def correlation_certificate_diagnostics(
    rhos: list[float] | None = None,
    reps_learning: int = 500,
    reps_failure: int = 15_000,
    sample_size: int = 100,
    eta: float = 0.05,
    seed: int = 20260630,
    max_learning_size: int = 8_000,
    learning_step: int = 25,
) -> pd.DataFrame:
    rhos = rhos or [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]
    rng = np.random.default_rng(seed)
    rows = []
    for rho in rhos:
        target_certificate = 1.0 - 0.5 * float(rho) ** 2
        learning_times = []
        for _ in range(reps_learning):
            certified_at = max_learning_size
            for t in range(max(sample_size, 25), max_learning_size + 1, learning_step):
                x, y = _one_arm_gaussian_sample(float(rho), t, rng)
                _, _, vhat = ols_intercept_slope_residual_variance(x, y)
                cert = variance_upper_certificate(t, vhat, eta)
                if cert <= target_certificate:
                    certified_at = t
                    break
            learning_times.append(certified_at)

        plugin_failures = 0
        certificate_failures = 0
        finite_certificates = 0
        for _ in range(reps_failure):
            x, y = _one_arm_gaussian_sample(float(rho), sample_size, rng)
            _, slope, vhat = ols_intercept_slope_residual_variance(x, y)
            true_resid_var = _true_residual_variance_for_slope(float(rho), slope)
            cert = variance_upper_certificate(sample_size, vhat, eta)
            plugin_failures += int(true_resid_var > vhat)
            certificate_failures += int(true_resid_var > cert)
            finite_certificates += int(np.isfinite(cert))

        rows.append(
            {
                "rho": float(rho),
                "target_certificate": target_certificate,
                "mean_certification_time": float(np.mean(learning_times)),
                "median_certification_time": float(np.median(learning_times)),
                "plugin_failure_rate": plugin_failures / reps_failure,
                "certificate_failure_rate": certificate_failures / reps_failure,
                "finite_certificate_rate": finite_certificates / reps_failure,
                "sample_size": int(sample_size),
                "eta": float(eta),
            }
        )
    return pd.DataFrame(rows)


def _ml_actions(n_arms: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n_arms)


def _ml_features(u: np.ndarray, action: np.ndarray) -> np.ndarray:
    action = action.reshape(-1, 1)
    return np.column_stack(
        [
            u,
            action,
            action * u[:, [0]],
            action * u[:, [1]],
            action**2,
        ]
    )


def _ml_signal(u: np.ndarray, action: np.ndarray) -> np.ndarray:
    return (
        0.90 * np.sin(u[:, 0] + action)
        + 0.55 * action * u[:, 1]
        + 0.35 * (u[:, 2] ** 2 - 1.0) * (action + 0.4)
        + 0.40 * np.cos(u[:, 3] - 0.5 * action)
        + 0.20 * u[:, 4]
    )


def _ml_normalizers(
    actions: np.ndarray,
    rng: np.random.Generator,
    n: int = 50_000,
    noise_sd: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    sds = []
    for action in actions:
        u = rng.normal(size=(n, 5))
        y = _ml_signal(u, np.full(n, action)) + noise_sd * rng.normal(size=n)
        means.append(float(np.mean(y)))
        sds.append(float(np.std(y)))
    return np.asarray(means), np.asarray(sds)


def _draw_ml_arm_pool(
    arm: int,
    action: float,
    reward_mean: float,
    normalizer_mean: float,
    normalizer_sd: float,
    pool_size: int,
    rng: np.random.Generator,
    noise_sd: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.normal(size=(pool_size, 5))
    noisy_signal = _ml_signal(u, np.full(pool_size, action)) + noise_sd * rng.normal(size=pool_size)
    reward = reward_mean + (noisy_signal - normalizer_mean) / max(normalizer_sd, 1e-12)
    features = _ml_features(u, np.full(pool_size, action))
    return reward.astype(float), features.astype(float)


def run_ml_proxy_simulation(
    reps: int = 120,
    train_size: int = 12_000,
    pool_size: int = 40_000,
    delta: float = 0.05,
    kappa: float = 1.0,
    seed: int = 20260630,
    max_pulls: int = 2_000_000,
) -> pd.DataFrame:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(seed)
    reward_means = DEFAULT_ML_REWARD_MEANS
    proxy_means = DEFAULT_ML_PROXY_MEANS
    n_arms = len(reward_means)
    actions = _ml_actions(n_arms)
    norm_means, norm_sds = _ml_normalizers(actions, rng)

    train_arms = rng.integers(0, n_arms, size=train_size)
    train_u = rng.normal(size=(train_size, 5))
    train_actions = actions[train_arms]
    train_signal = _ml_signal(train_u, train_actions) + 0.45 * rng.normal(size=train_size)
    train_y = np.asarray(reward_means)[train_arms] + (
        train_signal - norm_means[train_arms]
    ) / np.maximum(norm_sds[train_arms], 1e-12)
    train_x = _ml_features(train_u, train_actions)

    models = {
        "linear": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=40, random_state=seed + 1),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed + 2,
        ),
    }
    for model in models.values():
        model.fit(train_x, train_y)

    base_rewards = []
    arm_features = []
    for arm, action in enumerate(actions):
        reward, features = _draw_ml_arm_pool(
            arm,
            float(action),
            float(reward_means[arm]),
            float(norm_means[arm]),
            float(norm_sds[arm]),
            pool_size,
            rng,
        )
        base_rewards.append(reward)
        arm_features.append(features)

    base_arms = [
        SimulationArmData(
            arm=arm,
            reward=base_rewards[arm],
            proxy=None,
            reward_mean=float(reward_means[arm]),
            proxy_mean=0.0,
            reward_sd=float(np.std(base_rewards[arm])),
        )
        for arm in range(n_arms)
    ]
    baseline_reps = run_probe_on_sim_arms(
        base_arms,
        method="reward_only_probe",
        reps=reps,
        delta=delta,
        kappa=kappa,
        seed=seed + 31,
        max_pulls=max_pulls,
        use_proxy=False,
    )
    baseline_mean = float(baseline_reps["stop_pulls"].mean())

    rows = []
    for midx, (model_name, model) in enumerate(models.items()):
        arms = []
        correlations = []
        for arm in range(n_arms):
            raw_pred = model.predict(arm_features[arm])
            pred_sd = float(np.std(raw_pred))
            if pred_sd <= 1e-12:
                standardized = np.zeros_like(raw_pred)
            else:
                standardized = (raw_pred - float(np.mean(raw_pred))) / pred_sd
            proxy = float(proxy_means[arm]) + standardized
            corr = float(np.corrcoef(base_rewards[arm], proxy)[0, 1])
            correlations.append(corr)
            arms.append(
                SimulationArmData(
                    arm=arm,
                    reward=base_rewards[arm],
                    proxy=proxy,
                    reward_mean=float(reward_means[arm]),
                    proxy_mean=float(proxy_means[arm]),
                    reward_sd=float(np.std(base_rewards[arm])),
                )
            )
        reps_df = run_probe_on_sim_arms(
            arms,
            method=f"{model_name}_proxy_probe",
            reps=reps,
            delta=delta,
            kappa=kappa,
            seed=seed + 101 + 37 * midx,
            max_pulls=max_pulls,
            use_proxy=True,
        )
        proxy_mean_stop = float(reps_df["stop_pulls"].mean())
        rows.append(
            {
                "proxy_model": model_name,
                "mean_corr": float(np.mean(correlations)),
                "min_corr": float(np.min(correlations)),
                "max_corr": float(np.max(correlations)),
                "reward_only_mean_stop_pulls": baseline_mean,
                "probe_mean_stop_pulls": proxy_mean_stop,
                "probe_ratio_vs_reward_only": proxy_mean_stop / baseline_mean,
                "empirical_correct_at_stop": float(reps_df["correct"].mean()),
                "truncated_rate": float(reps_df["truncated"].mean()),
            }
        )
    return pd.DataFrame(rows)
