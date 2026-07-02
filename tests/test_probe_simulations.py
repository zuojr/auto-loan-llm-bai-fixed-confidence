import numpy as np

from src.probe_simulations import (
    correlation_certificate_diagnostics,
    historical_variance_upper_certificate,
    make_gaussian_arms,
    run_gaussian_benchmark,
    run_ml_proxy_simulation,
    run_probe_on_sim_arms,
    simulation_summary,
)


def test_gaussian_simulation_arms_match_tex_moments():
    arms = make_gaussian_arms(rho=0.8, pool_size=50_000, seed=123)

    reward_means = [arm.reward_mean for arm in arms]
    proxy_means = [arm.proxy_mean for arm in arms]
    empirical_corrs = [np.corrcoef(arm.reward, arm.proxy)[0, 1] for arm in arms]

    assert reward_means == [1.00, 0.90, 0.84, 0.76, 0.70, 0.62, 0.54, 0.44]
    assert proxy_means == [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75]
    assert reward_means[0] > reward_means[-1]
    assert proxy_means[0] < proxy_means[-1]
    assert all(abs(corr - 0.8) < 0.025 for corr in empirical_corrs)


def test_historical_certificate_uses_total_history_formula():
    s = 102
    vhat = 0.5
    expected = vhat / (1.0 - 2.0 * np.sqrt(1.0 / (s - 2.0)))

    assert np.isclose(historical_variance_upper_certificate(s, vhat), expected)


def test_proxy_probe_reduces_gaussian_stopping_time_at_high_correlation():
    arms = make_gaussian_arms(rho=0.8, pool_size=80_000, seed=456)

    reward_only = run_probe_on_sim_arms(
        arms,
        method="reward_only_probe",
        reps=8,
        delta=0.2,
        seed=111,
        max_pulls=800_000,
        use_proxy=False,
    )
    proxy_probe = run_probe_on_sim_arms(
        arms,
        method="proxy_probe",
        reps=8,
        delta=0.2,
        seed=111,
        max_pulls=800_000,
        use_proxy=True,
    )

    reward_summary = simulation_summary(reward_only, "reward_only_probe")
    proxy_summary = simulation_summary(proxy_probe, "proxy_probe")

    assert reward_summary["empirical_correct_at_stop"].iloc[0] >= 0.75
    assert proxy_summary["empirical_correct_at_stop"].iloc[0] >= 0.75
    assert proxy_summary["mean_stop_pulls"].iloc[0] < reward_summary["mean_stop_pulls"].iloc[0]


def test_gaussian_benchmark_reports_reward_proxy_and_oracle_rows():
    result = run_gaussian_benchmark(
        rhos=[0.0, 0.8],
        reps=3,
        pool_size=30_000,
        delta=0.2,
        seed=7,
        max_pulls=800_000,
    )

    assert set(result["rho"]) == {0.0, 0.8}
    assert set(result["method"]) == {"reward_only_probe", "known_oracle_probe", "unknown_probe"}
    assert "sample_ratio_vs_reward_only" in result.columns
    high = result[result["rho"] == 0.8].set_index("method")
    assert high.loc["known_oracle_probe", "oracle_residual_factor"] == 1.0 - 0.8**2
    assert high.loc["known_oracle_probe", "oracle_source"] == "known_residualized_reward"
    assert high.loc["unknown_probe", "sample_ratio_vs_reward_only"] < 1.0


def test_gaussian_known_oracle_is_simulated_not_scaled_from_reward_only():
    result = run_gaussian_benchmark(
        rhos=[0.8],
        reps=10,
        pool_size=40_000,
        delta=0.2,
        seed=13,
        max_pulls=80_000,
    )

    row = result[result["rho"] == 0.8].set_index("method")
    baseline_mean = row.loc["reward_only_probe", "mean_stop_pulls"]
    known = row.loc["known_oracle_probe"]

    assert known["oracle_source"] == "known_residualized_reward"
    assert not np.isclose(
        known["mean_stop_pulls"],
        baseline_mean * known["oracle_residual_factor"],
    )


def test_gaussian_unknown_probe_pays_learning_penalty_relative_to_known_oracle():
    result = run_gaussian_benchmark(
        rhos=[0.2],
        reps=5,
        pool_size=20_000,
        delta=0.2,
        seed=19,
        max_pulls=80_000,
    )

    row = result[result["rho"] == 0.2].set_index("method")

    assert row.loc["unknown_probe", "tir_variance_coef"] > row.loc[
        "known_oracle_probe", "tir_variance_coef"
    ]


def test_gaussian_benchmark_uses_historical_certificate_without_online_calibration():
    result = run_gaussian_benchmark(
        rhos=[0.0],
        reps=12,
        pool_size=60_000,
        delta=0.2,
        seed=17,
        max_pulls=50_000,
    )

    indexed = result[result["rho"] == 0.0].set_index("method")
    baseline = indexed.loc["reward_only_probe"]
    unknown = indexed.loc["unknown_probe"]

    assert baseline["certificate_source"] == "historical_pool"
    assert unknown["certificate_source"] == "historical_pool"
    assert baseline["s_cal"] == 0
    assert unknown["s_cal"] == 0
    assert 500 < baseline["mean_stop_pulls"] < 10_000
    assert baseline["empirical_correct_at_stop"] >= 0.8
    assert unknown["empirical_correct_at_stop"] >= 0.8


def test_certificate_diagnostics_show_probe_certificate_is_more_conservative():
    diag = correlation_certificate_diagnostics(
        rhos=[0.4, 0.8],
        reps_learning=20,
        reps_failure=200,
        sample_size=100,
        eta=0.05,
        seed=8,
    )

    assert set(diag["rho"]) == {0.4, 0.8}
    assert {"mean_certification_time", "plugin_failure_rate", "certificate_failure_rate"} <= set(diag.columns)
    assert (diag["certificate_failure_rate"] <= diag["plugin_failure_rate"]).all()


def test_ml_proxy_simulation_returns_proxy_model_ratios():
    result = run_ml_proxy_simulation(
        reps=3,
        train_size=600,
        pool_size=2_000,
        delta=0.2,
        seed=9,
        max_pulls=800_000,
    )

    assert set(result["proxy_model"]) == {"linear", "decision_tree", "gradient_boosting"}
    assert {"mean_corr", "min_corr", "probe_ratio_vs_reward_only"} <= set(result.columns)
    assert result["mean_corr"].between(-1.0, 1.0).all()
    assert result["probe_ratio_vs_reward_only"].notna().all()
