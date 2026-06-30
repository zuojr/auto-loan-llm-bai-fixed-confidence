import numpy as np

from src.probe_simulations import (
    correlation_certificate_diagnostics,
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
    assert set(result["method"]) == {"reward_only_probe", "proxy_probe", "oracle_residual_probe"}
    assert "sample_ratio_vs_reward_only" in result.columns
    high = result[result["rho"] == 0.8].set_index("method")
    assert high.loc["proxy_probe", "sample_ratio_vs_reward_only"] < 1.0
    assert high.loc["oracle_residual_probe", "sample_ratio_vs_reward_only"] < 1.0


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
