import numpy as np
import pandas as pd

from src.fixed_confidence_replay import build_arms, simulate_fixed_confidence


def _negative_proxy_pool(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(20260617)
    rows = []
    for arm, offset in [(0, 0.0), (1, 0.02)]:
        quality = np.linspace(0.05, 0.95, n)
        offer = 100.0 + 20.0 * rng.normal(size=n) + 10.0 * arm
        prob = np.clip(quality + offset, 0.01, 0.99)
        proxy = -quality + 0.01 * rng.normal(size=n)
        for i in range(n):
            rows.append(
                {
                    "record_id": f"{arm}_{i}",
                    "arm": arm,
                    "offer_mp": offer[i],
                    "prob_accept": prob[i],
                    "proxy_neg": proxy[i],
                }
            )
    return pd.DataFrame(rows)


def test_arm_params_preserve_signed_negative_rho():
    arms = build_arms(_negative_proxy_pool(), proxy_col="proxy_neg")

    assert arms[0].signed_rho < -0.3
    assert arms[0].rho == 0.0
    assert abs(arms[0].signed_rho) == arms[0].abs_rho


def test_signed_oracle_cv_uses_negative_correlation_variance_bound():
    paired = _negative_proxy_pool()

    _, params = simulate_fixed_confidence(
        paired,
        proxy_col="proxy_neg",
        estimators=["oracle_cv", "signed_oracle_cv"],
        reps=2,
        seed=7,
        max_pulls=200,
        batch_size=10,
    )

    arm0 = params.loc[params.arm == 0].iloc[0]
    assert arm0["reward_proxy_rho"] == 0.0
    assert arm0["reward_proxy_signed_rho"] < -0.3
    assert arm0["reward_proxy_abs_rho"] > 0.3
