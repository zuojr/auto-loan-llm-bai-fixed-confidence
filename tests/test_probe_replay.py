import math

import numpy as np
import pandas as pd

from src.probe_replay import (
    calibration_size,
    fresh_batch_size,
    simulate_probe,
    variance_upper_certificate,
)


def _easy_probe_pool(n: int = 1000) -> pd.DataFrame:
    rows = []
    for arm, prob in [(0, 0.1), (1, 0.9)]:
        proxy_center = -prob
        for i in range(n):
            proxy = proxy_center + 0.01 * ((i % 5) - 2)
            rows.append(
                {
                    "record_id": f"{arm}_{i}",
                    "arm": arm,
                    "offer_mp": 1.0,
                    "prob_accept": prob,
                    "proxy_easy": proxy,
                }
            )
    return pd.DataFrame(rows)


def test_probe_formula_helpers_match_algorithm_one():
    assert calibration_size(n_arms=2, delta=0.05, kappa=1.0) == 372
    assert math.isinf(variance_upper_certificate(batch_size=5, residual_variance=1.0, eta=0.01))
    cert = variance_upper_certificate(batch_size=1000, residual_variance=1.0, eta=0.01)
    assert cert > 1.0

    delta_r = 0.05 / (16 * 2 * 1**2)
    eps_r = 2.0**-1
    expected = max(
        3,
        math.ceil(1 + 16 * math.log(2 / delta_r)),
        math.ceil(8 * math.log(4 / delta_r)),
        math.ceil(64 * cert * math.log(2 / delta_r) * eps_r**-2),
    )
    assert fresh_batch_size(cert, delta_r, eps_r) == expected


def test_probe_runs_phase_elimination_with_calibration_and_fresh_batches():
    reps, params = simulate_probe(
        _easy_probe_pool(),
        proxy_col="proxy_easy",
        reps=5,
        delta=0.2,
        kappa=1.0,
        seed=123,
        max_pulls=100000,
        reward_scale=1.0,
    )

    assert set(reps["estimator"]) == {"probe"}
    assert reps["truncated"].sum() == 0
    assert reps["correct"].mean() == 1.0
    assert (reps["s_cal"] == calibration_size(2, 0.2, 1.0)).all()
    assert (reps["rounds"] >= 1).all()
    assert set(params["arm"]) == {0, 1}
