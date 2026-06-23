from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_historical_train_rows_uses_only_historical_labels(tmp_path):
    prep = load_module(ROOT / "scripts" / "01_prepare_replay_environment.py", "prepare_replay")
    train = pd.DataFrame(
        {
            "apply": [1, 0],
            "Primary_FICO": [720, 680],
            "Tier": [1, 4],
            "Term": [60, 72],
            "Amount_Approved": [25000.0, 30000.0],
            "Competition_rate": [5.1, 6.2],
            "mp": [500.0, 650.0],
            "mp_rto_amtfinance": [0.02, 0.0216667],
            "partnerbin": [1, 3],
            "CarType_id": [0, 1],
            "days": [10, 20],
            "weeks": [1, 3],
            "months": [0, 1],
            "termclass": [2, 3],
            "State": ["TX", "CA"],
            "Type": ["F", "N"],
            "CarType": ["U", "N"],
            "base_row_id": [99, 100],
            "prob_accept": [0.9, 0.1],
            "expected_reward_env": [450.0, 65.0],
            "observed_apply": [1, 0],
            "profile_text": ["leak", "leak"],
        }
    )
    out = tmp_path / "historical_train_rows.csv"

    prep.make_historical_train_rows(train, out)

    got = pd.read_csv(out)
    assert got["offer_mp"].tolist() == [500.0, 650.0]
    assert got["offer_ratio"].round(6).tolist() == [0.02, 0.021667]
    assert got["apply"].tolist() == [1, 0]
    forbidden = {"prob_accept", "expected_reward_env", "observed_apply", "base_row_id", "profile_text"}
    assert forbidden.isdisjoint(got.columns)


def test_tabpfn_feature_builder_excludes_leakage_and_handles_unknown_categories():
    tabpfn = load_module(ROOT / "scripts" / "10_predict_tabpfn_autodl.py", "tabpfn_predict")
    train = pd.DataFrame(
        {
            "record_id": ["train_should_not_be_used", "train_should_not_be_used_2"],
            "apply": [1, 0],
            "Primary_FICO": [720, 680],
            "Tier": [1, 4],
            "Term": [60, 72],
            "Amount_Approved": [25000.0, 30000.0],
            "Competition_rate": [5.1, 6.2],
            "offer_mp": [500.0, 650.0],
            "offer_ratio": [0.02, 0.0216667],
            "partnerbin": [1, 3],
            "CarType_id": [0, 1],
            "days": [10, 20],
            "weeks": [1, 3],
            "months": [0, 1],
            "termclass": [2, 3],
            "State": ["TX", "CA"],
            "Type": ["F", "N"],
            "CarType": ["U", "N"],
            "prob_accept": [0.9, 0.1],
            "expected_reward_env": [450.0, 65.0],
            "observed_apply": [1, 0],
            "reward": [500.0, 0.0],
            "profile_text": ["leak", "leak"],
        }
    )
    scoring = train.drop(columns=["apply"]).copy()
    scoring["record_id"] = ["row_1", "row_2"]
    scoring.loc[0, "State"] = "ZZ"
    scoring.loc[0, "Type"] = "UNKNOWN_TYPE"
    scoring.loc[0, "CarType"] = "UNKNOWN_CAR"

    X_train, y_train, X_score, record_ids, feature_names = tabpfn.build_feature_matrices(train, scoring)

    assert y_train.tolist() == [1, 0]
    assert record_ids == ["row_1", "row_2"]
    assert X_train.shape[0] == 2
    assert X_score.shape[0] == 2
    assert set(tabpfn.TABPFN_FEATURES).issubset(set(feature_names))
    assert not any(col in feature_names for col in tabpfn.LEAKAGE_COLUMNS)
