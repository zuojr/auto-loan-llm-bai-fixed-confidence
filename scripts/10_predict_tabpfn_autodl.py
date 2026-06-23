#!/usr/bin/env python
"""Predict auto-loan acceptance probabilities with an optional TabPFN proxy.

This script is for appendix/robustness experiments only. It trains TabPFN on
historical labeled rows and scores the same paired application-offer rows used
by the LLM methods. The output schema is exactly:

    record_id,pred_prob

No replay-environment columns are used as features.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import RANDOM_SEED, TARGET

NUMERIC_FEATURES = [
    "Primary_FICO",
    "Tier",
    "Term",
    "Amount_Approved",
    "Competition_rate",
    "offer_mp",
    "offer_ratio",
    "partnerbin",
    "CarType_id",
    "days",
    "weeks",
    "months",
    "termclass",
]

CATEGORICAL_FEATURES = ["State", "Type", "CarType"]
TABPFN_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LEAKAGE_COLUMNS = {
    "prob_accept",
    "expected_reward_env",
    "observed_apply",
    "reward",
    "realized_reward",
    "proxy",
    "proxy_tabpfn",
    "record_id",
    "base_row_id",
    "profile_text",
}


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def validate_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def sample_train_rows(train: pd.DataFrame, train_size: int | None, seed: int) -> pd.DataFrame:
    if train_size is None or train_size >= len(train):
        return train.reset_index(drop=True)
    return train.sample(n=train_size, random_state=seed).reset_index(drop=True)


def build_feature_matrices(
    train: pd.DataFrame,
    scoring: pd.DataFrame,
    train_size: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Build encoded train/scoring matrices from a strict feature whitelist."""
    validate_columns(train, TABPFN_FEATURES + [TARGET], "train")
    validate_columns(scoring, TABPFN_FEATURES + ["record_id"], "scoring")

    train = sample_train_rows(train, train_size=train_size, seed=seed)
    y_train = pd.to_numeric(train[TARGET], errors="raise").astype(int).to_numpy()
    if set(np.unique(y_train)) - {0, 1}:
        raise ValueError(f"{TARGET} must be binary 0/1 for TabPFN classification")

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", make_one_hot_encoder()),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )
    X_train = pre.fit_transform(train[TABPFN_FEATURES])
    X_score = pre.transform(scoring[TABPFN_FEATURES])
    record_ids = scoring["record_id"].astype(str).tolist()
    return (
        np.asarray(X_train, dtype=np.float32),
        y_train,
        np.asarray(X_score, dtype=np.float32),
        record_ids,
        list(TABPFN_FEATURES),
    )


def make_classifier(backend: str, device: str | None, seed: int):
    if backend == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise RuntimeError(
                "TabPFN is not installed. Install it with `pip install tabpfn`, "
                "or use `--backend sklearn` for a local smoke test only."
            ) from exc
        kwargs = {}
        if device:
            kwargs["device"] = device
        return TabPFNClassifier(**kwargs)
    if backend == "sklearn":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed, max_iter=50)
    raise ValueError(f"Unknown backend: {backend}")


def positive_class_probability(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        idx = int(np.where(classes == 1)[0][0])
    else:
        idx = min(1, proba.shape[1] - 1)
    return np.clip(np.asarray(proba[:, idx], dtype=float), 0.01, 0.99)


def write_predictions(path: Path, record_ids: list[str], pred_prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "pred_prob"])
        writer.writeheader()
        for rid, prob in zip(record_ids, pred_prob):
            writer.writerow({"record_id": rid, "pred_prob": float(prob)})


def main() -> None:
    ap = argparse.ArgumentParser(description="Score paired rows with optional TabPFN structured proxy.")
    ap.add_argument("--train", default=str(ROOT / "processed/historical_train_rows.csv"))
    ap.add_argument("--scoring", default=str(ROOT / "api_scoring/scoring_full_paired.csv"))
    ap.add_argument("--out", default=str(ROOT / "predictions/tabpfn_predictions.csv"))
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Optional scoring-row limit for pilots.")
    ap.add_argument("--backend", choices=["tabpfn", "sklearn"], default="tabpfn")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    train_path = Path(args.train)
    scoring_path = Path(args.scoring)
    out_path = Path(args.out)

    train = pd.read_csv(train_path)
    scoring = pd.read_csv(scoring_path)
    if args.limit is not None:
        scoring = scoring.head(args.limit).copy()

    X_train, y_train, X_score, record_ids, _ = build_feature_matrices(
        train, scoring, train_size=args.train_size, seed=args.seed
    )
    model = make_classifier(args.backend, args.device if args.backend == "tabpfn" else None, args.seed)
    print(
        f"Fitting {args.backend} proxy on {len(y_train)} historical rows; "
        f"scoring {len(record_ids)} paired rows with {X_train.shape[1]} encoded features."
    )
    model.fit(X_train, y_train)
    pred_prob = positive_class_probability(model, X_score)
    write_predictions(out_path, record_ids, pred_prob)
    print(f"Wrote {out_path} with columns record_id,pred_prob")


if __name__ == "__main__":
    main()
