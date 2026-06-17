from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


CLS_SYSTEM_PROMPT = "You are a calibrated auto-loan offer acceptance classifier. Return only yes or no."


def _clip_prob(x: np.ndarray | Iterable[float], lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), lo, hi)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p / (1.0 - p))


def make_binary_acceptance_prompt(row_or_text) -> str:
    """Prompt for yes/no LoRA classification using the same application-offer profile."""
    if isinstance(row_or_text, str):
        tail = row_or_text
    else:
        policy_line = (
            f"Pricing rule: {row_or_text.get('policy', 'NA')}; multiplier relative to historical offer: "
            f"{float(row_or_text.get('multiplier', 1.0)):.2f}.\n"
        )
        tail = f"{policy_line}Application-offer profile:\n{row_or_text['profile_text']}\n"
    return (
        "Predict whether the customer accepts this offered auto-loan contract.\n\n"
        "Target:\n"
        "Answer yes if the customer accepts/applies under this offered contract, otherwise answer no.\n\n"
        "This is an acceptance task, not a default-risk task and not a policy-ranking task.\n"
        "Return exactly one token: yes or no.\n\n"
        f"{tail}"
    )


def _prompt_tail_from_numeric_prompt(prompt: str) -> str:
    marker = "Pricing rule:"
    if marker in prompt:
        return prompt[prompt.index(marker) :]
    marker = "Application-offer profile:"
    if marker in prompt:
        return prompt[prompt.index(marker) :]
    return prompt


def convert_numeric_jsonl_to_yes_no(
    src_path: str | Path,
    out_path: str | Path,
    rewrite_prompt: bool = True,
) -> int:
    """Convert existing pred_prob SFT JSONL to yes/no completion JSONL."""
    src = Path(src_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            messages = list(obj["messages"])
            target = json.loads(messages[-1]["content"])["pred_prob"]
            if rewrite_prompt:
                if messages and messages[0].get("role") == "system":
                    messages[0] = {"role": "system", "content": CLS_SYSTEM_PROMPT}
                messages[-2] = {
                    "role": "user",
                    "content": make_binary_acceptance_prompt(_prompt_tail_from_numeric_prompt(messages[-2]["content"])),
                }
            messages[-1] = {"role": "assistant", "content": "yes" if float(target) > 0.5 else "no"}
            f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            n += 1
    return n


def yes_no_probability_from_logits(
    logits: np.ndarray | Iterable[float],
    yes_token_ids: list[int],
    no_token_ids: list[int],
) -> float:
    """Compute P(yes)/(P(yes)+P(no)) from next-token logits for token candidates."""
    arr = np.asarray(logits, dtype=float)
    if not yes_token_ids or not no_token_ids:
        raise ValueError("yes_token_ids and no_token_ids must be non-empty.")
    yes = arr[np.asarray(yes_token_ids, dtype=int)]
    no = arr[np.asarray(no_token_ids, dtype=int)]
    max_logit = float(np.max(np.concatenate([yes, no])))
    yes_mass = float(np.exp(yes - max_logit).sum())
    no_mass = float(np.exp(no - max_logit).sum())
    return yes_mass / max(yes_mass + no_mass, 1e-300)


def fit_platt_calibrator(raw_pred_prob: Iterable[float], y: Iterable[int]) -> dict[str, float]:
    """Fit logistic calibration on logit(raw_pred_prob)."""
    raw = _clip_prob(raw_pred_prob)
    labels = np.asarray(list(y), dtype=int)
    if raw.shape[0] != labels.shape[0]:
        raise ValueError("raw_pred_prob and y must have the same length.")
    if len(np.unique(labels)) < 2:
        base = float(np.clip(labels.mean() if len(labels) else 0.5, 0.01, 0.99))
        return {"kind": "constant", "prob": base}
    x = _logit(raw).reshape(-1, 1)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(x, labels)
    return {
        "kind": "platt",
        "intercept": float(clf.intercept_[0]),
        "coef": float(clf.coef_[0, 0]),
    }


def apply_platt_calibration(calibrator: dict[str, float], raw_pred_prob: Iterable[float]) -> np.ndarray:
    raw = _clip_prob(raw_pred_prob)
    if calibrator.get("kind") == "constant":
        return np.full(raw.shape, float(calibrator["prob"]))
    z = float(calibrator["intercept"]) + float(calibrator["coef"]) * _logit(raw)
    calibrated = 1.0 / (1.0 + np.exp(-z))
    return _clip_prob(calibrated)


def merge_calibrated_predictions(
    paired: pd.DataFrame,
    predictions: pd.DataFrame,
    method: str,
    pred_col: str = "calibrated_pred_prob",
) -> pd.DataFrame:
    """Attach calibrated probabilities to paired replay data and create proxy_{method}."""
    if pred_col not in predictions.columns:
        raise ValueError(f"{pred_col!r} not found in predictions.")
    pred = predictions[["record_id", pred_col]].copy()
    pred["record_id"] = pred["record_id"].astype(str)
    pred[pred_col] = pd.to_numeric(pred[pred_col], errors="coerce").clip(0.01, 0.99)
    pred = pred.dropna(subset=[pred_col]).drop_duplicates("record_id", keep="last")

    merged = paired.copy()
    merged["record_id"] = merged["record_id"].astype(str)
    merged = merged.merge(pred, on="record_id", how="left")
    missing = int(merged[pred_col].isna().sum())
    if missing:
        raise ValueError(f"{missing} paired rows missing calibrated predictions.")

    proxy_col = f"proxy_{method}"
    merged[proxy_col] = merged["offer_mp"].astype(float) * merged[pred_col].astype(float)
    return merged.drop(columns=[pred_col])
