import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.qwen_lora_cls_utils import (
    apply_platt_calibration,
    convert_numeric_jsonl_to_yes_no,
    fit_platt_calibrator,
    make_binary_acceptance_prompt,
    merge_calibrated_predictions,
    yes_no_probability_from_logits,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_convert_numeric_jsonl_to_yes_no_preserves_prompts_and_maps_labels(tmp_path: Path):
    src = tmp_path / "numeric.jsonl"
    dst = tmp_path / "cls.jsonl"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "prompt A"},
                {"role": "assistant", "content": json.dumps({"pred_prob": 0.95})},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "prompt B"},
                {"role": "assistant", "content": json.dumps({"pred_prob": 0.05})},
            ]
        },
    ]
    _write_jsonl(src, rows)

    written = convert_numeric_jsonl_to_yes_no(src, dst)

    out = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
    assert written == 2
    assert out[0]["messages"][0]["role"] == "system"
    assert "Return only yes or no." in out[0]["messages"][0]["content"]
    assert "Return exactly one token: yes or no." in out[0]["messages"][1]["content"]
    assert "prompt A" in out[0]["messages"][1]["content"]
    assert out[0]["messages"][-1] == {"role": "assistant", "content": "yes"}
    assert out[1]["messages"][-1] == {"role": "assistant", "content": "no"}


def test_make_binary_acceptance_prompt_uses_same_profile_fields():
    row = {
        "policy": "historical_rule",
        "multiplier": 1.0,
        "profile_text": "This auto-loan application has Primary FICO 700.",
    }

    prompt = make_binary_acceptance_prompt(row)

    assert "Return exactly one token: yes or no." in prompt
    assert "Pricing rule: historical_rule" in prompt
    assert "Primary FICO 700" in prompt


def test_yes_no_probability_from_logits_normalizes_candidate_token_sets():
    logits = np.array([0.0, 2.0, 1.0, -3.0])

    prob = yes_no_probability_from_logits(logits, yes_token_ids=[1, 2], no_token_ids=[0])

    expected_yes_mass = math.exp(2.0) + math.exp(1.0)
    expected_no_mass = math.exp(0.0)
    assert math.isclose(prob, expected_yes_mass / (expected_yes_mass + expected_no_mass))


def test_platt_calibration_maps_raw_scores_to_valid_probabilities():
    raw = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1])

    calibrator = fit_platt_calibrator(raw, y)
    calibrated = apply_platt_calibration(calibrator, raw)

    assert np.all(calibrated >= 0.01)
    assert np.all(calibrated <= 0.99)
    assert calibrated[0] < calibrated[-1]
    assert calibrated[:3].mean() < calibrated[3:].mean()


def test_merge_calibrated_predictions_creates_proxy_without_leaking_raw_pred(tmp_path: Path):
    paired = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "offer_mp": [100.0, 200.0],
            "prob_accept": [0.1, 0.2],
        }
    )
    pred = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "calibrated_pred_prob": [0.25, 0.50],
        }
    )

    merged = merge_calibrated_predictions(paired, pred, method="qwen_lora_cls_calibrated")

    assert "calibrated_pred_prob" not in merged.columns
    assert merged["proxy_qwen_lora_cls_calibrated"].tolist() == [25.0, 100.0]
    assert merged["prob_accept"].tolist() == [0.1, 0.2]
