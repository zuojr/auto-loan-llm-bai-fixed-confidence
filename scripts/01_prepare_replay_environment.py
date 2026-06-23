#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.auto_loan_utils import (
    load_chronological_split, train_environment_acceptance_model, predict_acceptance,
    build_two_policy_pool, policy_summary_from_environment, make_prompt_for_acceptance,
)
from src.config import TRAIN_N, EVAL_N, TARGET, RANDOM_SEED


TABPFN_TRAIN_COLUMNS = [
    TARGET,
    'Primary_FICO', 'Tier', 'Term', 'Amount_Approved', 'Competition_rate',
    'offer_mp', 'offer_ratio', 'partnerbin', 'CarType_id',
    'days', 'weeks', 'months', 'termclass',
    'State', 'Type', 'CarType',
]


def make_balanced_examples(train: pd.DataFrame, out_csv: Path, n_pos=80, n_neg=80, seed=20260610):
    rng = np.random.default_rng(seed)
    pos = train[train[TARGET] == 1]
    neg = train[train[TARGET] == 0]
    pos_s = pos.sample(n=min(n_pos, len(pos)), random_state=seed)
    neg_s = neg.sample(n=min(n_neg, len(neg)), random_state=seed + 1)
    ex = pd.concat([pos_s, neg_s], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    # Compact columns only; no hidden replay probabilities.
    keep = ['Tier','Primary_FICO','State','Type','Term','Amount_Approved','Competition_rate','CarType','partnerbin','mp','mp_rto_amtfinance', TARGET]
    ex[keep].to_csv(out_csv, index=False)


def make_historical_train_rows(train: pd.DataFrame, out_csv: Path) -> None:
    """Write historical labeled rows for optional structured TabPFN proxy training.

    This file intentionally contains only historical-block labels and structured
    application-offer features. It does not include replay-environment columns,
    row identifiers, natural-language prompt text, realized rewards, or proxy
    columns.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    hist = train.copy()
    hist['offer_mp'] = hist['mp'].astype(float)
    hist['offer_ratio'] = hist['mp_rto_amtfinance'].astype(float)
    missing = [c for c in TABPFN_TRAIN_COLUMNS if c not in hist.columns]
    if missing:
        raise ValueError(f"Historical train rows missing required TabPFN columns: {missing}")
    hist[TABPFN_TRAIN_COLUMNS].to_csv(out_csv, index=False)


def make_api_scoring_files(paired: pd.DataFrame, out_dir: Path, chunk_size=8000):
    out_dir.mkdir(parents=True, exist_ok=True)
    scoring_cols = [
        'record_id','base_row_id','arm','policy','multiplier','offer_mp','offer_ratio',
        'Tier','Primary_FICO','State','Type','Term','Amount_Approved','Competition_rate',
        'CarType','partnerbin','CarType_id','days','weeks','months','termclass','profile_text'
    ]
    scoring = paired[scoring_cols].copy()
    scoring.to_csv(out_dir/'scoring_full_paired.csv', index=False)
    scoring.head(1000).to_csv(out_dir/'scoring_pilot_1000.csv', index=False)
    manifest = []
    chunks = out_dir/'chunks'
    chunks.mkdir(parents=True, exist_ok=True)
    for k, start in enumerate(range(0, len(scoring), chunk_size), start=1):
        part = scoring.iloc[start:start+chunk_size].copy()
        path = chunks/f'scoring_chunk_{k:02d}.csv'
        part.to_csv(path, index=False)
        manifest.append({'chunk': k, 'path': str(path.relative_to(out_dir)), 'n_rows': len(part), 'start': start, 'end_exclusive': start+len(part)})
    pd.DataFrame(manifest).to_csv(out_dir/'scoring_manifest.csv', index=False)


def make_qwen_jsonl_from_df(df: pd.DataFrame, out_path: Path, label_col: str, use_soft_targets=True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for _, r in df.iterrows():
            # Historical train uses original offer columns; normalize to scoring-like row.
            row = r.copy()
            if 'offer_mp' not in row:
                row['offer_mp'] = row['mp']
                row['offer_ratio'] = row['mp_rto_amtfinance']
                row['multiplier'] = 1.0
                row['policy'] = 'historical_rule'
            from src.auto_loan_utils import make_profile_text
            row['profile_text'] = make_profile_text(row)
            prompt = make_prompt_for_acceptance(row, include_policy=True)
            lab = int(row[label_col])
            prob = 0.95 if lab == 1 else 0.05 if use_soft_targets else lab
            obj = {
                'messages': [
                    {'role': 'system', 'content': 'You are a calibrated auto-loan offer acceptance predictor. Return only valid JSON.'},
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': json.dumps({'pred_prob': prob})},
                ]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(ROOT/'data/raw/CPRM_AutoLOan_OnlineAutoLoanData.csv'))
    ap.add_argument('--out', default=str(ROOT))
    ap.add_argument('--chunk-size', type=int, default=8000)
    args = ap.parse_args()
    out = Path(args.out)
    for d in ['processed','results','models','api_scoring','api_scoring/chunks','qwen_finetune']:
        (out/d).mkdir(parents=True, exist_ok=True)

    df, train, eval_pool = load_chronological_split(args.csv, TRAIN_N, EVAL_N)
    model = train_environment_acceptance_model(train, RANDOM_SEED)
    joblib.dump(model, out/'models/replay_demand_model.joblib')

    train_pred = predict_acceptance(model, train)
    eval_pred = predict_acceptance(model, eval_pool)
    diagnostics = {
        'n_total': int(len(df)),
        'n_train': int(len(train)),
        'n_eval': int(len(eval_pool)),
        'train_apply_rate': float(train[TARGET].mean()),
        'eval_apply_rate': float(eval_pool[TARGET].mean()),
        'train_auc': float(roc_auc_score(train[TARGET], train_pred)),
        'eval_auc': float(roc_auc_score(eval_pool[TARGET], eval_pred)),
        'train_brier': float(brier_score_loss(train[TARGET], train_pred)),
        'eval_brier': float(brier_score_loss(eval_pool[TARGET], eval_pred)),
        'note': 'This fitted demand model defines the replay environment. It is not a reported proxy method.',
    }
    (out/'results/replay_environment_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))

    paired = build_two_policy_pool(model, eval_pool)
    paired.to_csv(out/'processed/paired_policy_eval_pool.csv', index=False)
    make_historical_train_rows(train, out/'processed/historical_train_rows.csv')
    policy_summary_from_environment(paired).to_csv(out/'results/policy_summary.csv', index=False)
    make_api_scoring_files(paired, out/'api_scoring', chunk_size=args.chunk_size)

    # Prompt files and calibration artifacts.
    make_balanced_examples(train, out/'api_scoring/calibration_examples_balanced.csv')
    global_stats = {
        'train_rows': int(len(train)),
        'train_apply_rate': float(train[TARGET].mean()),
        'train_mean_mp': float(train['mp'].mean()),
        'train_mean_offer_ratio': float(train['mp_rto_amtfinance'].mean()),
        'train_mean_fico': float(train['Primary_FICO'].mean()),
        'instruction': 'Use calibration examples as patterns, but anchor probabilities to the global apply rate.'
    }
    (out/'api_scoring/calibration_global_stats.json').write_text(json.dumps(global_stats, indent=2))
    prompt = '''You are a calibrated auto-loan offer acceptance predictor.

You will receive application profiles and offered loan contracts. For each row, estimate the probability that the customer accepts the offered contract.

The target is:
pred_prob = P(apply = 1 | application profile, offered contract).

Important:
- This is an acceptance-probability task, not a default-risk task.
- Do not decide which pricing policy is better.
- Do not optimize revenue directly.
- Estimate only the probability that this specific customer accepts this specific offered contract.
- The final proxy reward will be computed later as offered margin price multiplied by your predicted acceptance probability.
- Return only the requested JSON or CSV output.
'''
    (out/'api_scoring/llm_scoring_system_prompt.md').write_text(prompt)

    # Qwen fine-tune datasets from chronological historical block.
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(train))
    rng.shuffle(idx)
    val_idx = idx[:5000]
    train_idx = idx[5000:]
    train_ft = train.iloc[train_idx].reset_index(drop=True)
    val_ft = train.iloc[val_idx].reset_index(drop=True)
    make_qwen_jsonl_from_df(train_ft, out/'qwen_finetune/train_full.jsonl', TARGET)
    make_qwen_jsonl_from_df(val_ft, out/'qwen_finetune/val_full.jsonl', TARGET)
    # Balanced optional train set.
    pos = train[train[TARGET] == 1]
    neg = train[train[TARGET] == 0]
    n = min(len(pos), len(neg), 8000)
    balanced = pd.concat([pos.sample(n=n, random_state=1), neg.sample(n=n, random_state=2)], ignore_index=True).sample(frac=1, random_state=3)
    make_qwen_jsonl_from_df(balanced, out/'qwen_finetune/train_balanced.jsonl', TARGET)
    print('Prepared replay environment, API scoring files, and Qwen fine-tune data under', out)
    print(pd.read_csv(out/'results/policy_summary.csv').to_string(index=False))

if __name__ == '__main__':
    main()
