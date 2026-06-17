#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='CSV with record_id,pred_prob')
    ap.add_argument('--method', required=True, help='method name, e.g. qwen_api, gpt_api, qwen_ft')
    ap.add_argument('--base', default=str(ROOT))
    args = ap.parse_args()
    base = Path(args.base)
    paired = pd.read_csv(base/'processed/paired_policy_eval_pool.csv')
    pred = pd.read_csv(args.pred)
    pred['record_id'] = pred['record_id'].astype(str)
    pred['pred_prob'] = pd.to_numeric(pred['pred_prob'], errors='coerce').clip(0.01, 0.99)
    pred = pred.dropna(subset=['pred_prob']).drop_duplicates('record_id', keep='last')
    merged = paired.merge(pred[['record_id','pred_prob']], on='record_id', how='left')
    missing = int(merged['pred_prob'].isna().sum())
    if missing:
        raise ValueError(f'{missing} paired rows missing predictions. Complete scoring first or use --limit consistently for a pilot.')
    proxy_col = f'proxy_{args.method}'
    merged[proxy_col] = merged['offer_mp'].astype(float) * merged['pred_prob'].astype(float)
    out = base/'processed'/f'paired_policy_eval_pool_with_{args.method}.csv'
    merged.drop(columns=['pred_prob']).to_csv(out, index=False)
    print(f'Wrote {out}')
    print(f'Proxy column: {proxy_col}')

if __name__ == '__main__':
    main()
