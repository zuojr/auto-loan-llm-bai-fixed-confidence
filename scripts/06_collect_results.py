#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default='baseline,qwen_local_icl,qwen_lora,gpt_api')
    args = ap.parse_args()
    rows=[]
    for method in [m.strip() for m in args.methods.split(',') if m.strip()]:
        summ_path = ROOT/'results'/f'run_summary_{method}.csv'
        par_path = ROOT/'results'/f'arm_params_{method}.csv'
        if not summ_path.exists():
            continue
        summ = pd.read_csv(summ_path)
        if method == 'baseline':
            row = {'method': 'Reward-only EGE', 'proxy': 'none', 'rho_hist': None, 'rho_imp': None}
            b = summ[summ['estimator']=='baseline'].iloc[0]
            row['pulls_to_90pct'] = b['pulls_to_90pct']
            row['accuracy_at_100'] = b['accuracy_at_100']
        else:
            params = pd.read_csv(par_path)
            row = {'method': method, 'proxy': method, 'rho_hist': float(params.loc[0,'rho']), 'rho_imp': float(params.loc[1,'rho'])}
            lcb = summ[summ['estimator']=='lcb_cv'].iloc[0]
            row['pulls_to_90pct'] = lcb['pulls_to_90pct']
            row['accuracy_at_100'] = lcb['accuracy_at_100']
            # Also include oracle for reference.
            oracle = summ[summ['estimator']=='oracle_cv'].iloc[0]
            row['oracle_pulls_to_90pct'] = oracle['pulls_to_90pct']
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT/'results/final_comparison_table.csv', index=False)
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()
