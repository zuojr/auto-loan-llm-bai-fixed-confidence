#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.bai_replay import simulate_fixed_budget, threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(ROOT/'processed/paired_policy_eval_pool.csv'))
    ap.add_argument('--proxy-col', default=None, help='Proxy column. Omit for baseline-only.')
    ap.add_argument('--method', default='baseline')
    ap.add_argument('--reps', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=20260610)
    ap.add_argument('--budgets', default='20,40,60,80,100,120,140,160,180,200,240,280,320,360,400')
    args = ap.parse_args()
    base = ROOT
    paired = pd.read_csv(args.input)
    budgets = [int(x) for x in args.budgets.split(',') if x.strip()]
    if args.proxy_col and args.proxy_col not in paired.columns:
        raise ValueError(f'{args.proxy_col} not in {args.input}')
    acc, params = simulate_fixed_budget(paired, proxy_col=args.proxy_col, budgets=budgets, reps=args.reps, seed=args.seed)
    (base/'results').mkdir(exist_ok=True)
    (base/'figures').mkdir(exist_ok=True)
    acc_path = base/'results'/f'budget_accuracy_{args.method}.csv'
    par_path = base/'results'/f'arm_params_{args.method}.csv'
    acc.to_csv(acc_path, index=False)
    params.to_csv(par_path, index=False)
    rows=[]
    cols = ['baseline'] if args.proxy_col is None else ['baseline','oracle_cv','plugin_cv','lcb_cv']
    for col in cols:
        rows.append({
            'method': args.method,
            'estimator': col,
            'pulls_to_90pct': threshold(acc, col),
            'accuracy_at_100': float(acc.loc[acc.budget==100, col].iloc[0]) if any(acc.budget==100) else None,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(base/'results'/f'run_summary_{args.method}.csv', index=False)
    fig, ax = plt.subplots(figsize=(7,4.5))
    labels = {'baseline':'Reward-only EGE','lcb_cv':'Gen-CV-EGE','plugin_cv':'Plug-in CV','oracle_cv':'Oracle CV'}
    for col in cols:
        ax.plot(acc['budget'], acc[col], marker='o', label=labels.get(col,col))
    ax.axhline(0.9, linestyle='--', linewidth=1)
    ax.set_xlabel('Total online pulls')
    ax.set_ylabel('Probability of identifying improved pricing rule')
    ax.set_ylim(0.55,1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(base/'figures'/f'budget_accuracy_{args.method}.png', dpi=200)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(params.to_string(index=False))
    print('Wrote', acc_path, par_path)

if __name__ == '__main__':
    main()
