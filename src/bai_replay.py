from __future__ import annotations
import numpy as np
import pandas as pd


def theoretical_params(g: pd.DataFrame, proxy_col: str | None):
    p = g['prob_accept'].to_numpy(float)
    offer = g['offer_mp'].to_numpy(float)
    mu = float(np.mean(offer * p))
    varx = float(np.mean((offer ** 2) * p) - mu ** 2)
    sx = float(np.sqrt(max(varx, 1e-12)))
    if proxy_col is None:
        return dict(mu=mu, varx=varx, sx=sx, my=np.nan, sy=np.nan, cov=np.nan, rho=np.nan, beta=np.nan)
    y = g[proxy_col].to_numpy(float)
    my = float(np.mean(y))
    vary = float(np.var(y))
    sy = float(np.sqrt(max(vary, 1e-12)))
    cov = float(np.mean((offer * p - mu) * (y - my)))
    rho = float(cov / (sx * sy)) if sx > 0 and sy > 0 else 0.0
    beta = float(cov / max(vary, 1e-12))
    return dict(mu=mu, varx=varx, sx=sx, my=my, sy=sy, cov=cov, rho=rho, beta=beta)


def simulate_fixed_budget(paired: pd.DataFrame, proxy_col: str | None = None, budgets=None,
                          reps: int = 3000, seed: int = 20260610,
                          rad_const: float = 1.5, eta: float = 0.05):
    if budgets is None:
        budgets = list(range(20, 401, 20))
    rng = np.random.default_rng(seed)
    arms = []
    params = []
    for arm in sorted(paired['arm'].unique()):
        g = paired[paired.arm == arm].reset_index(drop=True)
        arms.append(g)
        params.append(theoretical_params(g, proxy_col))
    true_best = int(np.argmax([p['mu'] for p in params]))
    rows = []
    for T in budgets:
        n = max(1, T // len(arms))
        succ = dict(baseline=0)
        if proxy_col is not None:
            succ.update(dict(oracle_cv=0, plugin_cv=0, lcb_cv=0))
        lcb_ratios_by_arm = [[] for _ in arms]
        for rep in range(reps):
            estimates = {k: [] for k in succ}
            for ai, g in enumerate(arms):
                idx = rng.integers(0, len(g), size=n)
                prob = g['prob_accept'].to_numpy(float)[idx]
                offer = g['offer_mp'].to_numpy(float)[idx]
                x = offer * rng.binomial(1, prob)
                estimates['baseline'].append(float(np.mean(x)))
                if proxy_col is not None:
                    y = g[proxy_col].to_numpy(float)[idx]
                    par = params[ai]
                    estimates['oracle_cv'].append(float(np.mean(x - par['beta'] * (y - par['my']))))
                    beta_hat = float(np.cov(x, y, ddof=0)[0, 1] / (np.var(y) + 1e-12))
                    beta_hat = float(np.clip(beta_hat, 0, 3))
                    estimates['plugin_cv'].append(float(np.mean(x - beta_hat * (y - par['my']))))
                    xstd = (x - par['mu']) / par['sx']
                    ystd = (y - par['my']) / par['sy'] if par['sy'] > 1e-12 else np.zeros_like(y)
                    rho_hat = float(np.mean(xstd * ystd))
                    rad = rad_const * (np.sqrt(np.log(2 / eta) / max(n, 1)) + np.log(2 / eta) / max(n, 1))
                    L = float(np.clip(rho_hat - rad, 0, 1))
                    beta_lcb = L * par['sx'] / par['sy'] if par['sy'] > 1e-12 else 0.0
                    estimates['lcb_cv'].append(float(np.mean(x - beta_lcb * (y - par['my']))))
                    lcb_ratios_by_arm[ai].append(float(L / max(par['rho'], 1e-12)) if par['rho'] and par['rho'] > 0 else 0.0)
            for k in succ:
                succ[k] += int(np.argmax(estimates[k]) == true_best)
        row = {'budget': T, 'samples_per_arm': n}
        row.update({k: succ[k] / reps for k in succ})
        if proxy_col is not None:
            for ai in range(len(arms)):
                name = 'hist' if ai == 0 else 'imp' if ai == 1 else f'arm{ai}'
                row[f'mean_lcb_ratio_{name}'] = float(np.mean(lcb_ratios_by_arm[ai])) if lcb_ratios_by_arm[ai] else np.nan
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(params)


def threshold(df: pd.DataFrame, col: str, level: float = 0.90):
    hit = df.loc[df[col] >= level, 'budget']
    return int(hit.iloc[0]) if len(hit) else None
