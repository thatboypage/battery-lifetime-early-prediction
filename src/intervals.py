"""Stages 2-3: lifetime intervals and whether they survive batch shift.

All models predict log10(cycle life) from the six 'discharge' features (the best-generalising model
in the original study). Intervals are built in log space and mapped back to cycles.

Methods
  naive     : prediction +/- z * training residual SD (what most papers implicitly assume)
  bayes     : Bayesian ridge regression predictive interval (Gaussian, includes parameter uncertainty)
  jackknife+: distribution-free interval from leave-one-out refits on the 41 training cells
              (Barber et al., 2021); the whole fitting procedure, including the CV choice of penalty,
              is refit each time.

Coverage = share of test cells whose true lifetime falls inside their interval.
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, 'src')
from baseline import table, make_model
from features import MODELS

COLS = MODELS['discharge']
LEVELS = [0.8, 0.9]


def fit(X, y):
    return make_model('discharge').fit(X, y)


def jackknife_plus(Xtr, ytr, Xte, alpha):
    n = len(ytr)
    lo, hi = np.zeros((n, len(Xte))), np.zeros((n, len(Xte)))
    for i in range(n):
        keep = np.arange(n) != i
        m = fit(Xtr[keep], ytr[keep])
        r = abs(ytr[i] - m.predict(Xtr[i:i + 1])[0])
        mu = m.predict(Xte)
        lo[i], hi[i] = mu - r, mu + r
    k_lo = int(np.floor(alpha * (n + 1))) - 1
    k_hi = int(np.ceil((1 - alpha) * (n + 1))) - 1
    return np.sort(lo, 0)[max(k_lo, 0)], np.sort(hi, 0)[min(k_hi, n - 1)]


def intervals(F, train_mask):
    Xtr, ytr = F.loc[train_mask, COLS].values, np.log10(F.loc[train_mask, 'life'].values)
    X = F[COLS].values
    m = fit(Xtr, ytr)
    mu = m.predict(X)
    sd = np.std(ytr - m.predict(Xtr), ddof=len(COLS) + 1)
    br = make_pipeline(StandardScaler(), BayesianRidge()).fit(Xtr, ytr)
    bmu, bsd = br.predict(X, return_std=True)
    out = {'pred': 10 ** mu}
    for lev in LEVELS:
        z = norm.ppf(0.5 + lev / 2)
        out[f'naive_{lev}'] = (10 ** (mu - z * sd), 10 ** (mu + z * sd))
        out[f'bayes_{lev}'] = (10 ** (bmu - z * bsd), 10 ** (bmu + z * bsd))
        lo, hi = jackknife_plus(Xtr, ytr, X, 1 - lev)
        out[f'jackknife+_{lev}'] = (10 ** lo, 10 ** hi)
    return out


def coverage_table(F, iv, splits):
    rows = []
    for key, val in iv.items():
        if key == 'pred':
            continue
        method, lev = key.rsplit('_', 1)
        lo, hi = val
        for s in splits:
            k = (F.split == s).values
            inside = (F.life.values[k] >= lo[k]) & (F.life.values[k] <= hi[k])
            rows.append(dict(method=method, level=float(lev), split=s, coverage=100 * inside.mean(),
                             median_width=float(np.median(hi[k] - lo[k])),
                             median_rel_width=float(100 * np.median((hi[k] - lo[k]) / iv['pred'][k]))))
    return pd.DataFrame(rows)


def main():
    F = table()
    iv = intervals(F, (F.split == 'train').values)
    T = coverage_table(F, iv, ['primary', 'secondary'])
    T.to_csv('results/coverage_trained_on_train.csv', index=False)
    pd.set_option('display.width', 200)
    print(T.round(1).to_string(index=False))
    # residual bias by split: is the secondary batch shifted?
    r = np.log10(F.life.values) - np.log10(iv['pred'])
    bias = {s: dict(mean_log_residual=float(r[(F.split == s).values].mean()),
                    median_pct=float(100 * np.median(F.life.values[(F.split == s).values] / iv['pred'][(F.split == s).values] - 1)))
            for s in ['train', 'primary', 'secondary']}
    print(json.dumps(bias, indent=1))
    json.dump(bias, open('results/residual_bias.json', 'w'), indent=1)
    P = F[['name', 'split', 'life']].copy()
    P['pred'] = iv['pred']
    for key, (lo, hi) in iv.items() if False else [(k, v) for k, v in iv.items() if k != 'pred']:
        P[key + '_lo'], P[key + '_hi'] = lo, hi
    P.to_csv('results/intervals_trained_on_train.csv', index=False)


if __name__ == '__main__':
    main()
