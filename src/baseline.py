"""Stage 1: reproduce the three Severson et al. (2019) models.

Target = log10(cycle life). The one-feature 'variance' model is ordinary least squares; the other two
are elastic nets with the penalty chosen by cross-validation on the training cells only.
Errors: RMSE in cycles and mean absolute percentage error, per split. The primary test set is also
reported without its one outlier cell (148 cycles), as in the original paper.
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, 'src')
from data import load
from features import features, MODELS


def table():
    cells = load()
    F = pd.DataFrame([features(c) for c in cells])
    F['name'] = [c['name'] for c in cells]
    F['split'] = [c['split'] for c in cells]
    F['life'] = [c['life'] for c in cells]
    F['policy'] = [c['policy'] for c in cells]
    return F


def make_model(name):
    if name == 'variance':
        return make_pipeline(StandardScaler(), LinearRegression())
    return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=4,
                                                        alphas=100, max_iter=100000, random_state=0))


def errors(y, yhat):
    return dict(rmse=float(np.sqrt(np.mean((yhat - y) ** 2))), mape=float(100 * np.mean(np.abs(yhat - y) / y)))


def main():
    F = table()
    assert not F[sum(MODELS.values(), [])].isna().any().any(), 'missing feature values'
    tr = F.split == 'train'
    out, preds = {}, {}
    for name, cols in MODELS.items():
        m = make_model(name).fit(F.loc[tr, cols], np.log10(F.loc[tr, 'life']))
        p = 10 ** m.predict(F[cols])
        preds[name] = p
        res = {}
        for s in ['train', 'primary', 'secondary']:
            k = F.split == s
            res[s] = errors(F.life[k].values, p[k.values])
        k = (F.split == 'primary') & (F.life > 148)
        res['primary_no_outlier'] = errors(F.life[k].values, p[k.values])
        out[name] = res
    pd.set_option('display.width', 200)
    rows = [{'model': n, **{f'{s} RMSE': round(r[s]['rmse']) for s in r}, **{f'{s} %': round(r[s]['mape'], 1) for s in r}}
            for n, r in out.items()]
    print(pd.DataFrame(rows).to_string(index=False))
    json.dump(out, open('results/baseline_errors.json', 'w'), indent=1)
    P = F[['name', 'split', 'policy', 'life']].copy()
    for n, p in preds.items():
        P['pred_' + n] = p
    P.to_csv('results/baseline_predictions.csv', index=False)
    F.to_csv('results/features.csv', index=False)


if __name__ == '__main__':
    main()
