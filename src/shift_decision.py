"""Stage 3 (batch shift) and stage 4 (when can a lifetime test stop at cycle 100?).

Stage 3: the secondary batch was cycled about a year after the others. Compare intervals for it from
  (a) a model trained on the 41 training cells only, and
  (b) the same procedure retrained once the 43 primary-test cells have finished (84 cells),
  which is what a lab would do in practice.

Stage 4: a screening decision. A lab wants to know whether each cell lasts at least L* cycles.
  Rule 'interval': stop testing at cycle 100 only if the 90% jackknife+ interval lies wholly above or
  below L*; otherwise keep testing to end of life (where the answer is known exactly).
  Rule 'point'  : stop every test at cycle 100 and trust the point prediction.
  Reported: share of cells stopped early, wrong calls, and the share of total test cycles saved.
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, 'src')
from baseline import table
from intervals import intervals, coverage_table


def decision(life, pred, lo, hi, L):
    stop = (lo >= L) | (hi < L)
    wrong = stop & ((lo >= L) != (life >= L))
    saved = np.where(stop, life - 100, 0).sum() / life.sum()
    wrong_point = ((pred >= L) != (life >= L))
    return dict(threshold=L, n=len(life), stopped=int(stop.sum()), stopped_pct=100 * stop.mean(),
                wrong_calls=int(wrong.sum()), wrong_pct_of_stopped=100 * wrong.sum() / max(stop.sum(), 1),
                cycles_saved_pct=100 * saved,
                point_rule_wrong=int(wrong_point.sum()), point_rule_wrong_pct=100 * wrong_point.mean(),
                point_rule_saved_pct=100 * (life - 100).sum() / life.sum())


def main():
    F = table()
    # ---- stage 3
    iv_a = intervals(F, (F.split == 'train').values)
    iv_b = intervals(F, F.split.isin(['train', 'primary']).values)
    A = coverage_table(F, iv_a, ['secondary']).assign(trained_on='train (41)')
    B = coverage_table(F, iv_b, ['secondary']).assign(trained_on='train+primary (84)')
    S = pd.concat([A, B])
    S = S[S.method != 'naive'].reset_index(drop=True) if False else S
    sec = (F.split == 'secondary').values
    S['n'] = sec.sum()
    S['binom_p'] = [binomtest(int(round(r.coverage * r.n / 100)), int(r.n), r.level).pvalue for r in S.itertuples()]
    S.to_csv('results/shift_secondary.csv', index=False)
    pd.set_option('display.width', 220)
    print(S.round(3).to_string(index=False))
    err = lambda iv: float(100 * np.mean(np.abs(iv['pred'][sec] - F.life.values[sec]) / F.life.values[sec]))
    print('secondary MAPE: train-only %.1f%%, retrained %.1f%%' % (err(iv_a), err(iv_b)))

    # ---- stage 4: model trained on training cells, applied to both test sets (83 cells)
    te = F.split.isin(['primary', 'secondary']).values
    life = F.life.values[te]
    lo, hi = iv_a['jackknife+_0.9']
    rows = [decision(life, iv_a['pred'][te], lo[te], hi[te], L) for L in [600, 800, 1000]]
    D = pd.DataFrame(rows)
    D.to_csv('results/decision.csv', index=False)
    print(D.round(1).to_string(index=False))
    json.dump(dict(secondary_mape_train_only=err(iv_a), secondary_mape_retrained=err(iv_b)),
              open('results/shift_summary.json', 'w'), indent=1)
    P = F[['name', 'split', 'life']].copy()
    P['pred_train'], P['lo90_train'], P['hi90_train'] = iv_a['pred'], *iv_a['jackknife+_0.9']
    P['pred_retrained'], P['lo90_retrained'], P['hi90_retrained'] = iv_b['pred'], *iv_b['jackknife+_0.9']
    P.to_csv('results/intervals_final.csv', index=False)


if __name__ == '__main__':
    main()
