"""Physics step: what does the early-life signal mean physically?

Measured quantity: voltage fade at equal delivered charge,
    dV(q) = V_cycle10(q) - V_cycle100(q),   on the 4C discharge (I = 4.4 A).

Three candidate mechanisms and the shape each predicts:
  1. Ohmic resistance growth: a flat shift dV = I * dR. dR is measured independently (cell internal resistance).
  2. Loss of active graphite area (loss of active material of the negative electrode). Charge-transfer
     overpotential follows Butler-Volmer, eta = (2RT/F) asinh(I / (2 * I0)), with exchange current
     I0 proportional to active area and to sqrt(x (1 - x)), where x is the graphite lithium fraction.
     Losing a fraction f of the area raises eta by
         d_eta(q) = (2RT/F) [asinh(k / (phi(x) (1 - f))) - asinh(k / phi(x))],   phi(x) = 2 sqrt(x(1-x)),
     which grows as the graphite empties (x -> 0) during discharge.
  3. Capacity loss (lithium inventory or cathode material): the end-of-discharge knee moves to lower charge,
     modelled as V100(q) = V10(q / (1 - s)).

Per cell, two parameters are fitted, f (area lost) and s (capacity lost); the ohmic part is fixed at the
measured resistance change. One global constant, k = I / (2 I0,max), is chosen by fit quality across cells.
The graphite lithium fraction runs linearly from x0 = 0.80 to x_end = 0.02 over the discharge (typical
LFP/graphite balancing; an assumption, stated in the README).
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, 'src')
from data import load

F_CONST, R_GAS = 96485.0, 8.314
I_DIS = 4.4                      # 4C discharge of a 1.1 Ah cell
VGRID = np.linspace(3.5, 2.0, 1000)
X0, XEND = 0.80, 0.02


def v_at_q(q, qg):
    return np.interp(qg, q, VGRID, left=np.nan, right=np.nan)


def d_eta(x, f, k, T):
    phi = 2 * np.sqrt(np.clip(x * (1 - x), 1e-6, None))
    return (2 * R_GAS * T / F_CONST) * (np.arcsinh(k / (phi * (1 - f))) - np.arcsinh(k / phi))


def fit_cell(c, k):
    q10, q100 = c['Qdlin'][9], c['Qdlin'][99]
    qend = q10[-1]
    qg = np.linspace(0.05, 0.95 * min(qend, q100[-1]), 120)
    v100 = v_at_q(q100, qg)
    T = 273.15 + np.nanmean(c['Tdlin'][99])
    dR = c['IR'][99] - c['IR'][9]
    x = X0 - (X0 - XEND) * qg / qend

    def model(p):
        f, s = p
        return v_at_q(q10, qg / (1 - s)) - d_eta(x, f, k, T) - I_DIS * dR

    def resid(p):
        r = model(p) - v100
        return np.nan_to_num(r, nan=0.05)

    sol = least_squares(resid, [0.02, 0.01], bounds=([0, -0.05], [0.9, 0.2]))
    dv_meas = v_at_q(q10, qg) - v100
    r = resid(sol.x)
    r2 = 1 - np.sum(r ** 2) / np.sum((dv_meas - dv_meas.mean()) ** 2)
    # comparison: flat shift + capacity loss (no kinetics)
    def resid_flat(p):
        return np.nan_to_num(v_at_q(q10, qg / (1 - p[1])) - p[0] - v100, nan=0.05)
    fl = least_squares(resid_flat, [0.005, 0.01])
    rf = resid_flat(fl.x)
    r2_flat = 1 - np.sum(rf ** 2) / np.sum((dv_meas - dv_meas.mean()) ** 2)
    return dict(name=c['name'], split=c['split'], life=c['life'], f_area_loss=sol.x[0], s_capacity_loss=sol.x[1],
                ohmic_mV=1000 * I_DIS * dR, r2_kinetic=r2, r2_flat=r2_flat, flat_shift_mV=1000 * fl.x[0],
                flat_capacity_loss=fl.x[1], dT_C=float(np.nanmean(c['Tdlin'][99]) - np.nanmean(c['Tdlin'][9])),
                dv_mid_mV=1000 * np.nanmedian(dv_meas[(qg > 0.2) & (qg < 0.6)]))


def main():
    cells = load()
    scores = {}
    for k in [0.5, 1, 2, 3, 5, 8]:
        scores[k] = np.median([fit_cell(c, k)['r2_kinetic'] for c in cells[::4]])
    k_best = max(scores, key=scores.get)
    print('median R2 by k (subset):', {k: round(v, 3) for k, v in scores.items()}, '-> k =', k_best)
    D = pd.DataFrame([fit_cell(c, k_best) for c in cells])
    D.to_csv('results/physics_fits.csv', index=False)
    from scipy.stats import spearmanr
    summ = dict(k=k_best,
                median_r2_kinetic=float(D.r2_kinetic.median()), median_r2_flat=float(D.r2_flat.median()),
                median_area_loss_pct=float(100 * D.f_area_loss.median()),
                median_capacity_loss_pct=float(100 * D.s_capacity_loss.median()),
                median_flat_shift_mV=float(D.flat_shift_mV.median()), median_ohmic_mV=float(D.ohmic_mV.median()),
                spearman_life={c: float(spearmanr(D[c], D.life)[0]) for c in
                               ['f_area_loss', 's_capacity_loss', 'ohmic_mV', 'flat_shift_mV']})
    # do two physical quantities predict lifetime? (non-ohmic voltage shift in mV, capacity loss)
    from sklearn.linear_model import LinearRegression
    tr = D.split == 'train'
    out = {}
    for name, cols in [('physics: shift + capacity loss', ['flat_shift_mV', 'flat_capacity_loss']),
                       ('physics: shift only', ['flat_shift_mV'])]:
        X = np.column_stack([np.log10(np.clip(D[c].values, 1e-4, None)) if c == 'flat_shift_mV' else D[c].values for c in cols])
        m = LinearRegression().fit(X[tr], np.log10(D.life[tr]))
        p = 10 ** m.predict(X)
        out[name] = {s_: float(100 * np.mean(np.abs(p[(D.split == s_).values] - D.life[(D.split == s_).values]) / D.life[(D.split == s_).values]))
                     for s_ in ['train', 'primary', 'secondary']}
        k = ((D.split == 'primary') & (D.life > 148)).values
        out[name]['primary_no_outlier'] = float(100 * np.mean(np.abs(p[k] - D.life.values[k]) / D.life.values[k]))
    summ['lifetime_mape_pct'] = out
    summ['median_dT_C'] = float(D.dT_C.median())
    summ['spearman_life']['flat_capacity_loss'] = float(spearmanr(D.flat_capacity_loss, D.life)[0])
    json.dump(summ, open('results/physics_summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
