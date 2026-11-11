"""Mechanics step: diffusion-induced stress in graphite particles during each fast-charging protocol.

During charging, lithium enters each graphite particle through its surface. Because it diffuses inward
slowly, the surface swells more than the core, which loads the particle like a thermal-stress problem.
For a sphere of radius R under a constant surface flux J, the concentration settles (within about
R^2 / (20 D), ~30 s here) into a parabolic profile, c(r) - c_mean = (J R / 2D) [(r/R)^2 - 3/5].
Linear-elastic theory for a sphere (radial and hoop stress, as in a thermal-stress problem with
concentration in place of temperature) then gives a hoop stress at the surface of
    sigma_surface = - E * Omega * J * R / (15 (1 - nu) D)      (compressive while charging)
and the same magnitude in tension at the centre.

For a charge rate of n C, the particle's mean lithium fraction rises by n * dtheta per hour, so
    J = R * n * c_max * dtheta / (3 * 3600)   and   sigma = E Omega R^2 n c_max dtheta / (45 * 3600 * (1-nu) D).

Stress therefore scales linearly with C-rate and with R^2 / D. The material constants are typical
literature values for graphite and are uncertain by a factor of several, so absolute stresses are
indicative; comparisons between protocols are not affected (they depend only on the C-rates).

Each protocol "C1(Q1%)-C2" charges at C1 from 0 to Q1% state of charge, then at C2 to 80%
(then 1C to full, which is common to all cells).
"""
import re, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, 'src')
from data import load

PARAMS = dict(E=15e9, nu=0.3, Omega=3.65e-6, D=3.9e-14, R=5e-6, cmax=30555.0, dtheta=0.78)
LOW = dict(PARAMS, E=10e9, R=3e-6, D=1e-13)       # gentler end of the plausible range
HIGH = dict(PARAMS, E=32e9, R=8e-6, D=1e-14)      # harsher end


def stress_per_C(p):
    return p['E'] * p['Omega'] * p['R'] ** 2 * p['cmax'] * p['dtheta'] / (45 * 3600 * (1 - p['nu']) * p['D'])


def parse(policy):
    m = re.match(r'([\d.]+)C\((\d+)%\)-([\d.]+)', policy)
    c1, q1, c2 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return c1, min(q1, 80.0), c2


def main():
    cells = load()
    s1 = stress_per_C(PARAMS)
    rows = []
    for c in cells:
        c1, q1, c2 = parse(c['policy'])
        t1, t2 = (q1 / 100) / c1 * 3600, ((80 - q1) / 100) / c2 * 3600 if q1 < 80 else 0.0
        peak = max(c1, c2 if q1 < 80 else c1)
        mean_c = (q1 * c1 + (80 - q1) * c2) / 80          # charge-weighted C-rate, 0-80% SOC
        rows.append(dict(name=c['name'], batch=c['batch'], split=c['split'], life=c['life'], policy=c['policy'],
                         C1=c1, Q1=q1, C2=c2, charge_time_min=(t1 + t2) / 60, peak_C=peak, mean_C=mean_c,
                         peak_stress_MPa=peak * s1 / 1e6, mean_stress_MPa=mean_c * s1 / 1e6,
                         stress_x_time=(c1 * t1 + c2 * t2) * s1 / 1e6 / 60))   # MPa-minutes above zero
    M = pd.DataFrame(rows)
    M.to_csv('results/mechanics.csv', index=False)
    out = dict(params=PARAMS, stress_MPa_per_C=s1 / 1e6, stress_MPa_per_C_low=stress_per_C(LOW) / 1e6,
               stress_MPa_per_C_high=stress_per_C(HIGH) / 1e6,
               settle_time_s=PARAMS['R'] ** 2 / (20 * PARAMS['D']),
               peak_stress_MPa_range=[float(M.peak_stress_MPa.min()), float(M.peak_stress_MPa.max())])
    corr = {}
    for grp, mask in [('batches 1-2', M.batch.isin(['b1', 'b2'])), ('batch 3', M.batch == 'b3'), ('all', M.batch.notna())]:
        corr[grp] = {v: dict(rho=float(spearmanr(M.loc[mask, v], M.loc[mask, 'life'])[0]),
                             p=float(spearmanr(M.loc[mask, v], M.loc[mask, 'life'])[1]))
                     for v in ['peak_stress_MPa', 'mean_stress_MPa', 'charge_time_min']}
    out['spearman_vs_life'] = corr
    json.dump(out, open('results/mechanics_summary.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
