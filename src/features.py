"""Early-life features from the first 100 cycles, following the definitions in Severson et al. (2019).

dQ(V) = Q_100(V) - Q_10(V): the change in the discharge capacity-voltage curve between cycles 10 and 100.
Position j in each array is cycle j+1, so cycle 100 -> index 99, cycle 10 -> index 9.
"""
import numpy as np
from scipy.stats import skew, kurtosis

C = lambda n: n - 1   # cycle number -> array index


def dq(c):
    return c['Qdlin'][C(100)] - c['Qdlin'][C(10)]


def clean_capacity(qd):
    """Replace single-cycle logging glitches (physically impossible values for a 1.1 Ah cell)
    with the median of the neighbouring cycles. Two cells are affected (b1c0: 1.54 Ah, b1c18: 2.88 Ah)."""
    qd = qd.copy()
    for j in np.where((qd > 1.2) | ((qd < 0.8) & (np.arange(len(qd)) > 0)))[0]:
        nb = qd[max(j - 2, 0):j + 3]
        nb = nb[(nb <= 1.2) & (nb >= 0.8)]
        qd[j] = np.median(nb) if nb.size else np.nan
    return qd


def features(c):
    d = dq(c)
    qd = clean_capacity(c['QDischarge'])
    cyc = np.arange(2, 101)
    q2_100 = qd[C(2):C(100) + 1]
    slope, intercept = np.polyfit(cyc, q2_100, 1)
    ir = c['IR'][C(2):C(100) + 1]
    ir_pos = ir[ir > 0]
    return {
        # discharge-curve features
        'log_var_dQ': np.log10(abs(np.var(d))),
        'log_min_dQ': np.log10(abs(np.min(d))),
        'log_skew_dQ': np.log10(abs(skew(d))),
        'log_kurt_dQ': np.log10(abs(kurtosis(d, fisher=False))),
        # capacity-fade features
        'Q_cycle2': qd[C(2)],
        'Qmax_minus_Q2': np.max(q2_100) - qd[C(2)],
        'fade_slope': slope,
        'fade_intercept': intercept,
        # other
        'charge_time_5': np.mean(c['chargetime'][C(2):C(6)]),
        'temp_integral': np.sum(c['Tavg'][C(2):C(100) + 1]),
        'IR_min': np.min(ir_pos) if ir_pos.size else np.nan,
        'IR_100_minus_2': c['IR'][C(100)] - c['IR'][C(2)],
    }


MODELS = {
    'variance': ['log_var_dQ'],
    'discharge': ['log_min_dQ', 'log_var_dQ', 'log_skew_dQ', 'log_kurt_dQ', 'Q_cycle2', 'Qmax_minus_Q2'],
    'full': ['log_min_dQ', 'log_var_dQ', 'fade_slope', 'fade_intercept', 'Q_cycle2', 'charge_time_5',
             'temp_integral', 'IR_min', 'IR_100_minus_2'],
}
