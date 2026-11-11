import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

BLUE, ORANGE, GRAY, LGRAY, INK, INK2, RED, GREEN = '#2a78d6', '#eb6834', '#8a8984', '#c9c8c2', '#0b0b0b', '#52514e', '#c43c3c', '#3f9b62'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': LGRAY, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#ecebe7', 'grid.linewidth': 0.8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
P = pd.read_csv('results/intervals_final.csv')
L = 800

# ---- Fig 1: predicted vs actual with 90% intervals, primary and secondary test sets
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
for ax, s, ttl in [(axes[0], 'primary', 'Primary test (43 cells)'), (axes[1], 'secondary', 'Secondary test (40 cells, cycled ~1 year later)')]:
    d = P[P.split == s]
    inside = (d.life >= d.lo90_train) & (d.life <= d.hi90_train)
    for ok, col in [(True, BLUE), (False, RED)]:
        e = d[inside == ok]
        ax.errorbar(e.pred_train, e.life, xerr=[e.pred_train - e.lo90_train, e.hi90_train - e.pred_train], fmt='o', ms=4,
                    color=col, ecolor=col, elinewidth=0.8, alpha=0.85,
                    label=f'true life inside 90% interval ({inside.sum()})' if ok else f'outside ({(~inside).sum()})')
    ax.plot([100, 2400], [100, 2400], color=GRAY, lw=1, ls='--')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(120, 2600); ax.set_ylim(120, 2600)
    ticks = [200, 500, 1000, 2000]
    ax.set_xticks(ticks, [str(t) for t in ticks]); ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(ttl, loc='left', fontsize=10, color=INK)
    ax.set_xlabel('Predicted cycle life from the first 100 cycles')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
axes[0].set_ylabel('Actual cycle life')
fig.suptitle('Lifetime intervals from the first 100 cycles hold their 90% coverage on both test batches (91% and 88%)',
             x=0.01, ha='left', fontsize=10.5, color=INK)
plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig('figures/fig1_intervals.png', dpi=160); plt.close()

# ---- Fig 2: coverage, nominal vs achieved, secondary batch, before and after retraining
S = pd.read_csv('results/shift_secondary.csv')
S = S[S.level == 0.9]
lab = {'naive': 'Naive (±z·SD)', 'bayes': 'Bayesian ridge', 'jackknife+': 'Jackknife+'}
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(3); w = 0.36
for j, (tr, col) in enumerate([('train (41)', LGRAY), ('train+primary (84)', BLUE)]):
    v = [S[(S.method == m) & (S.trained_on == tr)].coverage.values[0] for m in lab]
    b = ax.bar(x + (j - 0.5) * w, v, w, color=col, label=f'trained on {tr} cells')
    for xi, vi in zip(x + (j - 0.5) * w, v):
        ax.text(xi, vi + 1, f'{vi:.1f}%', ha='center', fontsize=8.5, color=INK)
ax.axhline(90, color=RED, ls='--', lw=1); ax.text(2.62, 91, 'target 90%', color=RED, fontsize=8, ha='right')
ax.set_xticks(x, list(lab.values())); ax.set_ylim(50, 102); ax.set_ylabel('Secondary-batch cells inside interval (%)')
ax.grid(axis='x', visible=False)
ax.legend(fontsize=8, frameon=False, loc='lower right')
ax.set_title('On the later batch, retraining with the finished primary cells restores coverage', loc='left', fontsize=10.5, color=INK)
plt.tight_layout(); plt.savefig('figures/fig2_coverage_shift.png', dpi=160); plt.close()

# ---- Fig 3: screening decision at L* = 800 cycles
d = P[P.split != 'train'].sort_values('pred_train').reset_index(drop=True)
above, below = d.lo90_train >= L, d.hi90_train < L
stop = above | below
wrong = stop & (above != (d.life >= L))
fig, ax = plt.subplots(figsize=(10, 4.6))
y = np.arange(len(d))
for mask, col, lab in [(stop & ~wrong, GREEN, f'stop at cycle 100: interval clear of {L} ({(stop & ~wrong).sum()})'),
                       (~stop, LGRAY, f'keep testing: interval straddles {L} ({(~stop).sum()})'),
                       (wrong, RED, f'stopped, wrong call ({wrong.sum()})')]:
    e = d[mask]
    ax.vlines(e.index, e.lo90_train, e.hi90_train, color=col, lw=2.2, label=lab)
ax.scatter(y, d.life, s=10, color=INK, zorder=3, label='actual cycle life')
ax.axhline(L, color=ORANGE, lw=1.2, ls='--'); ax.text(1, L * 1.04, f'requirement: {L} cycles', color=ORANGE, fontsize=8.5)
ax.set_yscale('log'); ax.set_yticks([200, 500, 1000, 2000], ['200', '500', '1000', '2000']); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Test cells (primary + secondary), ordered by predicted life'); ax.set_ylabel('Cycle life (90% interval)')
ax.grid(axis='x', visible=False); ax.set_xlim(-1, len(d))
ax.legend(fontsize=8, frameon=False, loc='upper left')
ax.set_title(f'Stopping only when the interval is clear: 53 of 83 tests end at cycle 100, with 1 wrong call',
             loc='left', fontsize=10.5, color=INK)
plt.tight_layout(); plt.savefig('figures/fig3_decision.png', dpi=160); plt.close()
print('figures saved')
