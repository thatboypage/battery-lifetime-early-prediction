import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
sys.path.insert(0, 'src')
from data import load
from physics import v_at_q

BLUE, ORANGE, GRAY, LGRAY, INK, INK2, RED, GREEN = '#2a78d6', '#eb6834', '#8a8984', '#c9c8c2', '#0b0b0b', '#52514e', '#c43c3c', '#3f9b62'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': LGRAY, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#ecebe7', 'grid.linewidth': 0.8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
BCOL = {'b1': BLUE, 'b2': GREEN, 'b3': ORANGE}
BLAB = {'b1': 'batch 1', 'b2': 'batch 2', 'b3': 'batch 3 (cycled ~1 year later)'}

# ---- Fig 4: voltage fade at equal charge, and how little of it is ohmic
cells = load()
qg = np.linspace(0.05, 1.0, 96)
dV = np.array([(v_at_q(c['Qdlin'][9], qg) - v_at_q(c['Qdlin'][99], qg)) * 1000 for c in cells])
life = np.array([c['life'] for c in cells])
P = pd.read_csv('results/physics_fits.csv')
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for (lo, hi), col, lab in [((0, 550), RED, 'life under 550 cycles'), ((550, 1000), ORANGE, '550-1000 cycles'), ((1000, 9e9), BLUE, '1000 cycles or more')]:
    g = (life >= lo) & (life < hi)
    ax[0].plot(qg, np.nanmedian(dV[g], 0), color=col, lw=2, label=f'{lab} ({g.sum()} cells)')
ax[0].set_ylim(-2, 40); ax[0].set_xlim(0, 1.0)
ax[0].set_xlabel('Charge delivered during the 4C discharge (Ah)'); ax[0].set_ylabel('Voltage drop, cycle 10 to cycle 100 (mV)')
ax[0].legend(fontsize=8, frameon=False, loc='upper left')
ax[0].set_title('Short-lived cells sag more, at every point of the discharge', loc='left', fontsize=10, color=INK)
for b in ['b1', 'b2', 'b3']:
    d = P[P.name.str.startswith(b)]
    ax[1].scatter(d.life, d.flat_shift_mV, s=16, color=BCOL[b], label=BLAB[b], alpha=0.85)
ax[1].scatter(P.life, P.ohmic_mV, s=10, color=GRAY, marker='x', label='part explained by measured resistance rise')
ax[1].axhline(0, color=LGRAY, lw=1)
ax[1].set_xscale('log'); ax[1].set_xticks([200, 500, 1000, 2000], ['200', '500', '1000', '2000']); ax[1].xaxis.set_minor_formatter(NullFormatter())
ax[1].set_ylim(-3, 25)
ax[1].set_xlabel('Cycle life'); ax[1].set_ylabel('Voltage sag, cycles 10 to 100 (mV)')
ax[1].legend(fontsize=7.5, frameon=False, loc='upper right')
ax[1].set_title('The sag (median 4.9 mV) is not resistance (median 0.3 mV)', loc='left', fontsize=10, color=INK)
plt.tight_layout(); plt.savefig('figures/fig4_voltage_fade.png', dpi=160); plt.close()

# ---- Fig 5: particle stress per charging protocol vs cycle life
M = pd.read_csv('results/mechanics.csv')
S = json.load(open('results/mechanics_summary.json'))
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for b in ['b1', 'b2', 'b3']:
    d = M[M.batch == b]
    ax.scatter(d.peak_stress_MPa, d.life, s=18, color=BCOL[b], label=BLAB[b], alpha=0.85)
ax.set_yscale('log'); ax.set_yticks([200, 500, 1000, 2000], ['200', '500', '1000', '2000']); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(f"Peak hoop stress in graphite particles while charging (MPa; {S['stress_MPa_per_C']:.1f} MPa per C-rate)")
ax.set_ylabel('Cycle life')
r12 = S['spearman_vs_life']['batches 1-2']['peak_stress_MPa']['rho']; r3 = S['spearman_vs_life']['batch 3']['peak_stress_MPa']['rho']
ax.legend(fontsize=8, frameon=False, loc='lower left')
ax.set_title(f'Higher particle stress goes with shorter life in batch 3 (Spearman {r3:.2f}), not in batches 1-2 ({r12:.2f})',
             loc='left', fontsize=10, color=INK)
fig.text(0.01, 0.01, 'Linear-elastic diffusion-induced stress in a 5 µm graphite sphere. Absolute values are uncertain by a factor of 10-20\n'
         '(material constants); the ordering of protocols is not, because stress scales with C-rate.', fontsize=7.5, color=INK2)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig('figures/fig5_particle_stress.png', dpi=160); plt.close()
print('saved')
