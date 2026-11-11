"""Load the compact Severson et al. (2019) extract and apply the standard cleaning and split.

Cleaning follows the published data-loading notes for this dataset:
- batch 1: drop 5 cells that never reached 80% capacity (b1c8, b1c10, b1c12, b1c13, b1c22)
- batch 2: cells b2c7, b2c8, b2c9, b2c15, b2c16 are continuations of b1c0-b1c4; their cycles are
  added to those cells' lifetimes and the duplicates are dropped
- batch 3: drop 6 cells with noisy channels or no end of life (b3c2, b3c23, b3c32, b3c37, b3c42, b3c43)
That leaves 41 + 43 + 40 = 124 cells.

Split (as in the original study): batches 1+2 in order, odd positions -> training (41 cells),
even positions plus the last cell -> primary test (43); batch 3 -> secondary test (40), cycled later.

Index convention: position j in the per-cycle arrays is cycle j+1 (summary and Qdlin alike).
"""
import numpy as np

DATA = 'data/battery_first100.npz'
DROP = {'b1': [8, 10, 12, 13, 22], 'b2': [7, 8, 9, 15, 16], 'b3': [2, 23, 32, 37, 42, 43]}
CONTINUATION = {0: 7, 1: 8, 2: 9, 3: 15, 4: 16}   # b1 cell -> b2 cell that continues it
SUMMARY_KEYS = ['cycle', 'QDischarge', 'QCharge', 'IR', 'Tavg', 'Tmax', 'Tmin', 'chargetime']


def load(path=DATA):
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}   # decompress each array once
    batch, index = d['batch'], d['index']
    pos = {(b, int(i)): k for k, (b, i) in enumerate(zip(batch, index))}
    cells = []
    for b in ['b1', 'b2', 'b3']:
        for i in sorted(int(x) for x in index[batch == b]):
            if i in DROP[b]:
                continue
            k = pos[(b, i)]
            life = float(d['cycle_life'][k])
            if b == 'b1' and i in CONTINUATION:
                life += len(d['summary_QDischarge'][pos[('b2', CONTINUATION[i])]])
            cells.append(dict(
                name=f'{b}c{i}', batch=b, policy=str(d['policy'][k]), life=life,
                Qdlin=d['Qdlin'][k], Tdlin=d['Tdlin'][k],
                **{k2: np.asarray(d['summary_' + k2][k], float) for k2 in SUMMARY_KEYS}))
    n1 = sum(c['batch'] == 'b1' for c in cells)
    n2 = sum(c['batch'] == 'b2' for c in cells)
    n12 = n1 + n2
    split = np.array(['secondary'] * len(cells), dtype=object)
    split[np.arange(1, n12 - 1, 2)] = 'train'
    split[np.r_[np.arange(0, n12, 2), n12 - 1]] = 'primary'
    for c, s in zip(cells, split):
        c['split'] = s
    return cells


if __name__ == '__main__':
    cells = load()
    lives = np.array([c['life'] for c in cells])
    print(len(cells), 'cells;', {s: sum(c['split'] == s for c in cells) for s in ['train', 'primary', 'secondary']})
    print('cycle life %.0f-%.0f, median %.0f' % (lives.min(), lives.max(), np.median(lives)))
    print('shortest:', min(cells, key=lambda c: c['life'])['name'], 'split', min(cells, key=lambda c: c['life'])['split'])
