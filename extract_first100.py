"""Extract a compact copy of the Severson et al. (2019) battery data.

Run this on your own computer, in the folder that holds the three batch files
downloaded from https://data.matr.io/1/ :

    2017-05-12_batchdata_updated_struct_errorcorrect.mat
    2017-06-30_batchdata_updated_struct_errorcorrect.mat
    2018-04-12_batchdata_updated_struct_errorcorrect.mat

Usage:
    pip install h5py numpy
    python extract_first100.py

It writes one small file, battery_first100.npz (tens of MB), holding for every cell:
  - batch, charging policy, cycle life
  - per-cycle summary for the whole life: discharge capacity, internal resistance,
    average/max/min temperature, charge time
  - for cycles 1-100 only: discharge capacity vs voltage (Qdlin) and temperature (Tdlin),
    each interpolated on the dataset's fixed 1000-point voltage grid
Attach battery_first100.npz to the chat.
"""
import glob
import os
import sys

import h5py
import numpy as np

N_CYCLES = 100
SUMMARY_KEYS = ["cycle", "QDischarge", "QCharge", "IR", "Tavg", "Tmax", "Tmin", "chargetime"]


def read_str(f, ref):
    return "".join(chr(int(c)) for c in np.array(f[ref]).ravel())


def read_scalar(f, ref):
    a = np.array(f[ref]).ravel()
    return float(a[0]) if a.size else float("nan")


def extract(path, batch_label):
    out = []
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        n = batch["summary"].shape[0]
        for i in range(n):
            cl = read_scalar(f, batch["cycle_life"][i, 0])
            policy = read_str(f, batch["policy_readable"][i, 0])
            summ = f[batch["summary"][i, 0]]
            summary = {k: np.array(summ[k]).ravel().astype(np.float32) for k in SUMMARY_KEYS if k in summ}
            cycles = f[batch["cycles"][i, 0]]
            n_avail = cycles["Qdlin"].shape[0]
            m = min(N_CYCLES + 1, n_avail)  # index 0 is usually an empty formation entry
            qd = np.full((N_CYCLES + 1, 1000), np.nan, np.float32)
            td = np.full((N_CYCLES + 1, 1000), np.nan, np.float32)
            for j in range(m):
                q = np.array(f[cycles["Qdlin"][j, 0]]).ravel()
                t = np.array(f[cycles["Tdlin"][j, 0]]).ravel()
                if q.size == 1000:
                    qd[j] = q
                if t.size == 1000:
                    td[j] = t
            out.append(dict(batch=batch_label, index=i, policy=policy, cycle_life=cl,
                            summary=summary, Qdlin=qd, Tdlin=td))
            print(f"  {batch_label} cell {i + 1}/{n}: cycle life {cl:.0f}, {policy}")
    return out


def main():
    files = sorted(glob.glob("*batchdata*.mat"))
    if len(files) != 3:
        sys.exit(f"Expected the 3 batch .mat files in this folder, found {len(files)}: {files}")
    cells = []
    for k, p in enumerate(files, start=1):
        print(f"Reading {os.path.basename(p)} ...")
        cells += extract(p, f"b{k}")
    np.savez_compressed(
        "battery_first100.npz",
        batch=np.array([c["batch"] for c in cells]),
        index=np.array([c["index"] for c in cells]),
        policy=np.array([c["policy"] for c in cells]),
        cycle_life=np.array([c["cycle_life"] for c in cells], np.float32),
        Qdlin=np.stack([c["Qdlin"] for c in cells]),
        Tdlin=np.stack([c["Tdlin"] for c in cells]),
        **{f"summary_{k}": np.array([c["summary"].get(k, np.array([], np.float32)) for c in cells], dtype=object)
           for k in SUMMARY_KEYS},
    )
    size_mb = os.path.getsize("battery_first100.npz") / 1e6
    print(f"Wrote battery_first100.npz ({len(cells)} cells, {size_mb:.0f} MB). Attach it to the chat.")


if __name__ == "__main__":
    main()
