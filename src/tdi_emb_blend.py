"""Compare base vs CheMeLeon-augmented TDI OOF probas + rank mixtures.

Both OOF files are in X_train-dedup row order (run_tdi.py convention); labels
rebuilt exactly as run_tdi.py does. Best MCC found by scanning positive-fraction
(top-k) cutoffs, which is equivalent to threshold-invariance.
Writes cache/tdi_emb_blend.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")

base = np.load(os.path.join(CACHE, "tdi_oof.npz"))
emb = np.load(os.path.join(CACHE, "tdi_emb_oof.npz"))
tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
lab = (pd.concat([tdi[["SMILES", "CYP2D6_is_TDI", "CYP3A4_is_TDI"]],
                  emx[["SMILES", "CYP2D6_is_TDI", "CYP3A4_is_TDI"]]])
       .drop_duplicates("SMILES").set_index("SMILES").reindex(Xtr["SMILES"]))


def mcc(p, y, t):
    pr = (p >= t).astype(int)
    tp = ((pr == 1) & (y == 1)).sum()
    fp = ((pr == 1) & (y == 0)).sum()
    tn = ((pr == 0) & (y == 0)).sum()
    fn = ((pr == 0) & (y == 1)).sum()
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def best_by_frac(z, yy):
    n = len(yy)
    rows = []
    for f in np.arange(0.05, 0.61, 0.01):
        k = int(n * f)
        t = np.sort(z)[::-1][k]
        rows.append((float(mcc(z, yy, t)), float(f)))
    return max(rows)


out = {}
for iso in ["CYP2D6", "CYP3A4"]:
    y = lab[f"{iso}_is_TDI"].values
    m = ~pd.isna(y)
    yy = np.zeros(len(y))
    yy[m] = y[m].astype(bool).astype(float)
    yy = yy[m].astype(int)
    pb, pe = base[iso][m], emb[iso][m]
    zb = (pb - pb.mean()) / pb.std()
    ze = (pe - pe.mean()) / pe.std()
    rows = {}
    for w in (1.0, 0.7, 0.5, 0.3, 0.0):
        z = w * zb + (1 - w) * ze
        b = best_by_frac(z, yy)
        rows[w] = {"best_mcc": b[0], "best_frac": b[1]}
    out[iso] = rows
    print(iso, json.dumps(rows))

with open(os.path.join(CACHE, "tdi_emb_blend.json"), "w") as fh:
    json.dump(out, fh, indent=2)
