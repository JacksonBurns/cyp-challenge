"""Blend TabICL-on-embeddings TDI OOF with the base classifier OOF.

Both files are in X_train-dedup row order; labels rebuilt as run_tdi.py does.
Scans z-space mixture weight x positive fraction; reports best MCC per iso
vs base-only best. Writes cache/tdi_tabicl_blend.json. Run in cyp env.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")

base = np.load(os.path.join(CACHE, "tdi_oof.npz"))
tab = np.load(os.path.join(CACHE, "tdi_tabicl_oof.npz"))
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
    yl = np.array([1 if bool(v) else 0 for v in y[m]])
    pb, pt = base[iso][m], tab[iso][m]
    rb, fb = best_by_frac(pb, yl)
    rt0, ft0 = best_by_frac(pt, yl)
    zb = (pb - pb.mean()) / pb.std()
    zt = (pt - pt.mean()) / pt.std()
    best = (rb, 1.0, fb)
    for w in np.arange(0, 1.01, 0.05):
        z = w * zb + (1 - w) * zt
        r, f = best_by_frac(z, yl)
        if r > best[0]:
            best = (r, float(w), f)
    out[iso] = {"base_mcc": rb, "base_frac": fb, "tabicl_mcc": rt0, "tabicl_frac": ft0,
                "blend_mcc": best[0], "w_base": best[1], "blend_frac": best[2]}
    print(iso, json.dumps(out[iso]), flush=True)

with open(os.path.join(CACHE, "tdi_tabicl_blend.json"), "w") as fh:
    json.dump(out, fh, indent=2)
