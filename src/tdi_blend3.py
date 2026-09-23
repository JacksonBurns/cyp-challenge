"""3-way TDI blend scan: base GBM + emb-augmented GBM + TabICL-on-embeddings.

z-space weights on a coarse simplex grid; MCC at best fraction per iso.
Writes cache/tdi_blend3.json.
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")

base = np.load(os.path.join(CACHE, "tdi_oof.npz"))
emb = np.load(os.path.join(CACHE, "tdi_emb_oof.npz"))
tab = np.load(os.path.join(CACHE, "tdi_tabicl_oof.npz"))
tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
lab = (pd.concat([tdi[["SMILES", "CYP2D6_is_TDI", "CYP3A4_is_TDI"]],
                  emx[["SMILES", "CYP2D6_is_TDI", "CYP3A4_is_TDI"]]])
       .drop_duplicates("SMILES").set_index("SMILES").reindex(Xtr["SMILES"]))


def mcc(p, y, t):
    pr = (p >= t).astype(int)
    tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum()
    tn = ((pr == 0) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum()
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
    zs = {}
    for k, src in [("base", base), ("emb", emb), ("tab", tab)]:
        p = src[iso][m].astype(float)
        zs[k] = (p - p.mean()) / p.std()
    best = (-2, None, None)
    step = 0.1
    grid = [(a, b) for a in np.arange(0, 1.001, step)
            for b in np.arange(0, 1.001 - a + 1e-9, step)]
    for a, b in grid:
        c = 1 - a - b
        if c < -1e-9:
            continue
        z = a * zs["base"] + b * zs["emb"] + c * zs["tab"]
        r, f = best_by_frac(z, yl)
        if r > best[0]:
            best = (r, (round(a, 2), round(b, 2), round(c, 2)), f)
    rb, fb = best_by_frac(zs["base"], yl)
    out[iso] = {"base": [rb, fb], "best3": [best[0], list(best[1]), best[2]]}
    print(iso, json.dumps(out[iso]), flush=True)

with open(os.path.join(CACHE, "tdi_blend3.json"), "w") as fh:
    json.dump(out, fh, indent=2)
