"""Per-pi E[MCC]/E[acc] sensitivity for candidate (variant, fraction) cells."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
CACHE = 'cache'
from tdi_fraction_opt import rank_conditional_curve, rescale_to_pi, mcc_counts

ISO = ["CYP2D6", "CYP3A4"]
rng = np.random.default_rng(11)
Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
tdi = pd.read_csv("data/cyp-challenge-TRAIN_TDI.csv")
emx = pd.read_csv("data/cyp-challenge-TRAIN_Emax.csv")
lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                 emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
Y = lab.reindex(Xtr["SMILES"])
base = np.load("cache/tdi_oof.npz")
emb = np.load("cache/tdi_emb_oof.npz")
tab = np.load("cache/tdi_tabicl_oof.npz")


def z(v):
    return (v - v.mean()) / v.std()


def emcc_eacc(pr, f, nt=1500):
    n = len(pr)
    k = max(1, min(n - 1, int(round(f * n))))
    tm = ta = 0.0
    for _ in range(nt):
        yy = (rng.random(n) < pr).astype(int)
        tp = int(yy[:k].sum()); fp = k - tp
        P = int(yy.sum()); fn = P - tp; tn = n - k - fn
        tm += mcc_counts(tp, fp, tn, fn)
        ta += (tp + tn) / n
    return tm / nt, ta / nt


W_ST = {"CYP2D6": (0.9, 0.1, 0.0), "CYP3A4": (0.6, 0.2, 0.2)}
PIS = {"CYP2D6": [0.16, 0.216, 0.26, 0.30, 0.34], "CYP3A4": [0.16, 0.213, 0.25, 0.30, 0.35]}
FRS = {"CYP2D6": [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.33, 0.36],
       "CYP3A4": [0.20, 0.25, 0.28, 0.30, 0.33, 0.36, 0.40, 0.45]}
for iso in ISO:
    y = Y[f"{iso}_is_TDI"].values
    m = ~pd.isna(y)
    yl = np.array([1 if bool(v) else 0 for v in y[m]], dtype=int)
    a, b, c = W_ST[iso]
    variants = {"v2_static": a * z(base[iso][m]) + b * z(emb[iso][m]) + c * z(tab[iso][m])}
    for nm, fn in [("v2_cap", "tdi_fbnestcap_v2_{iso}.npy"),
                   ("v3_nested", "tdi_fbnest_v3_{iso}.npy"),
                   ("v3_cap", "tdi_fbnestcap_v3_{iso}.npy")]:
        p = os.path.join(CACHE, fn.format(iso=iso))
        if os.path.exists(p):
            variants[nm] = np.load(p)
    print(f"\n### {iso}  rows: E[MCC]/E[acc]; labeled n={m.sum()}, train rate={yl.mean():.3f}")
    for nm, p in variants.items():
        rates, _ = rank_conditional_curve(p, yl)
        print(f"-- {nm}")
        print("     f: " + " ".join(f"{f:>12.2f}" for f in FRS[iso]))
        for pi in PIS[iso]:
            pr = rescale_to_pi(rates, pi)
            cells = [emcc_eacc(pr, f) for f in FRS[iso]]
            print(f"  pi={pi}: " + " ".join(f"{c[0]:.3f}/{c[1]:.3f}" for c in cells))
