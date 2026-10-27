"""ADME2 gates:
  A2: does the 18-endpoint ADME2 block improve GBM pIC50 MAE per isoform?
      (stack vs stack+ADME2, 5-fold CV, challenge labels)
  B2: does the ADME2 block improve floor-hazard AUC per isoform?
Evidence discipline: report deltas only (pool-level gates run separately
with shuffled-member controls).
"""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CACHE = H.CACHE


def gbmae(X, y, seed=0):
    kf = KFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in kf.split(X):
        g = LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                          verbose=-1, random_state=seed)
        g.fit(X[tr], y[tr])
        p[te] = g.predict(X[te])
    return np.mean(np.abs(p - y)), p


def floor_auc(X, hb, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(hb))
    for tr, te in skf.split(X, hb):
        g = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=seed)
        g.fit(X[tr], hb[tr])
        p[te] = g.predict_proba(X[te])[:, 1]
    return roc_auc_score(hb, p)


def main():
    smi = H.Xtr["SMILES"].values
    A2 = np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy"))
    Xbase = H.Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    Xa = np.hstack([Xbase, np.nan_to_num(A2, nan=0.0)])
    print("ADME2 block:", A2.shape)

    # A2: regression MAE
    for iso in ISO:
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        m = np.isfinite(y)
        b, _ = gbmae(Xa[m][:, :Xbase.shape[1]], y[m])
        a, _ = gbmae(Xa[m], y[m])
        print(f"A2 {iso}: base MAE {b:.4f} -> adme2 {a:.4f} delta {b-a:+.4f}", flush=True)

    # B2: floor hazard (label at per-iso min => FLOOR group, top tercile HARD;
    # target = is FLOOR)
    for iso in ISO:
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        m = np.isfinite(y)
        lo = np.nanmin(y)
        hb = (y <= lo + 1e-9).astype(int)
        if hb[m].sum() < 30:
            print(f"B2 {iso}: {int(hb[m].sum())} floor rows, skip")
            continue
        b = floor_auc(Xbase[m], hb[m])
        a = floor_auc(Xa[m], hb[m])
        print(f"B2 {iso}: base AUC {b:.3f} -> adme2 {a:.3f} delta {a-b:+.3f}", flush=True)


if __name__ == "__main__":
    main()
