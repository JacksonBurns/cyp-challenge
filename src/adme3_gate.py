"""A3 gate: does the experimental-pKa predicted block add on top of A2?

Per isoform, GBM pIC50 MAE (KFold 5, seed 0):
  stack | stack+A2(26) | stack+A2+A3(7) | stack+A3(7)
The A2+A3 minus A2 delta isolates the pKa contribution.
"""
import os
import sys

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def gbmae(X, y, seed=0):
    kf = KFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in kf.split(X):
        g = LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                          verbose=-1, random_state=seed)
        g.fit(X[tr], y[tr])
        p[te] = g.predict(X[te])
    return float(np.mean(np.abs(p - y)))


def main():
    smi = H.Xtr["SMILES"].values
    A2 = np.nan_to_num(np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy")), nan=0.0)
    V1 = np.nan_to_num(np.load(os.path.join(CACHE, "adme_aux", "adme_feats_tr.npy")), nan=0.0)
    A2 = np.hstack([A2, V1])
    A3 = np.load(os.path.join(CACHE, "adme3", "adme3_feats_tr.npy"))
    D3 = np.load(os.path.join(CACHE, "adme3", "derived_tr.npy"))
    A3 = np.hstack([np.nan_to_num(A3, nan=0.0), np.nan_to_num(D3, nan=0.0)])
    print("A2", A2.shape, "A3", A3.shape, flush=True)
    Xbase = H.Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    for iso in ISO:
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        m = np.isfinite(y)
        b = gbmae(Xbase[m], y[m])
        w2 = gbmae(np.hstack([Xbase, A2])[m], y[m])
        w23 = gbmae(np.hstack([Xbase, A2, A3])[m], y[m])
        w3 = gbmae(np.hstack([Xbase, A3])[m], y[m])
        print(f"{iso}: base {b:.4f} +A2 {w2:.4f} (+{b-w2:.4f}) "
              f"| +A2+A3 {w23:.4f} (A3 delta {w2-w23:+.4f}) | +A3 {w3:.4f} ({b-w3:+.4f})",
              flush=True)


if __name__ == "__main__":
    main()
