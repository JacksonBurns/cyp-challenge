"""A4 gate: computed 3D/conformational physics features.

Per isoform GBM pIC50 MAE (KFold5 seed0):
  stack | stack+A4(12) | stack+A2+A4
A4 = strain/SASA/conformational features, computed deterministically
(ETKDG+MMFF) for every challenge molecule - no labels, test-computable.
Mechanism: desolvation penalty for polar/acidic floor rows; conformational
cost for 2D6 straddling pairs.
"""
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402

CACHE = "/home/jackson/cyp-challenge/cache"
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def mae_gate(X, iso, seed=0):
    y = H.targets[f"{iso}_pIC50_direct_inhibition"].values
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    kf = KFold(5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        g = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                          random_state=seed)
        g.fit(X[tr], y[tr])
        pred[te] = g.predict(X[te])
    return float(np.mean(np.abs(pred - y)))


def main():
    Xbase = H.Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    A2 = np.nan_to_num(np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy")),
                       nan=0.0)
    A4 = np.load(os.path.join(CACHE, "adme4", "adme4_feats_tr.npy"))
    A4 = np.nan_to_num(A4, nan=0.0)
    print("base", Xbase.shape, "A2", A2.shape, "A4", A4.shape, flush=True)
    for iso in ISO:
        m0 = mae_gate(Xbase, iso)
        m4 = mae_gate(np.hstack([Xbase, A4]), iso)
        ma = mae_gate(np.hstack([Xbase, A2]), iso)
        m2 = mae_gate(np.hstack([Xbase, A2, A4]), iso)
        print(f"{iso}: base {m0:.4f} +A4 +{m0-m4:.4f} "
              f"| A2 {m0-ma:.4f} A2+A4 {m0-m2:.4f} A4-on-A2 delta {ma-m2:+.4f}",
              flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
