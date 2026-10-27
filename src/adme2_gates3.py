"""A2+ gate: stack + ADME2 (18) + assay-interference alerts (~28).
GBM pIC50 MAE per isoform, 5-fold. Baselines: stack, stack+ADME2,
stack+ADME2+alerts. The alert delta isolates interference chemistry."""
import os
import sys

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

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
    return np.mean(np.abs(p - y))


def main():
    smi = H.Xtr["SMILES"].values
    A2 = np.nan_to_num(np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy")),
                       nan=0.0)
    AL = np.nan_to_num(np.load(os.path.join(CACHE, "adme2", "alerts_tr.npy")),
                       nan=0.0)
    Xbase = H.Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    Xa = np.hstack([Xbase, A2])
    Xb = np.hstack([Xbase, A2, AL])
    for iso in ISO:
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        m = np.isfinite(y)
        b = gbmae(Xbase[m], y[m])
        a = gbmae(Xa[m], y[m])
        c = gbmae(Xb[m], y[m])
        print(f"{iso}: base {b:.4f} | +ADME2 {a:.4f} ({b-a:+.4f}) | "
              f"+alerts {c:.4f} ({a-c:+.4f})", flush=True)


if __name__ == "__main__":
    main()
