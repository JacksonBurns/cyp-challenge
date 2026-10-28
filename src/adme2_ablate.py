"""ADME2 endpoint ablation: which endpoints drive the A2 gate gains?

For each isoform, GBM MAE with stack + single-endpoint blocks grouped by
physics family, and drop-one from the full 18. KFold(5) matches A2.
"""
import os
import sys

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CACHE = H.CACHE

# feature order from adme2_build.py (18): v1 8 + TDC 10
NAMES = ["caco2_wang", "hia_hou", "pgp", "bbb", "vdss", "ppbr", "aqsol",
         "half_life", "cl_hep", "cl_mic", "lipo_az",
         "cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith", "cyp1a2_veith",
         "cyp2c9_sub", "cyp2d6_sub", "cyp3a4_sub"]
V1N = ["logd", "logs_exp", "ksol", "hlm_clint", "mlm_clint", "caco2_papp",
       "efflux", "mppb"]


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
    A2 = np.nan_to_num(np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy")),
                       nan=0.0)
    V1 = np.nan_to_num(np.load(os.path.join(CACHE, "adme_aux", "adme_feats_tr.npy")),
                       nan=0.0)
    A2 = np.hstack([A2, V1])  # cols 18-25 = v1 block
    print("A2 block", A2.shape, flush=True)
    Xbase = H.Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    groups = {
        "solubility": [6, 20, 25],        # aqsol, ksol, mppb
        "permeability": [0, 1, 2, 3, 23, 24],  # caco2wang, hia, pgp, bbb, caco2papp, efflux
        "lipophilicity": [10, 18],        # lipo_az, logd
        "distribution": [4, 5],           # vdss, ppbr
        "metabolism": [7, 8, 9, 21, 22],  # half_life, cl_hep/mic, hlm/mlm clint
        "cyp_veith_sub": [11, 12, 13, 14, 15, 16, 17],
    }
    for iso in ISO:
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        m = np.isfinite(y)
        b = gbmae(Xbase[m], y[m])
        full = gbmae(np.hstack([Xbase, A2])[m], y[m])
        print(f"{iso}: base {b:.4f} full18 {full:.4f} gain {b-full:+.4f}", flush=True)
        for gname, cols in groups.items():
            Xg = np.hstack([Xbase, A2[:, cols]])
            g = gbmae(Xg[m], y[m])
            print(f"   +{gname} ({len(cols)}): {b-g:+.4f}", flush=True)
        # drop-one by group from full
        for gname, cols in groups.items():
            keep = [j for j in range(A2.shape[1]) if j not in cols]
            Xd = np.hstack([Xbase, A2[:, keep]])
            d = gbmae(Xd[m], y[m])
            print(f"   -{gname} from full: {d-full:+.4f}", flush=True)


if __name__ == "__main__":
    main()
