"""Build dscst_adme2 family: descriptor-stack GBM + ADME2 features, 5-fold
OOF (matches dscst family fold structure as built by tools/dsc_stack.py),
then pool gate zoo10 = zoo6 + dscst_adme2 with shuffled-member control
zoo10p. ADME-specific evidence = zoo10 gain - zoo10p gain per isoform."""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402
import run_regression as R  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CACHE = H.CACHE


def build():
    A2 = np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy"))
    Xbase = H.Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    X = np.hstack([Xbase, np.nan_to_num(A2, nan=0.0)])
    smi = H.Xtr["SMILES"].values
    groups = R.scaffold_groups(list(smi))
    folds = R.make_folds(groups, seed=7)
    out = pd.DataFrame({"SMILES": smi})
    for iso in ISO:
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        p = np.full(len(smi), np.nan)
        for f in range(5):
            trc = np.where((folds != f) & np.isfinite(y))[0]
            vac = np.where(folds == f)[0]
            g = LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63,
                              verbose=-1, random_state=7)
            g.fit(X[trc], y[trc])
            p[vac] = g.predict(X[vac])
        out[iso] = p
        print(f"built {iso}", flush=True)
    out.to_csv(os.path.join(CACHE, "ft_oof_dscst_adme2.csv"), index=False)
    rng = np.random.default_rng(7)
    sh = out.copy()
    for iso in ISO:
        v = np.array(sh[iso].values); ok = ~np.isnan(v)
        v[ok] = rng.permutation(v[ok]); sh[iso] = v
    sh.to_csv(os.path.join(CACHE, "ft_oof_dscst_adme2_shuf.csv"), index=False)
    print("built shuffled control")


def gate(mode):
    src = "ft_oof_dscst_adme2.csv" if mode == "real" else "ft_oof_dscst_adme2_shuf.csv"
    H.FAMS["dscst_adme2"] = [os.path.join(CACHE, src)]
    H.POOLS["zoo10"] = H.POOLS["zoo6"] + ["dscst_adme2"]
    tag = "zoo10" if mode == "real" else "zoo10p"
    H.main(emit=False, do_l4=True, pool="zoo10",
           wout=f"hinge_w_{tag}.json", report_name=f"hinge_blend_{tag}.json",
           out_csv=f"regression_{tag}_GATEONLY.csv")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        gate(sys.argv[1])
