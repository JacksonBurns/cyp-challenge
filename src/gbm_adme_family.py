"""Build cache/ft_oof_gbm_adme.csv: per-isoform GBM on stack+ADME features,
challenge folds. Then pool gate zoo6 vs zoo6+gbm_adme (zoo8)."""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CACHE = "/home/jackson/cyp-challenge/cache"
Afe = np.load(os.path.join(CACHE, "adme_aux", "adme_feats_tr.npy"))


def lgbm():
    return lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03,
                             num_leaves=63, min_child_samples=15,
                             subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, reg_alpha=0.5,
                             reg_lambda=2.0, n_jobs=-1, random_state=42,
                             verbosity=-1)


def main():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    smi = ft["SMILES"].values
    folds = ft["fold"].values
    Xbase = np.nan_to_num(H.Xtr.drop(columns=["SMILES"], errors="ignore").values.astype(float), nan=0.0)
    X = np.hstack([Xbase, np.nan_to_num(Afe, nan=0.0)])
    out = pd.DataFrame({"SMILES": smi})
    for iso in ISO:
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        oof = np.full(len(smi), np.nan)
        for f in range(5):
            tr = np.where((~np.isnan(y)) & (folds != f))[0]
            va = np.where(folds == f)[0]
            m = lgbm()
            m.fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])
        out[iso] = oof
        print(f"gbm_adme {iso} done", flush=True)
    out.to_csv(os.path.join(CACHE, "ft_oof_gbm_adme.csv"), index=False)
    print("wrote", os.path.join(CACHE, "ft_oof_gbm_adme.csv"))


if __name__ == "__main__":
    main()
