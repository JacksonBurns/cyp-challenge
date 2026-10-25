"""Descriptor arm: correlation screen + per-isoform LightGBM OOF + test preds.

Gate-ready artifacts: cache/ft_oof_dsc3d.csv, cache/ft_test_preds_dsc3d.csv.
Correlation report: cache/desc3d_corr.json (per-iso Spearman of each
descriptor vs pIC50, and vs each other for redundancy).
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def main(top=25):
    tr = pd.read_csv(os.path.join(CACHE, "desc3d_train.csv"))
    fd = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(HERE, "..", "data",
                                    "cyp-challenge-TEST-BLINDED.csv"))
    te = pd.read_csv(os.path.join(CACHE, "desc3d_test.csv"))
    fd = fd.set_index("SMILES")
    feats = [c for c in tr.columns if c != "SMILES"]

    lab = fd[ISO].reindex(tr["SMILES"])
    ok = lab.notna().any(axis=1).values
    X = tr[feats].loc[ok]
    folds = fd["fold"].reindex(tr["SMILES"]).values[ok]

    # correlation screen vs each isoform label
    corr = {}
    for iso in ISO:
        y = lab[iso].values[ok]
        m = ~pd.isna(y)
        cs = {}
        for c in feats:
            x = X[c].values[m]
            yy = y[m]
            if np.std(x) > 0 and len(x) > 50:
                rho = spearmanr(x, yy).statistic
                cs[c] = float(rho) if np.isfinite(rho) else 0.0
        corr[iso] = cs
    with open(os.path.join(CACHE, "desc3d_corr.json"), "w") as f:
        json.dump(corr, f, indent=1)
    for iso in ISO:
        top10 = sorted(corr[iso].items(), key=lambda kv: -abs(kv[1]))[:10]
        print(iso, [(k, round(v, 3)) for k, v in top10], flush=True)

    # top-descriptor union by |rho|
    score = {c: max(abs(corr[i].get(c, 0)) for i in ISO) for c in feats}
    keep = sorted(score, key=score.get, reverse=True)[:top]
    print("keep:", keep, flush=True)

    # per-iso GBM with grouped folds
    oof = pd.DataFrame(index=tr["SMILES"][ok], columns=ISO, dtype=float)
    Xte = te.set_index("SMILES")
    te_idx = test["SMILES"].drop_duplicates()
    Xte = Xte.reindex(te_idx)
    testp = pd.DataFrame(index=te_idx, columns=ISO, dtype=float)
    for iso in ISO:
        y = lab[iso].values
        m = ~pd.isna(y)
        Xm = X[keep].values[m]
        ym = y[m]
        fm = folds[m]
        preds = np.full(m.sum(), np.nan)
        for f in range(5):
            trn, val = fm != f, fm == f
            g = LGBMRegressor(n_estimators=400, learning_rate=0.05,
                              num_leaves=31, min_child_samples=20,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=0, verbose=-1)
            g.fit(Xm[trn], ym[trn])
            preds[val] = g.predict(Xm[val])
        oof.loc[tr["SMILES"][ok][m], iso] = preds
        g = LGBMRegressor(n_estimators=400, learning_rate=0.05,
                          num_leaves=31, min_child_samples=20,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=0, verbose=-1)
        g.fit(Xm, ym)
        testp[iso] = g.predict(Xte[keep].values)
    oof.insert(0, "SMILES", oof.index)
    testp.insert(0, "SMILES", testp.index)
    oof.to_csv(os.path.join(CACHE, "ft_oof_dsc3d.csv"), index=False)
    testp.to_csv(os.path.join(CACHE, "ft_test_preds_dsc3d.csv"), index=False)
    print("WROTE ft_oof_dsc3d.csv + ft_test_preds_dsc3d.csv", flush=True)


if __name__ == "__main__":
    main()
