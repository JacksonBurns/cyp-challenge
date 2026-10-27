"""Overlap + signal audit for public ADME sets vs challenge molecules."""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa
from rdkit import Chem

SANG = "/home/jackson/cyp-challenge/external/user_supplied/sangster_datasets.xlsx"


def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def gbm_cv_r2(X, y):
    import lightgbm as lgb
    from sklearn.metrics import r2_score
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[m], y[m]
    if len(y) < 100:
        return np.nan, len(y)
    pred = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        g = lgb.LGBMRegressor(n_estimators=300, num_leaves=31,
                              learning_rate=0.05, verbose=-1)
        g.fit(X[tr], y[tr])
        pred[te] = g.predict(X[te])
    return r2_score(y, pred), len(y)


def main():
    xl = pd.ExcelFile(SANG)
    print("sangster sheets:", xl.sheet_names)
    tr_sm = {canon(s) for s in H.targets.index}
    import os
    te = pd.read_csv(os.path.join(H.R.DATA, "cyp-challenge-TEST-BLINDED.csv"))
    te_sm = {canon(s) for s in te.SMILES}
    Xdf = H.Xtr.drop(columns=["SMILES"])
    Xdf["cs"] = [canon(s) for s in H.Xtr.SMILES]
    Xm = Xdf.set_index("cs")
    cols = list(Xdf.columns)[:-1]
    for sh in xl.sheet_names:
        d = xl.parse(sh)
        cols_d = list(d.columns)
        smi_col = next((c for c in cols_d if "smi" in str(c).lower()), None)
        y_col = next((c for c in cols_d
                      if any(k in str(c).lower() for k in
                             ("logp", "logd", "solu", "abs", "ppb", "bbl", "cli"))), None)
        if smi_col is None:
            print(f" {sh}: cols {cols_d[:5]}"); continue
        d["cs"] = d[smi_col].map(canon)
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        d = d.dropna(subset=["cs", y_col]).drop_duplicates("cs")
        ov_tr = set(d.cs) & tr_sm
        ov_te = set(d.cs) & te_sm
        print(f" {sh}: n={len(d)} prop={y_col} overlap_train={len(ov_tr)} "
              f"overlap_TEST={len(ov_te)}")
        if len(ov_tr) >= 30:
            sub = d[d.cs.isin(ov_tr)]
            Xs = Xm.loc[list(sub.cs), cols].values.astype(float)
            med = np.nanmedian(np.where(np.isfinite(Xs), Xs, np.nan), axis=0)
            Xs = np.where(np.isfinite(Xs), Xs, med)
            r2, n = gbm_cv_r2(Xs, sub[y_col].values)
            print(f"   GBM CV R2 from existing stack: {r2:.3f} (n={n})")


main()
