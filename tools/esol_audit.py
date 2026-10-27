"""ESOL (Delaney) solubility audit: overlap + predictability from stack."""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa
from rdkit import Chem


def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def gbm_cv_r2(X, y):
    import lightgbm as lgb
    from sklearn.metrics import r2_score
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[m], y[m]
    pred = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        g = lgb.LGBMRegressor(n_estimators=300, num_leaves=31,
                              learning_rate=0.05, verbose=-1)
        g.fit(X[tr], y[tr])
        pred[te] = g.predict(X[te])
    return r2_score(y, pred), len(y)


es = pd.read_csv("/home/jackson/cyp-challenge/external/user_supplied/delaney-processed.csv")
ec = [c for c in es.columns if "SMILES" in c.upper()][0]
es["cs"] = es[ec].map(canon)
es = es.dropna(subset=["cs"]).drop_duplicates("cs")
col = "measured log solubility in mols per litre"
es[col] = pd.to_numeric(es[col], errors="coerce")
es = es.dropna(subset=[col])
tr_sm = {canon(s) for s in H.targets.index}
import os
te = pd.read_csv(os.path.join(H.R.DATA, "cyp-challenge-TEST-BLINDED.csv"))
te_sm = {canon(s) for s in te.SMILES}
ov_tr, ov_te = set(es.cs) & tr_sm, set(es.cs) & te_sm
print(f"ESOL: n={len(es)} overlap_train={len(ov_tr)} overlap_TEST={len(ov_te)}")
Xdf = H.Xtr.drop(columns=["SMILES"])
Xdf["cs"] = [canon(s) for s in H.Xtr.SMILES]
Xm = Xdf.set_index("cs")
cols = list(Xdf.columns)[:-1]
if len(ov_tr) >= 30:
    sub = es[es.cs.isin(ov_tr)]
    Xs = Xm.loc[list(sub.cs), cols].values.astype(float)
    med = np.nanmedian(np.where(np.isfinite(Xs), Xs, np.nan), axis=0)
    Xs = np.where(np.isfinite(Xs), Xs, med)
    r2, n = gbm_cv_r2(Xs, sub[col].values)
    print(f"ESOL GBM CV R2 from existing stack (on overlap): {r2:.3f} (n={n})")
r2all, nall = gbm_cv_r2(es.drop(columns=["cs", "SMILES", "Compound ID"]).select_dtypes("number").values, es[col].values)
print(f"(note: ESOL's own 8 descriptors R2={r2all:.3f} n={nall} - reference)")
