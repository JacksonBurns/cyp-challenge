"""Audit user-supplied ADME + rotational-barrier data for the aux-task idea.

1. coverage: rows, non-null per property, canonical-SMILES overlap with
   challenge train (6145) and test (750).
2. signal: GBM CV R2 predicting each property from the existing 3456-col
   descriptor stack (if R2 high, the signal is ALREADY in features ->
   aux tasks add little; if low, encoder may be blind -> aux tasks help).
3. relevance: property distributions for FLOOR vs HARD vs EASY challenge
   molecules on the overlap.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa
from rdkit import Chem

EXP = "/home/jackson/cyp-challenge/external/user_supplied/expansion_data_train.csv"
ROT = "/home/jackson/cyp-challenge/external/user_supplied/rotational_barrier_xinli.csv"


def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def gbm_cv_r2(X, y):
    import lightgbm as lgb
    from sklearn.metrics import r2_score
    m = np.isfinite(y)
    X, y = X[m], y[m]
    if len(y) < 100:
        return np.nan, 0
    pred = np.full(len(y), np.nan)
    kf = KFold(5, shuffle=True, random_state=0)
    for tr, te in kf.split(X):
        g = lgb.LGBMRegressor(n_estimators=300, num_leaves=31,
                              learning_rate=0.05, verbose=-1)
        g.fit(X[tr], y[tr])
        pred[te] = g.predict(X[te])
    return r2_score(y, pred), len(y)


def main():
    exp = pd.read_csv(EXP, encoding="utf-8-sig")
    print("== expansion: shape", exp.shape)
    props = [c for c in exp.columns if c not in ("Molecule Name", "SMILES")]
    for c in props:
        exp[c] = pd.to_numeric(exp[c], errors="coerce")
    for c in props:
        print(f"  {c}: n={exp[c].notna().sum()}")
    exp["cs"] = exp["SMILES"].map(canon)
    exp = exp.dropna(subset=["cs"]).drop_duplicates("cs")
    rot = pd.read_csv(ROT, encoding="utf-8-sig")
    rot.columns = ["SMILES", "tempK", "dG"]
    for c in ("tempK", "dG"):
        rot[c] = pd.to_numeric(rot[c], errors="coerce")
    rot["cs"] = rot["SMILES"].map(canon)
    rot = rot.dropna(subset=["cs"]).drop_duplicates("cs")
    print("== rot: rows", len(rot), "T range", rot.tempK.min(), rot.tempK.max(),
          "dG range", rot.dG.min().round(1), rot.dG.max().round(1))
    # overlap
    tr_sm = {canon(s) for s in H.targets.index}
    import os
    te = pd.read_csv(os.path.join(H.R.DATA, "cyp-challenge-TEST-BLINDED.csv"))
    te_sm = {canon(s) for s in te.SMILES}
    for name, df in [("exp", exp), ("rot", rot)]:
        cs = set(df.cs)
        print(f"== {name}: canonical {len(cs)} | overlap train {len(cs & tr_sm)} "
              f"| overlap TEST {len(cs & te_sm)}")
    # signal: GBM CV R2 from descriptor stack
    Xdf = H.Xtr.drop(columns=["SMILES"])
    Xdf["cs"] = [canon(s) for s in H.Xtr.SMILES]
    Xm = Xdf.set_index("cs")
    cols = list(Xdf.columns[:-1])
    # property -> transform
    def prep(df, col, log=False):
        d = df.dropna(subset=[col]).copy()
        y = d[col].values.astype(float)
        if log:
            y = np.log10(np.clip(y, 1e-3, None))
        idx = [c for c in d.cs if c in Xm.index]
        Xsub = Xm.loc[idx, cols].values.astype(float)
        med = np.nanmedian(np.where(np.isfinite(Xsub), Xsub, np.nan), axis=0)
        Xsub = np.where(np.isfinite(Xsub), Xsub, med)
        if len(idx) != len(y):
            return None
        return Xsub, y[:len(idx)]
    print("== GBM CV R2 from EXISTING descriptor stack ==")
    for c in props:
        r = prep(exp, c, log=("CLint" in c or "Papp" in c))
        if r is None:
            print(f"  {c}: align fail"); continue
        r2, n = gbm_cv_r2(*r)
        print(f"  {c}: n={n} R2={r2}")
    r2, n = gbm_cv_r2(*prep(rot, "dG"))
    print(f"  rot_dG: n={n} R2={r2}")
    # relevance: floor/hard/easy on overlap rows
    tgt = H.targets
    prof = pd.read_csv("/home/jackson/cyp-challenge/cache/chem_profile_rows.csv") \
        if os.path.exists("/home/jackson/cyp-challenge/cache/chem_profile_rows.csv") else None
    print("(relevance section: see next step if chem_profile_rows.csv exists)")


main()
