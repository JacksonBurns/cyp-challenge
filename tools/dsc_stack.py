"""dscst arm: LightGBM stacking descriptors + cpmed OOF + Morgan PCA.

Precedent: ft_tabpfn_cpmed (TabPFN over cpmed) is the top-weighted
family in every gate. This arm stacks on the SAME incumbent OOF with
descriptor + fingerprint side features so the GBM can learn where
cpmed is biased (e.g. 2C9 vs flat/lipophilic axis, rho .485).
CPU arm (cyp env, lightgbm). Artifacts: ft_oof_dscst.csv,
ft_test_preds_dscst.csv.
"""
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
NFP = 64


def morgan_pca(train_smi, test_smi, seed=0):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def counts(smis):
        rows = []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                rows.append(np.zeros(2048))
                continue
            fp = gen.GetCountFingerprintAsNumPy(m)
            rows.append(fp)
        return np.asarray(rows, dtype=np.float32)

    Xtr, Xte = counts(train_smi), counts(test_smi)
    pc = PCA(n_components=NFP, random_state=seed).fit(np.vstack([Xtr, Xte]))
    return pc.transform(Xtr), pc.transform(Xte)


def main():
    fd = pd.read_csv(os.path.join(CACHE, "ft_data.csv")).set_index("SMILES")
    tr = pd.read_csv(os.path.join(CACHE, "desc3d_train.csv")).set_index("SMILES")
    te = pd.read_csv(os.path.join(CACHE, "desc3d_test.csv")).set_index("SMILES")
    cp = pd.read_csv(os.path.join(CACHE, "ft_oof_cpmed.csv")).set_index("SMILES")
    cpt = pd.read_csv(os.path.join(CACHE, "ft_test_preds_cpmed.csv")).set_index("SMILES")
    with open(os.path.join(CACHE, "desc3d_corr.json")) as f:
        corr = json.load(f)
    score = {c: max(abs(corr[i].get(c, 0)) for i in ISO) for c in tr.columns}
    keep = sorted(score, key=score.get, reverse=True)[:25]

    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
    te_idx = test["SMILES"].drop_duplicates()
    tr_idx = fd.index[fd["fold"].notna() & fd[ISO].notna().any(axis=1)]
    folds = fd.loc[tr_idx, "fold"].astype(int).values

    Ftr_d, Fte_d = tr[keep].loc[tr_idx].fillna(0.0), te[keep].reindex(te_idx).fillna(0.0)
    Ptr, Pte = cp[ISO].loc[tr_idx], cpt[ISO].reindex(te_idx)
    Mtr, Mte = morgan_pca(list(tr_idx), list(te_idx))

    oof = pd.DataFrame(index=tr_idx, columns=ISO, dtype=float)
    testp = pd.DataFrame(index=te_idx, columns=ISO, dtype=float)
    for iso in ISO:
        y = fd.loc[tr_idx, iso].values
        m = ~pd.isna(y)
        Xtr = np.hstack([Ftr_d.values[m], Ptr[[iso]].values[m], Mtr[m]])
        ym = y[m]
        fm = folds[m]
        preds = np.full(m.sum(), np.nan)
        for f in range(5):
            trm, vam = fm != f, fm == f
            g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                  num_leaves=31, min_child_samples=40,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=0, verbose=-1)
            g.fit(Xtr[trm], ym[trm])
            preds[vam] = g.predict(Xtr[vam])
        # index map back to full tr_idx
        full = np.full(len(tr_idx), np.nan)
        full[np.flatnonzero(m)] = preds
        oof[iso] = full
        g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                              num_leaves=31, min_child_samples=40,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=0, verbose=-1)
        g.fit(Xtr, ym)
        Xte = np.hstack([Fte_d[[c for c in keep]].values,
                         Pte[[iso]].values, Mte])
        testp[iso] = g.predict(Xte)
    oof.insert(0, "SMILES", oof.index)
    testp.insert(0, "SMILES", testp.index)
    oof.to_csv(os.path.join(CACHE, "ft_oof_dscst.csv"), index=False)
    testp.to_csv(os.path.join(CACHE, "ft_test_preds_dscst.csv"), index=False)
    print("WROTE ft_oof_dscst.csv + ft_test_preds_dscst.csv")


if __name__ == "__main__":
    main()
