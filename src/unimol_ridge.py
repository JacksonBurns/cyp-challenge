"""Uni-Mol 512-d embeddings + ridge probe (regression, Task 3 member).

chemeleon_ridge.py recipe applied to the Uni-Mol 3D-conformer embeddings
(cache/unimol_emb.parquet): standardize, RidgeCV big-alpha grid only
(100/1000/10000), scaffold folds seed 7 from ft_data.csv, blend-ready outputs
ft_oof_unimol.csv + ft_test_preds_unimol.csv. Uni-Mol is the NEW encoder
family (3D, not 2D graph) every top report credits for representation
diversity (stir_bar rank 6 TDI / 28 reg).

Run (cyp env):
  ~/miniforge3/envs/cyp/bin/python src/unimol_ridge.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
ALPHAS = [100.0, 1000.0, 10000.0]


def main():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    dall = pd.read_parquet(os.path.join(CACHE, "unimol_emb.parquet")).set_index("SMILES")
    cols = list(dall.columns)
    Etr = dall[cols].reindex(ft["SMILES"]).values
    Ete = dall[cols].reindex(test["SMILES"]).values
    assert not np.isnan(Etr).any(), "missing train embeddings"
    assert not np.isnan(Ete).any(), "missing test embeddings"
    print(f"emb {Etr.shape}/{Ete.shape}", flush=True)
    folds = ft["fold"].values
    oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    tpreds = pd.DataFrame(np.nan, index=test.index, columns=ISO)
    for j, iso in enumerate(ISO):
        y = ft[iso].values
        m = ~np.isnan(y)
        sc = StandardScaler().fit(Etr[m])
        Xt, Xe = sc.transform(Etr), sc.transform(Ete)
        for f in range(5):
            trn = m & (folds != f)
            va = m & (folds == f)
            r = RidgeCV(alphas=ALPHAS)
            r.fit(Xt[trn], y[trn])
            oof.loc[va, iso] = r.predict(Xt[va])
        r = RidgeCV(alphas=ALPHAS)
        r.fit(Xt[m], y[m])
        tpreds[iso] = r.predict(Xe)
        print(f"  {iso}: OOF pearson {pearsonr(y[m], oof[iso].values[m]).statistic:.3f} "
              f"(alpha={r.alpha_:.0f})", flush=True)
    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, "ft_oof_unimol.csv"), index=False)
    tpreds.insert(0, "SMILES", test["SMILES"])
    tpreds.to_csv(os.path.join(CACHE, "ft_test_preds_unimol.csv"), index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
