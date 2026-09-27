"""Frozen CheMeLeon MP 2048-d embeddings + ridge probe (regression), step 2a.

Cheap proven lever per the Sep 26 handoff: heterogeneous frozen-embedding
ridge probes into the blend pool (the cpmed/cpchm probes produced the last
+0.026 nested / +0.045 blind R2). This is the CheMeLeon analogue: the frozen
CheMeLeon MP (the 'emb' family's encoder) mean-pooled 2048-d, ridge with the
CORRECTED big-alpha grid only (100/1000/10000; small alphas overfit, see
NOTES sec 12). Blend-ready output ft_oof_chmridge.csv + ft_test_preds_
chmridge.csv, same scaffold folds (seed 7, cache/folds.npy via ft_data.csv)
and same conventions as src/cp_embeddings.py.

Run (cyp env; embeddings already cached in cache/chemeleon_emb.parquet):
  ~/miniforge3/bin/conda run -n cyp python src/chemeleon_ridge.py
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
    dall = pd.read_parquet(os.path.join(CACHE, "chemeleon_emb.parquet")).set_index("SMILES")
    cols = [c for c in dall.columns if c != "SMILES"]
    Etr = dall[cols].reindex(ft["SMILES"]).fillna(0.0).values
    Ete = dall[cols].reindex(test["SMILES"]).fillna(0.0).values
    n_missing = int(dall.index.intersection(ft["SMILES"]).shape[0])
    print(f"emb {Etr.shape}/{Ete.shape}; train SMILES covered {n_missing}/{len(ft)}", flush=True)
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
    oof.to_csv(os.path.join(CACHE, "ft_oof_chmridge.csv"), index=False)
    tpreds.insert(0, "SMILES", test["SMILES"])
    tpreds.to_csv(os.path.join(CACHE, "ft_test_preds_chmridge.csv"), index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
