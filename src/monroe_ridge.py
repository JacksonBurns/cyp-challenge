"""Monroe L5 ridge probe: frozen Monroe 720-d embeddings -> blend-family members.

Mirrors src/chemeleon_ridge.py exactly (same folds seed 7 via ft_data.csv fold
column, RidgeCV big-alpha grid only, StandardScaler on train-fold labeled rows,
OOF + all-data test preds) but on cache/monroe_emb.parquet (src/monroe_embed.py).
GO gate (FINAL_PUSH_PLAN 1.3): per-isoform OOF pearson beats the chmridge
singles floor 0.468/0.585/0.344/0.723 on >= 2 isoforms. If GO, ft_oof_monridge
joins blend pools as its own family; if NO-GO, log and proceed (plan).

Run (cyp env): ~/miniforge3/envs/cyp/bin/python src/monroe_ridge.py
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
CHMRIDGE_FLOOR = {"CYP1A2": 0.468, "CYP2C9": 0.585, "CYP2D6": 0.344, "CYP3A4": 0.723}


def main():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    dall = pd.read_parquet(os.path.join(CACHE, "monroe_emb.parquet"))
    if "SMILES" not in dall.columns:
        dall = dall.reset_index()
    dall = dall.set_index("SMILES")
    cols = list(dall.columns)
    Etr = dall[cols].reindex(ft["SMILES"]).fillna(0.0).values
    Ete = dall[cols].reindex(test["SMILES"]).fillna(0.0).values
    cov_tr = int(dall.index.intersection(ft["SMILES"]).shape[0])
    cov_te = int(dall.index.intersection(test["SMILES"]).shape[0])
    print(f"emb {Etr.shape}/{Ete.shape}; coverage train {cov_tr}/{len(ft)} test {cov_te}/{len(test)}", flush=True)
    folds = ft["fold"].values
    oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    tpreds = pd.DataFrame(np.nan, index=test.index, columns=ISO)
    go = 0
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
        pr = float(pearsonr(y[m], oof[iso].values[m]).statistic)
        beat = pr > CHMRIDGE_FLOOR[iso]
        go += int(beat)
        print(f"  {iso}: OOF pearson {pr:.3f} (alpha={r.alpha_:.0f}) floor {CHMRIDGE_FLOOR[iso]} -> {'BEAT' if beat else 'under'}", flush=True)
    print(f"L5 GATE: beats chmridge on {go}/4 isoforms (GO needs >= 2) -> {'GO' if go >= 2 else 'NO-GO'}", flush=True)
    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, "ft_oof_monridge.csv"), index=False)
    tpreds.insert(0, "SMILES", test["SMILES"])
    tpreds.to_csv(os.path.join(CACHE, "ft_test_preds_monridge.csv"), index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
