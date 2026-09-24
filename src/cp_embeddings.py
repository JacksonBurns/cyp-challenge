"""Frozen chemprop_medium / chemprop_chemeleon embeddings + ridge probe (regression).

jeremy's public kit ships two adme_pretrain D-MPNN checkpoints (pretrained on
public CYP/ADME data, NOT on challenge labels). Per his README the frozen
encoder + tabular model beats fine-tuning on this population. This script:
  1. extracts message_passing+agg embeddings for train (ft_data order) + test;
  2. ridge CV probe per isoform -> cache/oof_cpemb_<src>.csv + test preds
     (blend-ready format: SMILES + 4 iso columns; eval via eval_ft.py tag
     cpemb_<src>, blend via blendN_all family 'ft_cpemb_<src>'... actually the
     oof naming below: ft_oof_cpemb_<src>.csv to reuse the blend machinery).

Run (chemprop-dev env):
  ~/miniforge3/envs/chemprop-dev/bin/python src/cp_embeddings.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
CKPT_DIR = "/tmp/jeremyscripts/CYP_Challenge/checkpoints"
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
BATCH = 64


def extract(ckpt, smiles):
    model = models.MPNN.load_from_file(ckpt)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pts = [data.MoleculeDatapoint.from_smi(s) for s in smiles]
    dset = data.MoleculeDataset(pts, featurizer)
    loader = data.build_dataloader(dset, batch_size=BATCH, num_workers=0, shuffle=False)
    outs = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d = batch[0], batch[1], batch[2]
            bmg.to(dev)
            H = model.agg(model.message_passing(bmg, V_d), bmg.batch)
            outs.append(H.detach().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.vstack(outs)


def main():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES")
    smi_all = list(pd.unique(pd.concat([Xtr["SMILES"], test["SMILES"]])))
    for src, ckpt in [("cpmed", os.path.join(CKPT_DIR, "chemprop_medium.pt")),
                      ("cpchm", os.path.join(CKPT_DIR, "chemprop_chemeleon.pt"))]:
        print(f"== {src}", flush=True)
        fpall = os.path.join(CACHE, f"emb_{src}_all.parquet")
        if os.path.exists(fpall):
            dall = pd.read_parquet(fpall).set_index("SMILES")
        else:
            Eall = extract(ckpt, smi_all)
            dall = pd.DataFrame(Eall, index=pd.Index(smi_all, name="SMILES")).reset_index()
            dall.to_parquet(fpall)
        print("emb", dall.shape, flush=True)
        cols = [c for c in dall.columns if c != "SMILES"]
        Etr = dall[cols].reindex(ft["SMILES"]).fillna(0.0).values
        Ete = dall[cols].reindex(test["SMILES"]).fillna(0.0).values
        # ridge probe per isoform with scaffold folds
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
                r = RidgeCV(alphas=[100.0, 1000.0, 10000.0])
                r.fit(Xt[trn], y[trn])
                oof.loc[va, iso] = r.predict(Xt[va])
            r = RidgeCV(alphas=[100.0, 1000.0, 10000.0])
            r.fit(Xt[m], y[m])
            tpreds[iso] = r.predict(Xe)
            from scipy.stats import pearsonr
            print(f"  {iso}: OOF pearson {pearsonr(y[m], oof[iso].values[m]).statistic:.3f}", flush=True)
        oof.insert(0, "SMILES", ft["SMILES"])
        oof.to_csv(os.path.join(CACHE, f"ft_oof_{src}.csv"), index=False)
        tpreds.insert(0, "SMILES", test["SMILES"])
        tpreds.to_csv(os.path.join(CACHE, f"ft_test_preds_{src}.csv"), index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
