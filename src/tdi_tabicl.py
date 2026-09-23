"""TDI via TabICL on CheMeLeon frozen embeddings (per jeremy: TabICL > TabPFN).

Scaffold-grouped CV (same folds/labels/NN-positives as run_tdi.py minus the
tree-only NN features - TabICL gets raw embedding + FP-desc tables). Two tables:
  emb: 2048-d CheMeLeon embeddings
  both: embeddings + a small RDKit subset is NOT included (jeremy used raw fp);
  keep it simple: embeddings only.
Writes cache/tdi_tabicl_oof.npz (labeled-row full-length arrays), test probas,
and per-iso OOF MCC (best fraction) + blend-with-base scan.

Run in the tabicl-chemeleon env (GPU):
  ~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from tabicl import TabICLClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)

ISO = ["CYP2D6", "CYP3A4"]
K = 5


def scaffold_groups(smiles_list):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    keys = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        try:
            k = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else ""
        except Exception:
            k = ""
        keys.append(k)
    fixed, bucket = [], 0
    for k in keys:
        if k == "":
            fixed.append(f"__NONE__{bucket // 40}")
            bucket += 1
        else:
            fixed.append(k)
    return fixed


def make_folds(group_keys, k=5, seed=0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(group_keys, return_inverse=True)
    sizes = np.bincount(inv)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-sizes[order])]
    fold = np.empty(len(group_keys), dtype=int)
    load = np.zeros(k)
    for g in order:
        f = int(np.argmin(load))
        fold[inv == g] = f
        load[f] += sizes[g]
    return fold


def mcc(p, y, t):
    pr = (p >= t).astype(int)
    tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum()
    tn = ((pr == 0) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum()
    return float((tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12))


def main():
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    Xte = pd.read_parquet(os.path.join(CACHE, "X_test.parquet"))
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                     emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab.reindex(Xtr["SMILES"])

    emb = pd.read_parquet(os.path.join(CACHE, "chemeleon_emb.parquet")).set_index("SMILES")
    Etr = emb.reindex(Xtr["SMILES"]).fillna(0.0).values.astype(np.float32)
    Ete = emb.reindex(Xte["SMILES"]).fillna(0.0).values.astype(np.float32)
    folds = make_folds(scaffold_groups(Xtr["SMILES"].tolist()), seed=7)

    oof = {i: np.full(len(Xtr), np.nan) for i in ISO}
    tepro = {i: [] for i in ISO}
    for f in range(K):
        for iso in ISO:
            y = Y[f"{iso}_is_TDI"].values
            l = ~pd.isna(y)
            tr = np.where((folds != f) & l)[0]
            va = np.where((folds == f) & l)[0]
            yy = np.array([1 if bool(v) else 0 for v in y[tr]])
            clf = TabICLClassifier(n_estimators=8, random_state=42, device="cuda")
            clf.fit(Etr[tr], yy)
            oof[iso][va] = clf.predict_proba(Etr[va])[:, 1]
            tepro[iso].append(clf.predict_proba(Ete)[:, 1])
            print(f"fold {f} {iso} done", flush=True)
            del clf
            import torch
            torch.cuda.empty_cache()
    for iso in ISO:
        y = Y[f"{iso}_is_TDI"].values
        l = ~pd.isna(y)
        yl = np.array([1 if bool(v) else 0 for v in y[l]])
        p = oof[iso][l]
        best = max(((mcc(p, yl, np.quantile(p, 1 - fr)), fr) for fr in np.arange(0.05, 0.61, 0.01)))
        print(iso, "OOF best MCC", round(best[0], 4), "@frac", best[1], flush=True)
    np.savez(os.path.join(CACHE, "tdi_tabicl_oof.npz"), **oof)
    pd.DataFrame({"SMILES": Xte["SMILES"], **{f"{i}_proba": np.mean(tepro[i], 0) for i in ISO}}).to_csv(
        os.path.join(CACHE, "tdi_tabicl_test_probs.csv"), index=False)


if __name__ == "__main__":
    main()
