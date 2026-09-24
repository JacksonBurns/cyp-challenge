"""TDI via TabICL on jeremy's frozen chemprop_medium embeddings (his 0.402 route).

Same structure as src/tdi_tabicl.py (scaffold folds seed 7, PCA-256 whitened,
n_estimators=4) but the embedding source is the adme_pretrain D-MPNN (trained
on public CYP/ADME + Novartis-surrogate panel) instead of CheMeLeon. Adds an
EXTERNAL-POSITIVES variant: rows of AID1851 with binary labels embedded and
joined to the TRAINING set (in-context examples), which is jeremy's precision
fix on this track. Writes tdi_tabicl_cp_{oof.npz,test_probs.csv} (+ _ext).

Run (tabicl-chemeleon env, GPU):
  ~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl_cp.py [--ext 1]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tabicl import TabICLClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")
CKPT = "/tmp/jeremyscripts/CYP_Challenge/checkpoints/chemprop_medium.pt"
ISO = ["CYP2D6", "CYP3A4"]
K = 5
BATCH = 64
NPC = 256


def mcc(p, y, t):
    pr = (p >= t).astype(int)
    tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum()
    tn = ((pr == 0) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum()
    return float((tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12))


def extract(ckpt, smiles):
    import torch
    from chemprop import data, featurizers, models
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
            H = model.agg(model.message_passing(bmg, V_d, X_d), bmg.batch)
            outs.append(H.detach().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.vstack(outs)


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


def main(use_ext):
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    Xte = pd.read_parquet(os.path.join(CACHE, "X_test.parquet"))
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                     emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab.reindex(Xtr["SMILES"])

    # embeddings for train+test (+ external rows when use_ext)
    fpath = os.path.join(CACHE, f"tdi_cpmed_emb{'_ext' if use_ext else ''}.npz")
    if os.path.exists(fpath):
        z = np.load(fpath)
        Etr, Ete, Eex, ex_lab = z["Etr"], z["Ete"], z["Eex"], z["ex_lab"]
    else:
        from rdkit import Chem
        can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
        E_map = {}

        def get_emb(smis):
            E = extract(CKPT, smis)
            return {can(s): E[i] for i, s in enumerate(smis) if can(s)}
        tr_c = [can(s) for s in Xtr["SMILES"]]
        te_c = [can(s) for s in Xte["SMILES"]]
        need = [s for s in pd.Series(tr_c + te_c).dropna().unique() if s not in E_map]
        E_map.update(get_emb(list(need)))
        Etr = np.array([E_map.get(c, np.zeros_like(next(iter(E_map.values())))) for c in tr_c], dtype=np.float32)
        Ete = np.array([E_map.get(c, np.zeros_like(next(iter(E_map.values())))) for c in te_c], dtype=np.float32)
        Eex = np.zeros((0, Etr.shape[1]), dtype=np.float32)
        ex_lab = np.zeros((0, 2))
        if use_ext:
            pc = pd.read_csv(os.path.join(HERE, "..", "external", "pubchem_cyp_qhts_aid1851.csv"))
            pc["canon"] = pc["smiles"].map(can)
            pc = pc.dropna(subset=["canon"])
            both = pc.dropna(subset=[f"{i}_active" for i in ISO]).drop_duplicates("canon")
            both = both[~both["canon"].isin(set(tr_c) | set(te_c))]
            Eex = np.array([E_map[c] if c in E_map else extract(CKPT, [c])[0] for c in both["canon"]],
                           dtype=np.float32)
            ex_lab = both[[f"{i}_active" for i in ISO]].values.astype(float)
        np.savez(fpath, Etr=Etr, Ete=Ete, Eex=Eex, ex_lab=ex_lab)

    from sklearn.decomposition import PCA
    stack = [Etr, Ete] + ([Eex] if len(Eex) else [])
    pca = PCA(n_components=NPC, whiten=True, random_state=0)
    pcs = [pca.transform(x).astype(np.float32) for x in stack]
    Etr, Ete, Eex = pcs[0], pcs[1], pcs[2] if len(pcs) > 2 else np.zeros((0, NPC), dtype=np.float32)

    folds = make_folds(scaffold_groups(Xtr["SMILES"].tolist()), seed=7)
    oof = {i: np.full(len(Xtr), np.nan) for i in ISO}
    tepro = {i: [] for i in ISO}
    for f in range(K):
        for j, iso in enumerate(ISO):
            y = Y[f"{iso}_is_TDI"].values
            l = ~pd.isna(y)
            tr = np.where((folds != f) & l)[0]
            va = np.where((folds == f) & l)[0]
            yy = np.array([1 if bool(v) else 0 for v in y[tr]])
            X_fit, y_fit = Etr[tr], yy
            if len(Eex):
                el = ~np.isnan(ex_lab[:, j])
                X_fit = np.vstack([X_fit, Eex[el]])
                y_fit = np.concatenate([y_fit, ex_lab[el, j].astype(int)])
            clf = TabICLClassifier(n_estimators=4, random_state=42, device="cuda")
            clf.fit(X_fit, y_fit)
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
    sfx = "_ext" if use_ext else ""
    np.savez(os.path.join(CACHE, f"tdi_tabicl_cp{sfx}_oof.npz"), **oof)
    pd.DataFrame({"SMILES": Xte["SMILES"], **{f"{i}_proba": np.mean(tepro[i], 0) for i in ISO}}).to_csv(
        os.path.join(CACHE, f"tdi_tabicl_cp{sfx}_test_probs.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", type=int, default=0)
    a = ap.parse_args()
    main(bool(a.ext))
