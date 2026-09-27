"""Parameterized TabICL-on-frozen-chemprop-checkpoint TDI (step 1b diversification).

Generalizes src/tdi_tabicl_cp.py (which is hardwired to chemprop_medium) to any
of jeremy's frozen checkpoints via --src. Step 1b uses --src cpchm
(chemprop_chemeleon.pt): a DIFFERENT pretraining corpus => a genuinely new
MEMBER FAMILY for the family-block audit, to dilute v3's 2D6 cp concentration
(0.65 mean / 0.83 max on chemprop_medium, the shipped-v3 failure mode).

Same conventions as tdi_tabicl_cp.py: scaffold folds seed 7, PCA-256 whitened
(fit on train+test, unsupervised), TabICL n_estimators=4, challenge rows only
(NO AID1851 in-context rows - the --ext route is a proven dead end, sec 12/14).
Writes cache/tdi_tabicl_<src>_oof.npz + tdi_tabicl_<src>_test_probs.csv.

Run (tabicl-chemeleon env, GPU):
  ~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl_cp2.py --src cpchm
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
CKPT_DIR = "/tmp/jeremyscripts/CYP_Challenge/checkpoints"
CKPTS = {"cpmed": "chemprop_medium.pt", "cpchm": "chemprop_chemeleon.pt"}
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
            H = model.agg(model.message_passing(bmg, V_d), bmg.batch)
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


def emb_map(src, Xtr, Xte):
    """SMILES-keyed embeddings for train+test (cache/emb_<src>_all.parquet)."""
    from rdkit import Chem
    can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
    dall = os.path.join(CACHE, f"emb_{src}_all.parquet")
    smi_all = list(pd.unique(pd.concat([Xtr["SMILES"], Xte["SMILES"]])))
    if os.path.exists(dall):
        d0 = pd.read_parquet(dall)
    else:
        Eall = extract(os.path.join(CKPT_DIR, CKPTS[src]), smi_all)
        d0 = pd.DataFrame(Eall, index=pd.Index(smi_all, name="SMILES")).reset_index()
        d0.to_parquet(dall)
    E_map = {}
    for s, vec in zip(d0["SMILES"], d0.drop(columns=["SMILES"]).values):
        c = can(s)
        if c is not None:
            E_map[c] = np.asarray(vec, dtype=np.float32)
    tr_c = [can(s) for s in Xtr["SMILES"]]
    te_c = [can(s) for s in Xte["SMILES"]]
    dim = len(next(iter(E_map.values())))
    zero = np.zeros(dim, dtype=np.float32)
    Etr = np.array([E_map.get(c, zero) for c in tr_c], dtype=np.float32)
    Ete = np.array([E_map.get(c, zero) for c in te_c], dtype=np.float32)
    return Etr, Ete


def main(src):
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    Xte = pd.read_parquet(os.path.join(CACHE, "X_test.parquet"))
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                     emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab.reindex(Xtr["SMILES"])

    Etr, Ete = emb_map(src, Xtr, Xte)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=NPC, whiten=True, random_state=0)
    pca.fit(np.vstack([Etr, Ete]))
    Etr = pca.transform(Etr).astype(np.float32)
    Ete = pca.transform(Ete).astype(np.float32)

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
            clf = TabICLClassifier(n_estimators=4, random_state=42, device="cuda")
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
    np.savez(os.path.join(CACHE, f"tdi_tabicl_{src}_oof.npz"), **oof)
    pd.DataFrame({"SMILES": Xte["SMILES"], **{f"{i}_proba": np.mean(tepro[i], 0) for i in ISO}}).to_csv(
        os.path.join(CACHE, f"tdi_tabicl_{src}_test_probs.csv"), index=False)
    print("DONE", src, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=sorted(CKPTS), required=True)
    a = ap.parse_args()
    main(a.src)
