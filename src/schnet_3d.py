"""3D GNN member - SchNet on MMFF94 conformers (Avenue 1).

NEW representation family: a 2D MPNN (chemprop, our backbone) only sees
connectivity. SchNet sees 3D geometry (bond lengths/angles/torsions from a
MMFF94-minimized conformer). CYP2D6 is the most STEREOSENSITIVE isoform
(strong enantioselectivity) - the one isoform where 3D geometry should matter
most and where our 2D pool hits a hard ceiling (~0.49 nested). This is the
direct test of "does 3D break the 2D6 ceiling."

Distinction from the dead list: the Uni-Mol result is FROZEN embeddings + ridge
(no CYP training). This is an END-TO-END trained 3D GNN on CYP labels.

Protocol (exact repo honest recipe, per isoform, single-task):
  per scaffold fold f (seed 0, K=5):
    train SchNet on folds != f (3D conformers), predict fold f, Pearson on val
  nested = Fisher-z avg;  R2 = nested^2
  anti-leak: conformers are SMILES-only (no target in X).

Output: cache/ft_oof_schnet{tag}.csv, cache/ft_test_preds_schnet{tag}.csv
Run quick:  ~/miniforge3/envs/unimol/bin/python src/schnet_3d.py --quick
Run full:   ~/miniforge3/envs/unimol/bin/python src/schnet_3d.py
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

NCONF = 1
EPOCHS = 30
BATCH = 64
LR = 1e-3
PATIENCE = 8
HIDDEN = 128
GAUSSIANS = 50
INTERACTIONS = 3
MAX_ATOMS = 150
_CONF_CACHE = {}


def make_conformers(smi, nconf):
    """Return (mol_with_H, [pos arrays]) best-energy first; mol is None if bad.

    In-memory cache: conformers are SMILES-only and reused across all folds,
    so each molecule is embedded exactly once (was 20x redundant).
    """
    key = (smi, nconf)
    if key in _CONF_CACHE:
        return _CONF_CACHE[key]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        _CONF_CACHE[key] = (None, [])
        return _CONF_CACHE[key]
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() > MAX_ATOMS:
        _CONF_CACHE[key] = (mol, [])
        return _CONF_CACHE[key]
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=nconf, randomSeed=7,
                                      useRandomCoords=False)
    if len(list(cids)) == 0:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=nconf, randomSeed=7,
                                          useRandomCoords=True)
    if len(list(cids)) == 0:
        _CONF_CACHE[key] = (mol, [])
        return _CONF_CACHE[key]
    props = AllChem.MMFFGetMoleculeProperties(mol)
    confs = []
    for cid in range(mol.GetNumConformers()):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            e = ff.CalcEnergy()
        except Exception:
            e = 0.0
        pos = np.array(mol.GetConformer(cid).GetPositions(), dtype=np.float32)
        confs.append((e, pos))
    confs.sort(key=lambda x: x[0])
    _CONF_CACHE[key] = (mol, [p for _, p in confs[:nconf]])
    return _CONF_CACHE[key]


def atom_feats(mol):
    f = []
    for a in mol.GetAtoms():
        f.append([a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
                  a.GetTotalNumHs(), float(a.GetIsAromatic())])
    return np.array(f, dtype=np.float32)


def mol_to_data(pos, mol, yval):
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # SchNet's embedding indexes by atomic number (long), not a float vec.
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    d = Data(x=z, pos=torch.tensor(pos), edge_index=edge_index,
             y=torch.tensor([yval], dtype=torch.float))
    return d


def build_ds(rows, smiles, y, nconf):
    ds = []
    for i in rows:
        mol, confs = make_conformers(smiles[i], nconf)
        if mol is None or not confs:
            continue
        for pos in confs:
            ds.append(mol_to_data(pos, mol, y[i]))
    return ds


def make_interaction(cutoff=10.0):
    """Pure-torch radius graph (avoids pyg-lib, which has no wheel here).

    SchNet calls this as (pos, batch) -> (edge_index, edge_weight).
    """
    def interaction_graph(pos, batch):
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # (N,N,3)
        dist = diff.norm(dim=-1)                     # (N,N)
        same = batch.unsqueeze(0) == batch.unsqueeze(1)
        mask = (dist < cutoff) & same
        mask.fill_diagonal_(False)
        src, dst = mask.nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0).long()
        edge_weight = dist[src, dst]
        return edge_index, edge_weight
    return interaction_graph


def make_model():
    return SchNet(hidden_channels=HIDDEN, num_filters=HIDDEN,
                  num_interactions=INTERACTIONS, num_gaussians=GAUSSIANS,
                  readout='add', interaction_graph=make_interaction(cutoff=10.0))


def train_schnet(train_ds, val_ds, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model().to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tr = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    va = DataLoader(val_ds, batch_size=256, shuffle=False) if val_ds else None

    def val_mse():
        if va is None:
            return float("inf")
        model.eval()
        vp, vy = [], []
        with torch.no_grad():
            for b in va:
                b = b.to("cuda")
                vp.append(fwd(model, b).cpu().numpy())
                vy.append(b.y.cpu().numpy())
        vp, vy = np.concatenate(vp), np.concatenate(vy)
        m = ~np.isnan(vy)
        return float(np.mean((vp[m] - vy[m]) ** 2)) if m.sum() > 5 else float("inf")

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for b in tr:
            b = b.to("cuda")
            opt.zero_grad()
            pred = fwd(model, b)
            m = ~torch.isnan(b.y)
            if m.sum() == 0:
                continue
            loss = F.mse_loss(pred[m], b.y[m])
            loss.backward()
            opt.step()
        v = val_mse()
        if v < best - 1e-5:
            best = v
            best_state = {k: t.cpu().clone() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, ds):
    if not ds:
        return np.array([])
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for b in loader:
            b = b.to("cuda")
            out.append(model(b.x, b.pos, b.batch).cpu().numpy())
    return np.concatenate(out).ravel()


def fwd(model, b):
    """This torch_geometric SchNet takes (z, pos, batch) explicitly."""
    return model(b.x, b.pos, b.batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--nconf", type=int, default=1)
    a = ap.parse_args()
    global NCONF
    NCONF = a.nconf

    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    smiles = ft["SMILES"].tolist()
    folds = ft["fold"].values
    test = pd.read_csv(os.path.join(CACHE, "..", "data",
                                    "cyp-challenge-TEST-BLINDED.csv"))

    isoforms = ["CYP2D6"] if a.quick else ISO
    fold_list = [0] if a.quick else range(5)
    epochs = 4 if a.quick else EPOCHS

    oof = pd.DataFrame(np.nan, index=ft.index, columns=isoforms)
    t0 = time.time()
    for iso in isoforms:
        y = ft[iso].values.astype(float)
        for fd in fold_list:
            trm = np.where(~np.isnan(y) & (folds != fd))[0]
            vam = np.where(~np.isnan(y) & (folds == fd))[0]
            train_ds = build_ds(trm, smiles, y, NCONF)
            val_ds = build_ds(vam, smiles, y, NCONF)
            model = train_schnet(train_ds, val_ds, seed=7000 + fd, epochs=epochs)
            vp = predict(model, val_ds)
            # map val preds back to rows (avg over NCONF per row, in order)
            pred_by_row = np.full(len(ft), np.nan)
            for k, ri in enumerate(vam):
                chunk = vp[k * NCONF:(k + 1) * NCONF]
                if len(chunk):
                    pred_by_row[ri] = np.mean(chunk)
            oof.loc[vam, iso] = pred_by_row[vam]
            del model
            torch.cuda.empty_cache()
            r = pearsonr(y[vam], pred_by_row[vam]).statistic
            print(f"  {iso} fold{fd}: val Pearson {r:.4f} [{time.time()-t0:.0f}s]",
                  flush=True)

    nested = {}
    for iso in isoforms:
        y = ft[iso].values.astype(float)
        m = ~np.isnan(y)
        rs = []
        for fd in fold_list:
            vam = m & (folds == fd)
            r = pearsonr(y[vam], oof[iso].values[vam]).statistic
            if not np.isnan(r):
                rs.append(r)
        nested[iso] = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
        print(f"  {iso}: nested={nested[iso]:.4f} R2={nested[iso]**2:.4f}", flush=True)

    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, f"ft_oof_schnet{a.tag}.csv"), index=False)
    print(f"saved ft_oof_schnet{a.tag}.csv", flush=True)

    # test preds: full-train model per isoform
    tdf = pd.DataFrame(index=range(len(test)), columns=isoforms)
    for iso in isoforms:
        y = ft[iso].values.astype(float)
        m = np.where(~np.isnan(y))[0]
        ds = build_ds(m, smiles, y, NCONF)
        model = train_schnet(ds, ds[:64], seed=500, epochs=EPOCHS)
        tds = build_ds(range(len(test)), test["SMILES"].tolist(),
                       np.full(len(test), np.nan), NCONF)
        tp = predict(model, tds)
        for k in range(len(test)):
            chunk = tp[k * NCONF:(k + 1) * NCONF]
            tdf.loc[k, iso] = np.mean(chunk) if len(chunk) else np.nan
        del model
        torch.cuda.empty_cache()
    tdf.insert(0, "SMILES", test["SMILES"])
    tdf.to_csv(os.path.join(CACHE, f"ft_test_preds_schnet{a.tag}.csv"), index=False)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
