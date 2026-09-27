"""D-MPNN binary classifier head for TDI (step 1b diversification member).

Challenge rows only (external qHTS pooling is a proven dead end, sec 12/14):
2 task heads (2D6/3A4 is_TDI) with BCE, NaN-masked like ft_dmpnn, same
scaffold folds (seed 7) and label union (TDI+Emax) as the rest of the TDI
pipeline. Standalone it is expected to be weak; its job is DECORRELATION in
the family-block blend audit (new family = dmpnn) to dilute the 2D6 cp
concentration that sank shipped v3.

Run (chemprop-dev env, GPU):
  ~/miniforge3/envs/chemprop-dev/bin/python src/tdi_dmpnn.py --seed 7
Writes cache/tdi_dmpnn_oof.npz + tdi_dmpnn_test_probs.csv.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from chemprop.nn.metrics import BCELoss
from lightning import pytorch as pl, seed_everything
from lightning.pytorch.callbacks import EarlyStopping

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def scaffold_groups(smiles_list):
    # identical to run_regression.scaffold_groups (inlined: that module drags
    # in lightgbm, which the chemprop-dev env does not have)
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


CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
ISO = ["CYP2D6", "CYP3A4"]
K = 5
MAX_EPOCHS = 60
PATIENCE = 10
BATCH = 64
VAL_FRAC = 0.1
D_HIDDEN = 300
D_FFN = 300
FEAT = featurizers.SimpleMoleculeMolGraphFeaturizer()


def build_points(df, y=None):
    pts = []
    for i, smi in enumerate(df["SMILES"]):
        t = None if y is None else y[i]
        pts.append(data.MoleculeDatapoint.from_smi(smi, t))
    return pts


def make_model(scaler=None):
    mp = nn.BondMessagePassing(d_v=FEAT.atom_fdim, d_e=FEAT.bond_fdim, d_h=D_HIDDEN,
                               bias=True, depth=4, dropout=0.0, undirected=True)
    agg = nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN(input_dim=mp.output_dim, hidden_dim=D_FFN,
                                     n_layers=2, n_tasks=2, dropout=0.1,
                                     criterion=BCELoss())
    return models.MPNN(mp, agg, ffn, batch_norm=False)


def train_one(fit_df, fit_y, ev_df, ev_y, tag, seed):
    tr_dset = data.MoleculeDataset(build_points(fit_df, fit_y), FEAT)
    va_dset = data.MoleculeDataset(build_points(ev_df, ev_y), FEAT)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=2)
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH, num_workers=2, shuffle=False)
    model = make_model()
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    seed_everything(seed)
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    print(f"[{tag}] epoch_run={trainer.current_epoch}", flush=True)
    return model


def predict_proba(model, smiles_df):
    pts = [data.MoleculeDatapoint.from_smi(s) for s in smiles_df["SMILES"]]
    dset = data.MoleculeDataset(pts, FEAT)
    loader = data.build_dataloader(dset, batch_size=256, num_workers=2, shuffle=False)
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d = batch[0], batch[1], batch[2]
            bmg.to(dev)
            y = model(bmg, V_d, X_d)
            outs.append(torch.sigmoid(y).detach().cpu().numpy())
    return np.vstack(outs)


def mcc(p, y, t):
    pr = (p >= t).astype(int)
    tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum()
    tn = ((pr == 0) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum()
    return float((tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12))


def main(seed):
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    Xte = pd.read_parquet(os.path.join(CACHE, "X_test.parquet"))
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                     emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab.reindex(Xtr["SMILES"])
    Yn = np.full((len(Xtr), 2), np.nan)
    for j, iso in enumerate(ISO):
        col = Y[f"{iso}_is_TDI"]
        Yn[:, j] = [np.nan if pd.isna(v) else float(v) for v in col]
    folds = make_folds(scaffold_groups(Xtr["SMILES"].tolist()), seed=7)

    oof = {i: np.full(len(Xtr), np.nan) for i in ISO}
    tepro = {i: [] for i in ISO}
    t0 = time.time()
    for f in range(K):
        lab_rows = np.where(~np.isnan(Yn).all(axis=1))[0]
        tr_rows = lab_rows[folds[lab_rows] != f]
        va_rows = lab_rows[folds[lab_rows] == f]
        rng = np.random.default_rng(7000 + seed * 100 + f)
        vi = rng.choice(len(tr_rows), size=int(len(tr_rows) * VAL_FRAC), replace=False)
        fit_idx, ev_idx = np.delete(tr_rows, vi), tr_rows[vi]
        model = train_one(Xtr.iloc[fit_idx], Yn[fit_idx], Xtr.iloc[ev_idx], Yn[ev_idx],
                          f"fold{f}", seed)
        P = predict_proba(model, Xtr.iloc[va_rows])
        for j, iso in enumerate(ISO):
            oof[iso][va_rows] = P[:, j]
        del model
        torch.cuda.empty_cache()
        print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)

    lab_rows = np.where(~np.isnan(Yn).all(axis=1))[0]
    rng = np.random.default_rng(500 + seed)
    vi = rng.choice(len(lab_rows), size=int(len(lab_rows) * VAL_FRAC), replace=False)
    fit_idx = np.delete(lab_rows, vi)
    model = train_one(Xtr.iloc[fit_idx], Yn[fit_idx], Xtr.iloc[lab_rows[vi]], Yn[lab_rows[vi]],
                      "all", seed)
    Pte = predict_proba(model, Xte)
    del model
    torch.cuda.empty_cache()

    for j, iso in enumerate(ISO):
        y = Y[f"{iso}_is_TDI"].values
        l = ~pd.isna(y)
        yl = np.array([1 if bool(v) else 0 for v in y[l]])
        p = oof[iso][l]
        best = max(((mcc(p, yl, np.quantile(p, 1 - fr)), fr) for fr in np.arange(0.05, 0.61, 0.01)))
        print(iso, "OOF best MCC", round(best[0], 4), "@frac", best[1], flush=True)
    sfx = "" if seed == 7 else f"_s{seed}"
    np.savez(os.path.join(CACHE, f"tdi_dmpnn{sfx}_oof.npz"), **oof)
    pd.DataFrame({"SMILES": Xte["SMILES"], **{f"{i}_proba": Pte[:, j] for j, i in enumerate(ISO)}}).to_csv(
        os.path.join(CACHE, f"tdi_dmpnn{sfx}_test_probs.csv"), index=False)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.seed)
