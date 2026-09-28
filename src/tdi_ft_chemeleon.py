"""CheMeLeon encoder fine-tuned on the TDI BINARY task (Task 1, sec 18 queue).

Copy of src/tdi_dmpnn.py (2-head BCE is_TDI on CYP2D6/3A4, challenge rows
ONLY, TDI+Emax label union, scaffold folds seed 7) with the encoder swapped:
nn.BondMessagePassing loaded from ~/chemeleon_nazarov/chemeleon_mp.pt via the
ft_chemeleon.py pattern (weights_only=False, hyper_parameters, 2-head
BinaryClassificationFFN). --variant fullft tunes the encoder; --variant frozen
trains only the FFN head. d_h comes from the checkpoint (2048), not 300.

Run (chemeleon env, GPU):
  ~/miniforge3/envs/chemeleon/bin/python src/tdi_ft_chemeleon.py --seed 0 --variant fullft
Writes cache/tdi_ft_chm{,_frozen}{sfx}_oof.npz + _test_probs.csv
(same schema as tdi_dmpnn_oof.npz: per-iso proba arrays, NaN where unlabeled).
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
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
WEIGHTS = os.path.expanduser("~/chemeleon_nazarov/chemeleon_mp.pt")

ISO = ["CYP2D6", "CYP3A4"]
K = 5
MAX_EPOCHS = 60
PATIENCE = 10
BATCH = 64
VAL_FRAC = 0.1
FFN_HIDDEN = 512


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


FEAT = featurizers.SimpleMoleculeMolGraphFeaturizer()


def build_points(df, y=None):
    pts = []
    for i, smi in enumerate(df["SMILES"]):
        t = None if y is None else y[i]
        pts.append(data.MoleculeDatapoint.from_smi(smi, t))
    return pts


def load_encoder(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    return mp


def make_model(weights_path, full_ft):
    mp = load_encoder(weights_path)
    if not full_ft:
        for p in mp.parameters():
            p.requires_grad = False
    ffn = nn.BinaryClassificationFFN(input_dim=mp.output_dim, hidden_dim=FFN_HIDDEN,
                                     n_layers=2, n_tasks=2, dropout=0.1,
                                     criterion=BCELoss())
    return models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)


def train_one(fit_df, fit_y, ev_df, ev_y, tag, seed, weights_path, full_ft):
    tr_dset = data.MoleculeDataset(build_points(fit_df, fit_y), FEAT)
    va_dset = data.MoleculeDataset(build_points(ev_df, ev_y), FEAT)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=2)
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH, num_workers=2, shuffle=False)
    model = make_model(weights_path, full_ft)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] trainable params {n_tr/1e6:.1f}M", flush=True)
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
    loader = data.build_dataloader(dset, batch_size=128, num_workers=2, shuffle=False)
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


def main(seed, variant, weights_path):
    full_ft = variant == "fullft"
    base = "tdi_ft_chm" if full_ft else "tdi_ft_chm_frozen"
    sfx = "" if seed == 0 else f"_s{seed}"
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
    t0 = time.time()
    for f in range(K):
        lab_rows = np.where(~np.isnan(Yn).all(axis=1))[0]
        tr_rows = lab_rows[folds[lab_rows] != f]
        va_rows = lab_rows[folds[lab_rows] == f]
        rng = np.random.default_rng(7000 + seed * 100 + f)
        vi = rng.choice(len(tr_rows), size=int(len(tr_rows) * VAL_FRAC), replace=False)
        fit_idx, ev_idx = np.delete(tr_rows, vi), tr_rows[vi]
        model = train_one(Xtr.iloc[fit_idx], Yn[fit_idx], Xtr.iloc[ev_idx], Yn[ev_idx],
                          f"fold{f}", seed, weights_path, full_ft)
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
                      "all", seed, weights_path, full_ft)
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
    np.savez(os.path.join(CACHE, f"{base}{sfx}_oof.npz"), **oof)
    pd.DataFrame({"SMILES": Xte["SMILES"], **{f"{i}_proba": Pte[:, j] for j, i in enumerate(ISO)}}).to_csv(
        os.path.join(CACHE, f"{base}{sfx}_test_probs.csv"), index=False)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=["fullft", "frozen"], default="fullft")
    ap.add_argument("--weights", default=WEIGHTS)
    a = ap.parse_args()
    main(a.seed, a.variant, a.weights)
