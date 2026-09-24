"""D-MPNN multitask (chemprop) with the same external-aux heads as ft_ext.

Members: 4 challenge primary pIC50 heads + 4 ChEMBL + 4 AID1851 aux heads
(AUX_W flat), same table builder as src/ft_ext.load_tables, same scaffold
folds (cache/ft_data.csv 'fold'). Distinguishing featurizer vs CheMeLeon:
MultiHotAtomFeaturizer + BondMessagePassing D-MPNN, ensemble_size=1 per fit.
OOF written for eval_ft.py (tag dmpnn); separate test preds for the blend.

Run (chemprop-dev env):
  ~/miniforge3/envs/chemprop-dev/bin/python src/ft_dmpnn.py --seed 7
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from chemprop.nn.metrics import MSE
from lightning import pytorch as pl, seed_everything
from lightning.pytorch.callbacks import EarlyStopping

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ft_ext import COLS, ISO, load_tables  # noqa: E402

CACHE = os.path.join(HERE, "..", "cache")
MAX_EPOCHS = 60
PATIENCE = 10
BATCH = 64
VAL_FRAC = 0.1
AUX_W = 0.3
EXT_CAP = 9000
D_HIDDEN = 500
D_FFN = 2560
FEAT = featurizers.SimpleMoleculeMolGraphFeaturizer()


def build_points(df):
    pts = []
    for _, row in df.iterrows():
        t = [float(row[c]) if pd.notna(row[c]) else float("nan") for c in COLS]
        pts.append(data.MoleculeDatapoint.from_smi(row["SMILES"], t))
    return pts


def make_model(n_tasks, scaler, wts=None):
    mp = nn.BondMessagePassing(d_v=FEAT.atom_fdim, d_e=FEAT.bond_fdim, d_h=D_HIDDEN,
                               bias=True, depth=4, dropout=0.0, undirected=True)
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=D_FFN, n_layers=2,
                           n_tasks=n_tasks, dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
                           criterion=MSE(task_weights=wts))
    return models.MPNN(mp, agg, ffn, batch_norm=False)


def task_weights(tr_df):
    counts = tr_df[COLS].notna().sum().values.astype(float)
    n_prim = 4
    prim_scale = 1.0 / np.maximum(counts[:n_prim], 1)
    prim_scale = prim_scale / prim_scale.mean()
    return np.concatenate([prim_scale, np.full(len(COLS) - n_prim, AUX_W)])


def train_one(fit_df, ev_df, tag, seed):
    tr_pts = build_points(fit_df)
    tr_dset = data.MoleculeDataset(tr_pts, FEAT)
    scaler = tr_dset.normalize_targets()
    va_pts = build_points(ev_df)
    va_dset = data.MoleculeDataset(va_pts, FEAT)
    va_dset.normalize_targets(scaler)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=2)
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH, num_workers=2, shuffle=False)
    model = make_model(len(COLS), scaler, wts=task_weights(fit_df))
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    seed_everything(seed)
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    print(f"[{tag}] epoch_run={trainer.current_epoch}", flush=True)
    return model


def predict(model, smiles_df):
    pts = [data.MoleculeDatapoint.from_smi(s, [float("nan")] * len(COLS)) for s in smiles_df["SMILES"]]
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
            outs.append(np.atleast_2d(y.detach().cpu().numpy()))
    return np.vstack(outs)[:, :len(COLS)]


def main(seed):
    ft, test, allrows = load_tables()
    lab = allrows[COLS].notna().any(axis=1)
    ch_mask = (allrows["_src"] == "challenge").values
    print("rows:", len(allrows), "challenge:", ch_mask.sum(), flush=True)
    oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    folds = ft["fold"].values
    t0 = time.time()
    for f in range(5):
        va_idx = np.where(folds == f)[0]
        va_df = allrows[ch_mask].iloc[va_idx]
        tr_pool = allrows[ch_mask].drop(index=va_idx).reset_index(drop=True)
        tr_extra = allrows[~ch_mask & lab]
        rng = np.random.default_rng(7000 + seed * 100 + f)
        if len(tr_extra) > EXT_CAP:
            keep = rng.choice(len(tr_extra), EXT_CAP, replace=False)
            tr_extra = tr_extra.iloc[keep]
        lab_tr = tr_pool[COLS].notna().any(axis=1).values
        vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC), replace=False)
        fit_df = pd.concat([tr_pool.drop(index=vi), tr_extra], ignore_index=True)
        model = train_one(fit_df, tr_pool.iloc[vi], f"fold{f}", seed)
        P = predict(model, va_df)
        for j, iso in enumerate(ISO):
            oof.loc[va_idx, iso] = P[:, j]
        del model
        torch.cuda.empty_cache()
        print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)
    sfx = "" if seed == 7 else f"_s{seed}"
    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, f"ft_oof_dmpnn{sfx}.csv"), index=False)

    tr_pool = allrows[ch_mask].reset_index(drop=True)
    rng = np.random.default_rng(500 + seed)
    l = tr_pool[COLS].notna().any(axis=1).values
    vi = rng.choice(np.where(l)[0], size=int(l.sum() * VAL_FRAC), replace=False)
    ext = allrows[(~ch_mask) & lab.values]
    if len(ext) > EXT_CAP:
        ext = ext.iloc[rng.choice(len(ext), EXT_CAP, replace=False)]
    fit_df = pd.concat([tr_pool.drop(index=vi), ext], ignore_index=True)
    model = train_one(fit_df, tr_pool.iloc[vi], "all", seed)
    P = predict(model, test)
    tdf = pd.DataFrame(P[:, :4], columns=ISO)
    tdf.insert(0, "SMILES", test["SMILES"])
    tdf.to_csv(os.path.join(CACHE, f"ft_test_preds_dmpnn{sfx}.csv"), index=False)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.seed)
