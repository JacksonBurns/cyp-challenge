"""Phase 1 (NEXT_ITER_PLAN sec 4) - Chemprop D-MPNN with the x_d mordred channel.

Clean 2x2 factorial, all on the SAME scaffold folds (ft_data.csv 'fold', seed 0),
SAME external-aux table as ft_ext/ft_dmpnn (challenge + ChEMBL + AID1851,
label-separated, EXT_CAP), SAME architecture / LR schedule / early-stopping as
ft_dmpnn so the ONLY variables are aux heads and x_d:

  V1  control:    challenge-only D-MPNN, no x_d          (bare control)
  V2  aux:        challenge + external aux heads, no x_d  (= ft_dmpnn, CACHED)
  V3  x_d:        challenge-only D-MPNN + mordred as x_d  (the NEW lever, clean)
  V4  aux + x_d:  challenge + external aux heads + mordred as x_d  (the bet)

x_d mechanism (chemprop 2.3.1): set dp.x_d = tensor(n_mordred) on each
MoleculeDatapoint; the dataloader batches it to X_d [B, n_mordred];
MPNN.fingerprint returns cat(H, X_d_transform(X_d)) so the FFN input_dim must be
mp.output_dim + n_mordred. V1/V2 leave x_d=None -> input_dim = mp.output_dim.

Anti-leak: the mordred matrix is imputed + standardized on CHALLENGE TRAIN only
(src/mordred_features.py / _ext.py); x_d is a raw input feature (no label info).

GATE 1: V4 (or V3) must beat V2 (cached ft_dmpnn) on nested macro R2 by >= +0.003.
Eval: per-isoform OOF Pearson + nested macro R2 (repo honest metric). Winner also
on Butina folds (Phase 1b, separate run).

Run (chemeleon env, one variant at a time):
  ~/miniforge3/envs/chemeleon/bin/python src/chemprop_xd.py --variant V3 --seed 7
Outputs cache/ft_oof_xd{V1|V3|V4}.csv + cache/ft_test_preds_xd{V1|V3|V4}.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from chemprop.nn.metrics import MSE
from lightning import pytorch as pl, seed_everything
from lightning.pytorch.callbacks import EarlyStopping

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ft_ext import COLS, ISO, load_tables  # noqa: E402

CACHE = os.path.join(HERE, "..", "cache")
# IDENTICAL architecture / schedule to ft_dmpnn (the V2 baseline)
MAX_EPOCHS = 60
PATIENCE = 10
BATCH = 64
VAL_FRAC = 0.1
AUX_W = 0.3
EXT_CAP = 9000
D_HIDDEN = 500
D_FFN = 2560
FEAT = featurizers.SimpleMoleculeMolGraphFeaturizer()


def load_xd():
    """Combined SMILES -> x_d(1429) lookup from challenge + external mordred matrices."""
    parts = [
        os.path.join(CACHE, "emb_mordred_all.parquet"),
        os.path.join(CACHE, "emb_mordred_ext.parquet"),
    ]
    xdict = {}
    for p in parts:
        df = pd.read_parquet(p)
        if "SMILES" in df.columns:
            df = df.set_index("SMILES")
        cols = list(df.columns)
        for smi in df.index:
            xdict[smi] = df.loc[smi, cols].to_numpy(dtype=np.float32)
    return xdict


def build_points_fast(df, xdict, use_xd, cols):
    """Vectorized: extract the target matrix once, then build datapoints.
    Test set (ft_test_smiles.csv) has no target columns -> all-NaN targets."""
    smis = df["SMILES"].to_numpy()
    if all(c in df.columns for c in cols):
        T = df[cols].to_numpy(dtype=np.float64)  # NaN for unlabeled
    else:
        T = np.full((len(smis), len(cols)), np.nan)
    pts = []
    for i in range(len(smis)):
        dp = data.MoleculeDatapoint.from_smi(smis[i], T[i].tolist())
        if use_xd:
            dp.x_d = torch.tensor(xdict[smis[i]], dtype=torch.float32)
        pts.append(dp)
    return pts


def make_model(n_tasks, scaler, wts, use_xd, n_xd):
    mp = nn.BondMessagePassing(d_v=FEAT.atom_fdim, d_e=FEAT.bond_fdim, d_h=D_HIDDEN,
                               bias=True, depth=4, dropout=0.0, undirected=True)
    agg = nn.MeanAggregation()
    input_dim = mp.output_dim + (n_xd if use_xd else 0)
    ffn = nn.RegressionFFN(input_dim=input_dim, hidden_dim=D_FFN, n_layers=2,
                           n_tasks=n_tasks, dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
                           criterion=MSE(task_weights=wts))
    return models.MPNN(mp, agg, ffn, batch_norm=False)


def task_weights(counts, n_prim):
    prim_scale = 1.0 / np.maximum(counts[:n_prim], 1)
    prim_scale = prim_scale / prim_scale.mean()
    return np.concatenate([prim_scale, np.full(len(counts) - n_prim, AUX_W)])


def train_one(fit_df, ev_df, tag, seed, use_xd, n_xd, xdict, cols):
    tr_pts = build_points_fast(fit_df, xdict, use_xd, cols)
    tr_dset = data.MoleculeDataset(tr_pts, FEAT)
    scaler = tr_dset.normalize_targets()
    va_pts = build_points_fast(ev_df, xdict, use_xd, cols)
    va_dset = data.MoleculeDataset(va_pts, FEAT)
    va_dset.normalize_targets(scaler)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=2)
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH, num_workers=2, shuffle=False)
    wts = task_weights(fit_df[cols].notna().sum().values.astype(float), 4)
    model = make_model(len(cols), scaler, wts, use_xd, n_xd)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    seed_everything(seed)
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    print(f"[{tag}] epoch_run={trainer.current_epoch}", flush=True)
    return model


def predict(model, df, xdict, use_xd, cols):
    pts = build_points_fast(df, xdict, use_xd, cols)
    dset = data.MoleculeDataset(pts, FEAT)
    loader = data.build_dataloader(dset, batch_size=256, num_workers=2, shuffle=False)
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d = batch[0], batch[1], batch[2]
            bmg.to(dev)
            if X_d is not None:
                X_d = X_d.to(dev)
            y = model(bmg, V_d, X_d)
            outs.append(np.atleast_2d(y.detach().cpu().numpy()))
    return np.vstack(outs)[:, :4]  # first 4 = primary challenge isoforms


def main(variant, seed, max_folds=5):
    use_xd = variant in ("V3", "V4")
    use_aux = variant in ("V2", "V4")
    n_xd = 1429
    print(f"variant {variant} | use_xd={use_xd} use_aux={use_aux} seed={seed} max_folds={max_folds}", flush=True)
    xdict = load_xd() if use_xd else None
    if use_xd:
        print(f"  xdict: {len(xdict)} SMILES -> x_d({n_xd})", flush=True)

    ft, test, allrows = load_tables()
    lab = allrows[COLS].notna().any(axis=1)
    ch_mask = (allrows["_src"] == "challenge").values
    cols = COLS  # 12 tasks (aux); primary = first 4
    folds = ft["fold"].values
    oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    t0 = time.time()

    for f in range(min(max_folds, 5)):
        va_idx = np.where(folds == f)[0]
        va_df = allrows[ch_mask].iloc[va_idx]
        tr_pool = allrows[ch_mask].drop(index=va_idx).reset_index(drop=True)
        if use_aux:
            tr_extra = allrows[~ch_mask & lab]
            rng = np.random.default_rng(7000 + seed * 100 + f)
            if len(tr_extra) > EXT_CAP:
                keep = rng.choice(len(tr_extra), EXT_CAP, replace=False)
                tr_extra = tr_extra.iloc[keep]
            lab_tr = tr_pool[COLS].notna().any(axis=1).values
            vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC), replace=False)
            fit_df = pd.concat([tr_pool.drop(index=vi), tr_extra], ignore_index=True)
        else:
            # challenge-only: 10% of the (non-val) challenge pool as val split
            rng = np.random.default_rng(7000 + seed * 100 + f)
            lab_tr = tr_pool[COLS].notna().any(axis=1).values
            vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC), replace=False)
            fit_df = tr_pool.drop(index=vi)
        ev_df = tr_pool.iloc[vi]
        model = train_one(fit_df, ev_df, f"fold{f}", seed, use_xd, n_xd, xdict, cols)
        P = predict(model, va_df, xdict, use_xd, cols)
        for j, iso in enumerate(ISO):
            oof.loc[va_idx, iso] = P[:, j]
        del model
        torch.cuda.empty_cache()
        print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)

    oof.insert(0, "SMILES", ft["SMILES"])
    oof_sfx = "" if max_folds >= 5 else "_smoke"
    oof.to_csv(os.path.join(CACHE, f"ft_oof_xd{variant}{oof_sfx}.csv"), index=False)

    if max_folds < 5:
        print(f"SMOKE TEST only (max_folds={max_folds}); wrote {oof_sfx} oof, skipping test preds", flush=True)
        return

    # test preds: fit on all (challenge [+ external]) with a val split
    tr_pool = allrows[ch_mask].reset_index(drop=True)
    rng = np.random.default_rng(500 + seed)
    l = tr_pool[COLS].notna().any(axis=1).values
    vi = rng.choice(np.where(l)[0], size=int(l.sum() * VAL_FRAC), replace=False)
    if use_aux:
        ext = allrows[(~ch_mask) & lab.values]
        if len(ext) > EXT_CAP:
            ext = ext.iloc[rng.choice(len(ext), EXT_CAP, replace=False)]
        fit_df = pd.concat([tr_pool.drop(index=vi), ext], ignore_index=True)
    else:
        fit_df = tr_pool.drop(index=vi)
    model = train_one(fit_df, tr_pool.iloc[vi], "all", seed, use_xd, n_xd, xdict, cols)
    P = predict(model, test, xdict, use_xd, cols)
    tdf = pd.DataFrame(P, columns=ISO)
    tdf.insert(0, "SMILES", test["SMILES"])
    tdf.to_csv(os.path.join(CACHE, f"ft_test_preds_xd{variant}.csv"), index=False)
    print("DONE", variant, time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["V1", "V3", "V4"], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-folds", type=int, default=5)
    a = ap.parse_args()
    main(a.variant, a.seed, max_folds=a.max_folds)
