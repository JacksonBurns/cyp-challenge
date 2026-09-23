"""CheMeLeon multi-task fine-tune with OUR scaffold folds (chemeleon env).

One 4-head model (per-isoform direct pIC50, NaN labels masked by chemprop),
5 fold-models for OOF + 1 all-data model for test preds. Mirrors jeremy's
public approach (freeze/fine-tune switch, normalized targets, fresh FFN head).
Frozen-encoder head training first (his finding: fine-tuning hurt; ours to
re-verify). Writes cache/ft_oof.csv + cache/ft_test_preds.csv.

Run: ~/miniforge3/envs/chemeleon/bin/python src/ft_chemeleon.py [--full-ft]
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
WEIGHTS = os.path.expanduser("~/chemeleon_nazarov/chemeleon_mp.pt")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
MAX_EPOCHS = 60
PATIENCE = 10
BATCH = 64
VAL_FRAC = 0.1


def build_points(df):
    pts = []
    for _, row in df.iterrows():
        t = [float(row[c]) if pd.notna(row[c]) else float("nan") for c in ISO]
        pts.append(data.MoleculeDatapoint.from_smi(row["SMILES"], t))
    return pts


def train_fold(tr_df, va_df, full_ft, tag):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_dset = data.MoleculeDataset(build_points(tr_df), featurizer)
    scaler = tr_dset.normalize_targets()
    va_dset = data.MoleculeDataset(build_points(va_df), featurizer)
    va_dset.normalize_targets(scaler)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=0)
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH, num_workers=0, shuffle=False)

    st = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    if not full_ft:
        for p in mp.parameters():
            p.requires_grad = False
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=2048, n_layers=2,
                           n_tasks=len(ISO), dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    model = models.MPNN(mp, agg, ffn, batch_norm=False)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] trainable params {n_tr/1e6:.1f}M", flush=True)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    return model


def predict(model, smiles_df):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pts = [data.MoleculeDatapoint.from_smi(s, [float("nan")] * len(ISO)) for s in smiles_df["SMILES"]]
    dset = data.MoleculeDataset(pts, featurizer)
    loader = data.build_dataloader(dset, batch_size=BATCH, num_workers=0, shuffle=False)
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d = batch[0], batch[1], batch[2]
            bmg.to(dev)
            y = model(bmg, V_d, X_d)
            outs.append(np.atleast_2d(y.detach().cpu().numpy()))
    return np.vstack(outs)[:, : len(ISO)]


def main(full_ft):
    df = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    oof = pd.DataFrame(np.nan, index=df.index, columns=ISO)
    t0 = time.time()
    for f in range(5):
        tr_df = df[df.fold != f].reset_index(drop=True)
        va_df = df[df.fold == f].reset_index(drop=True)
        # internal val split for early stopping
        lab_tr = tr_df[ISO].notna().any(axis=1)
        rng = np.random.default_rng(f)
        idx = np.where(lab_tr.values)[0]
        val_i = rng.choice(idx, size=int(len(idx) * VAL_FRAC), replace=False)
        fit_df = tr_df.drop(index=val_i).reset_index(drop=True)
        ev_df = tr_df.iloc[val_i].reset_index(drop=True)
        model = train_fold(fit_df, ev_df, full_ft, f"fold{f}")
        P = predict(model, va_df)
        for j, iso in enumerate(ISO):
            oof.loc[df.fold == f, iso] = P[:, j]
        del model
        torch.cuda.empty_cache()
        print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)
    tag = "fullft" if full_ft else "frozen"
    oof.insert(0, "SMILES", df["SMILES"])
    oof.insert(1, "fold", df["fold"])
    oof.to_csv(os.path.join(CACHE, f"ft_oof_{tag}.csv"), index=False)

    # all-data model -> test preds
    lab = df[ISO].notna().any(axis=1)
    rng = np.random.default_rng(99)
    idx = np.where(lab.values)[0]
    val_i = rng.choice(idx, size=int(len(idx) * VAL_FRAC), replace=False)
    fit_df = df.drop(index=val_i).reset_index(drop=True)
    ev_df = df.iloc[val_i].reset_index(drop=True)
    model = train_fold(fit_df, ev_df, full_ft, "all")
    P = predict(model, test)
    tdf = pd.DataFrame(P, columns=ISO)
    tdf.insert(0, "SMILES", test["SMILES"])
    tdf.to_csv(os.path.join(CACHE, f"ft_test_preds_{tag}.csv"), index=False)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ft", action="store_true")
    main(ap.parse_args().full_ft)
