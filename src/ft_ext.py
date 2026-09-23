"""CheMeLeon multitask fine-tune WITH external auxiliary heads (jeremy's pattern).

Heads: 4 primary challenge pIC50 (inverse-count weights) + 4 ChEMBL IC50 pIC50
+ 4 PubChem AID1851 pIC50-like (flat 0.3x weight each). External compounds
(ChEMBL/AID rows) join TRAINING with only their aux head labeled - the encoder
sees ~50k extra molecules while primary heads stay challenge-only. Validate on
OUR scaffold folds; OOF must beat cache/oof_pearson.json to earn blend weight.

Writes cache/ft_oof_ext.csv + cache/ft_test_preds_ext.csv
Run (chemeleon env):
  ~/miniforge3/envs/chemeleon/bin/python src/ft_ext.py [--seed 0] [--full-ft]
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from chemprop.nn.metrics import MSE
from lightning import pytorch as pl, seed_everything
from lightning.pytorch.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
EXT = os.path.join(HERE, "..", "external")
WEIGHTS = os.path.expanduser("~/chemeleon_nazarov/chemeleon_mp.pt")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
PRIMARY = ISO
AUX = [f"{i}_chembl" for i in ISO] + [f"{i}_pubchem" for i in ISO]
COLS = PRIMARY + AUX
MAX_EPOCHS = 40
PATIENCE = 8
BATCH = 128
VAL_FRAC = 0.1
AUX_W = 0.3
EXT_CAP = 9000  # subsample external rows per fold (throughput cap)


def load_tables():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    rows = [ft.assign(_src="challenge")]
    ch = pd.read_csv(os.path.join(EXT, "chembl_cyp_ic50.csv"))
    ch = ch[ch["standard_relation"] == "="]
    ch["standard_value"] = pd.to_numeric(ch["standard_value"], errors="coerce")
    u = ch["standard_units"].astype(str).str.strip().str.lower()
    ch = ch[u.isin(["nm"])].dropna(subset=["standard_value", "canonical_smiles"])
    ch["pic50"] = 9.0 - np.log10(ch["standard_value"])
    g = ch.groupby(["isoform", "canonical_smiles"])["pic50"].median().reset_index()
    wide = g.pivot(index="canonical_smiles", columns="isoform", values="pic50")
    cdf = pd.DataFrame({f"{i}_chembl": wide[i] if i in wide else np.nan for i in ISO})
    cdf["SMILES"] = cdf.index
    rows.append(cdf.assign(_src="chembl"))
    pc = pd.read_csv(os.path.join(EXT, "pubchem_cyp_qhts_aid1851.csv"))
    pdf = pd.DataFrame({f"{i}_pubchem": pd.to_numeric(pc.get(f"{i}_pIC50"), errors="coerce")
                        for i in ISO})
    pdf["SMILES"] = pc["smiles"]
    pdf = pdf.dropna(subset=[c for c in pdf.columns if c != "SMILES"], how="all")
    pdf = pdf.groupby("SMILES").first().reset_index()
    rows.append(pdf.assign(_src="pubchem"))
    allrows = pd.concat(rows, ignore_index=True)
    for c in COLS:
        if c not in allrows:
            allrows[c] = np.nan
    ch_smiles = set(ft["SMILES"]) | set(test["SMILES"])
    allrows = allrows[~allrows["SMILES"].isin(ch_smiles) | (allrows["_src"] == "challenge")]
    allrows = allrows.drop_duplicates("SMILES").reset_index(drop=True)
    return ft, test, allrows


def build_points(df):
    pts = []
    for _, row in df.iterrows():
        t = [float(row[c]) if pd.notna(row[c]) else float("nan") for c in COLS]
        pts.append(data.MoleculeDatapoint.from_smi(row["SMILES"], t))
    return pts


def train_model(tr_df, va_df, full_ft, tag, seed=0):
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
    counts = tr_df[COLS].notna().sum().values.astype(float)
    prim_scale = (1.0 / np.maximum(counts[:4], 1))
    prim_scale = prim_scale / prim_scale.mean()
    wts = np.concatenate([prim_scale, np.full(len(AUX), AUX_W)])
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=2048, n_layers=2,
                           n_tasks=len(COLS), dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
                           criterion=MSE(task_weights=wts))
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    print(f"[{tag}] trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    seed_everything(seed)
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    return model


def predict(model, smiles_df):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pts = [data.MoleculeDatapoint.from_smi(s, [float("nan")] * len(COLS)) for s in smiles_df["SMILES"]]
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
    return np.vstack(outs)[:, : len(COLS)]


def main(full_ft, seed):
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
        rng = np.random.default_rng(1000 * seed + f)
        if len(tr_extra) > EXT_CAP:
            keep = rng.choice(len(tr_extra), EXT_CAP, replace=False)
            tr_extra = tr_extra.iloc[keep]
        lab_tr = tr_pool[COLS].notna().any(axis=1).values
        vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC), replace=False)
        fit_df = pd.concat([tr_pool.drop(index=vi), tr_extra], ignore_index=True)
        ev_df = tr_pool.iloc[vi]
        model = train_model(fit_df, ev_df, full_ft, f"fold{f}", seed=seed)
        P = predict(model, va_df)
        for j, iso in enumerate(ISO):
            oof.loc[va_idx, iso] = P[:, j]
        del model
        torch.cuda.empty_cache()
        print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)
    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, "ft_oof_ext.csv"), index=False)

    lab_all = allrows[COLS].notna().any(axis=1)
    tr_pool = allrows[ch_mask].reset_index(drop=True)
    rng = np.random.default_rng(99 + seed)
    l = tr_pool[COLS].notna().any(axis=1).values
    vi = rng.choice(np.where(l)[0], size=int(l.sum() * VAL_FRAC), replace=False)
    ext = allrows[~ch_mask.values & lab_all.values]
    if len(ext) > EXT_CAP:
        ext = ext.iloc[rng.choice(len(ext), EXT_CAP, replace=False)]
    fit_df = pd.concat([tr_pool.drop(index=vi), ext], ignore_index=True)
    model = train_model(fit_df, tr_pool.iloc[vi], full_ft, "all", seed=seed)
    P = predict(model, test)
    tdf = pd.DataFrame(P[:, :4], columns=ISO)
    tdf.insert(0, "SMILES", test["SMILES"])
    tdf.to_csv(os.path.join(CACHE, "ft_test_preds_ext.csv"), index=False)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ft", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.full_ft, a.seed)
