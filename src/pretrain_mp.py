"""Pretrain CheMeLeon MP on external CYP pIC50 multitask (ChEMBL + AID1851).

Population-shift caveat (jeremy README: ChEMBL is ~1-1.65 log more potent than
the challenge population): this affects FINE-TUNE target scale, not the encoder
representation learned here - the FT stage re-inits heads and renormalizes
targets from challenge rows only. To further hedge we train on the UNION of
ChEMBL IC50 and AID1851 Fit_LogAC50-derived pIC50 (4 heads, NaN-masked MSE),
which brackets both potency scales.

Challenge train/test SMILES are EXCLUDED from pretraining (leak-hygiene: our
OOF must stay honest even though external labels are external).

Writes cache/mp_pretrained_cyp.pt {state_dict, hyper_parameters, meta}.
Run (chemeleon env):
  ~/miniforge3/envs/chemeleon/bin/python src/pretrain_mp.py [--rows 20000]
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
CACHE = os.path.join(HERE, "..", "cache")
EXT = os.path.join(HERE, "..", "external")
WEIGHTS = os.path.expanduser("~/chemeleon_nazarov/chemeleon_mp.pt")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
BATCH = 128
MAX_EPOCHS = 30
PATIENCE = 5
VAL_FRAC = 0.1


def load_external_table(exclude):
    rows = []
    ch = pd.read_csv(os.path.join(EXT, "chembl_cyp_ic50.csv"))
    ch = ch[ch["standard_relation"] == "="]
    ch["standard_value"] = pd.to_numeric(ch["standard_value"], errors="coerce")
    u = ch["standard_units"].astype(str).str.strip().str.lower()
    ch = ch[u.isin(["nm"])].dropna(subset=["standard_value", "canonical_smiles"])
    ch = ch[ch["standard_value"] > 0]
    ch["pic50"] = 9.0 - np.log10(ch["standard_value"])
    g = ch.groupby(["isoform", "canonical_smiles"])["pic50"].median().reset_index()
    wide = g.pivot(index="canonical_smiles", columns="isoform", values="pic50")
    cdf = pd.DataFrame({i: (wide[i] if i in wide else np.nan) for i in ISO})
    cdf["SMILES"] = cdf.index
    rows.append(cdf.assign(_src="chembl"))
    pc = pd.read_csv(os.path.join(EXT, "pubchem_cyp_qhts_aid1851.csv"))
    pdf = pd.DataFrame({i: pd.to_numeric(pc.get(f"{i}_pIC50"), errors="coerce") for i in ISO})
    pdf["SMILES"] = pc["smiles"]
    pdf = pdf.dropna(subset=ISO, how="all")
    pdf = pdf.groupby("SMILES").first().reset_index()
    rows.append(pdf.assign(_src="pubchem"))
    tb = pd.concat(rows, ignore_index=True)
    tb = tb[tb["SMILES"].isin(exclude) == False]  # noqa: E712
    tb = tb.drop_duplicates("SMILES").reset_index(drop=True)
    return tb[tb[ISO].notna().any(axis=1)].reset_index(drop=True)


def main(rows_cap, seed=0):
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    exclude = set(ft["SMILES"]) | set(test["SMILES"])
    tb = load_external_table(exclude)
    if len(tb) > rows_cap:
        rng = np.random.default_rng(seed)
        tb = tb.iloc[rng.choice(len(tb), rows_cap, replace=False)]
    print("pretrain rows:", len(tb), {i: int(tb[i].notna().sum()) for i in ISO}, flush=True)

    idx = np.arange(len(tb))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_tr = int(len(tb) * (1 - VAL_FRAC))
    tr_df = tb.iloc[idx[:n_tr]]
    va_df = tb.iloc[idx[n_tr:]]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def pts(df):
        out = []
        for _, r in df.iterrows():
            t = [float(r[c]) if pd.notna(r[c]) else float("nan") for c in ISO]
            out.append(data.MoleculeDatapoint.from_smi(r["SMILES"], t))
        return out

    tr_dset = data.MoleculeDataset(pts(tr_df), featurizer)
    scaler = tr_dset.normalize_targets()
    va_dset = data.MoleculeDataset(pts(va_df), featurizer)
    va_dset.normalize_targets(scaler)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=0)
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH, num_workers=0, shuffle=False)

    st = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=2048, n_layers=2,
                           n_tasks=len(ISO), dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
                           criterion=MSE())
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    print("trainable", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, flush=True)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    seed_everything(seed)
    t0 = time.time()
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    mp_state = model.message_passing.state_dict()
    torch.save({"state_dict": {k: v.cpu() for k, v in mp_state.items()},
                "hyper_parameters": st["hyper_parameters"],
                "meta": {"rows": len(tb), "epochs_run": trainer.current_epoch,
                         "sources": "chembl+pubchem", "seed": seed}},
               os.path.join(CACHE, "mp_pretrained_cyp.pt"))
    print("DONE pretrain", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    a = ap.parse_args()
    main(a.rows)
