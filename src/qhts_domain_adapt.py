"""R1: QHTS-ONLY domain adaptation then freeze + embed (jeremy's best 2D6 recipe).

CRITICAL anti-leak design: the domain adaptation trains the encoder on qHTS
ONLY (AID1851 pIC50 as the adaptation task) - the encoder NEVER sees challenge
labels. That is what makes the downstream OOF ridge honest: the "frozen"
adapted encoder carries no challenge-label information, so the OOF Pearson is a
real number (not the 0.80+ leaked artifact of training the DA on challenge).

Flow: (1) full-FT the frozen chemprop_medium encoder (admecd_ckpt_med.pt) on
qHTS rows only (4 pIC50 heads, the domain-shift signal); (2) FREEZE the adapted
encoder; (3) extract 600-d mp+agg embeddings for all train+test SMILES ->
cache/emb_qhtsda_all.parquet; (4) ridge probe (src/qhtsda_ridge.py, OOF scaffold
folds) gates it: beat chmridge floor 0.468/0.585/0.344/0.723 on >= 2 isoforms,
then family-block as a NEW family.

GPU job (chemprop-dev env). One tracked process.
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
PRETRAINED = os.path.join(CACHE, "admecd_ckpt_med.pt")  # frozen chemprop_medium
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
MAX_EPOCHS = 30
PATIENCE = 6
BATCH = 128
VAL_FRAC = 0.1
QHTS_W = 0.5  # qHTS adaptation weight (real targets, not 0.3 aux)
EXT_CAP = 9000


def load_qhts_only():
    """challenge + AID1851 qHTS (no ChEMBL). qHTS pIC50 as real 4-head targets."""
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    ch = ft.copy()
    ch["_src"] = "challenge"
    pc = pd.read_csv(os.path.join(EXT, "pubchem_cyp_qhts_aid1851.csv"))
    pdf = pd.DataFrame({i: pd.to_numeric(pc.get(f"{i}_pIC50"), errors="coerce") for i in ISO})
    pdf["SMILES"] = pc["smiles"]
    pdf = pdf.dropna(how="all").groupby("SMILES").first().reset_index()
    pdf["_src"] = "qhts"
    allrows = pd.concat([ch, pdf], ignore_index=True)
    ch_smiles = set(ft["SMILES"]) | set(test["SMILES"])
    allrows = allrows[~allrows["SMILES"].isin(ch_smiles) | (allrows["_src"] == "challenge")]
    allrows = allrows.drop_duplicates("SMILES").reset_index(drop=True)
    return ft, test, allrows


def build_points(df, cols):
    pts = []
    for _, row in df.iterrows():
        t = [float(row[c]) if pd.notna(row[c]) else float("nan") for c in cols]
        pts.append(data.MoleculeDatapoint.from_smi(row["SMILES"], t))
    return pts


def train_model(tr_df, va_df, cols, wts, seed=0):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_dset = data.MoleculeDataset(build_points(tr_df, cols), featurizer)
    scaler = tr_dset.normalize_targets()
    va_dset = data.MoleculeDataset(build_points(va_df, cols), featurizer)
    va_dset.normalize_targets(scaler)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=0)
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH, num_workers=0, shuffle=False)
    st = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=2048, n_layers=2,
                           n_tasks=len(cols), dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
                           criterion=MSE(task_weights=wts))
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    print(f"  trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    seed_everything(seed)
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    return model


def embed_all(model, smiles):
    """Freeze + extract the mp+agg 600-d embeddings for a list of SMILES."""
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pts = [data.MoleculeDatapoint.from_smi(s) for s in smiles]
    dset = data.MoleculeDataset(pts, featurizer)
    loader = data.build_dataloader(dset, batch_size=64, num_workers=0, shuffle=False)
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d = batch[0], batch[1], batch[2]
            bmg.to(dev)
            H = model.agg(model.message_passing(bmg, V_d), bmg.batch)
            outs.append(H.detach().cpu().numpy())
    return np.vstack(outs)


def main(seed):
    ft, test, allrows = load_qhts_only()
    cols = ISO  # 4 pIC50 heads
    lab = allrows[cols].notna().any(axis=1)
    ch_mask = (allrows["_src"] == "challenge").values
    qhts_rows = allrows[(~ch_mask) & lab.values].reset_index(drop=True)
    print(f"rows: {len(allrows)} challenge: {ch_mask.sum()} qhts-labeled: {len(qhts_rows)}", flush=True)

    # DOMAIN ADAPTATION: train the encoder on qHTS ONLY (no challenge labels).
    # qHTS pIC50 are the adaptation targets (the domain-shift signal). The
    # encoder never sees challenge labels -> downstream OOF ridge stays honest.
    # Equal head weights (qHTS is the whole task here).
    wts = np.ones(len(cols))
    rng = np.random.default_rng(99 + seed)
    if len(qhts_rows) > EXT_CAP:
        qhts_fit = qhts_rows.iloc[rng.choice(len(qhts_rows), EXT_CAP, replace=False)].reset_index(drop=True)
    else:
        qhts_fit = qhts_rows
    # small internal val for early stopping (qHTS-only, 10%)
    vi = rng.choice(len(qhts_fit), size=int(len(qhts_fit) * VAL_FRAC), replace=False)
    fit_df = qhts_fit.drop(index=vi).reset_index(drop=True)
    ev_df = qhts_fit.iloc[vi].reset_index(drop=True)
    model = train_model(fit_df, ev_df, cols, wts, seed=seed)

    # FREEZE + embed all train+test SMILES
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES")
    smi_all = list(pd.unique(pd.concat([Xtr["SMILES"], test["SMILES"]])))
    print(f"embedding {len(smi_all)} SMILES...", flush=True)
    E = embed_all(model, smi_all)
    dall = pd.DataFrame(E, index=pd.Index(smi_all, name="SMILES")).reset_index()
    dall.to_parquet(os.path.join(CACHE, "emb_qhtsda_all.parquet"))
    print(f"saved emb_qhtsda_all.parquet {dall.shape}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    main(a.seed)
    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)
