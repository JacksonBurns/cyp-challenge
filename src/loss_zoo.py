"""Loss zoo: interval_arm.py trainer, primary-head loss swapped per --mode.

Theory (LEDGER.md): the pool's binding constraint is DECORRELATION, not
accuracy; loss-axis diversity beats seed-axis diversity (intarm evidence:
standalone WORSE, best pool addition we've made). This script measures that
by training identical data/arch with different primary objectives:

  hinge : squared hinge outside RAW credible interval == intarm (control)
  mae   : plain |z - y| on labeled primary rows (loss-diversity control)
  rank  : squared hinge outside interval + pairwise logistic ranking on
          labeled primary pairs within each isoform (metric-native: zero
          cost inside the interval + ranking pressure inside it)
  mse   : plain z-space MSE on labeled primary rows

Aux heads stay MSE at --auxw (sweep tests the aux-regularization effect
size on the combiner gate). Output tags: zoo_<mode>[_w<AUXW>] so each arm
is its own family in the hinge_blend pool glob (ft_oof_zoo_*.csv).

Gate: NOT standalone score. Add to cp18i pool, run hinge_blend --pool,
read marginal per-isoform CV gain (intarm reference: +0.003 on 3A4,
flips 2C9/2D6 gates).

Run (chemeleon env, ONE GPU job at a time):
  ~/miniforge3/envs/chemeleon/bin/python src/loss_zoo.py --mode mae
  ~/miniforge3/envs/chemeleon/bin/python src/loss_zoo.py --mode rank
  ~/miniforge3/envs/chemeleon/bin/python src/loss_zoo.py --mode hinge --auxw 0.0
  ~/miniforge3/envs/chemeleon/bin/python src/loss_zoo.py --mode hinge --auxw 1.0
  ~/miniforge3/envs/chemeleon/bin/python src/loss_zoo.py --mode hinge --auxw 3.0
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from chemprop import data, models, nn
from chemprop.nn.metrics import MSE
from lightning import pytorch as pl, seed_everything
from lightning.pytorch.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import ft_ext  # noqa: E402
from interval_arm import bounds_by_smi, build_points_ext, soft_rae, ISO, NPRIM, NAUX, BOUNDC, LO_SL, HI_SL  # noqa: E402

COLS = ft_ext.COLS + BOUNDC
NALL = len(COLS)
NAUXD = NAUX
LO_SL = slice(NPRIM + NAUX, NPRIM + NAUX + 8, 2)
HI_SL = slice(NPRIM + NAUX + 1, NPRIM + NAUX + 8, 2)
LABCOLS = ft_ext.COLS
HOM = ["CYP2C19_hom", "CYP2C8_hom", "CYP2B6_hom", "CYP2A6_hom"]
HOM_CSV = os.path.join(HERE, "..", "external", "chembl_cyp_homolog_ic50.csv")
BATCH, VAL_FRAC, MAX_EPOCHS, PATIENCE = 128, 0.1, 40, 8
TRIPWIRE = 0.085
RANK_W = 0.5  # pairwise-ranking term weight (relative to z-space hinge units)


def load_homolog():
    """ChEMBL homolog IC50 (2C19/2C8/2B6/2A6) -> wide pIC50 frame, same
    filter pipeline as ft_ext's chembl load (rel '=', units nM, median per
    smiles)."""
    ch = pd.read_csv(HOM_CSV)
    ch = ch[ch["standard_relation"] == "="]
    ch["standard_value"] = pd.to_numeric(ch["standard_value"], errors="coerce")
    u = ch["standard_units"].astype(str).str.strip().str.lower()
    ch = ch[u.isin(["nm"])].dropna(subset=["standard_value", "canonical_smiles"])
    ch = ch[ch["standard_value"] > 0]
    ch["pic50"] = 9.0 - np.log10(ch["standard_value"])
    g = ch.groupby(["isoform", "canonical_smiles"])["pic50"].median().reset_index()
    wide = g.pivot(index="canonical_smiles", columns="isoform", values="pic50")
    df = pd.DataFrame({f"{i}_hom": wide[i] if i in wide else np.nan
                       for i in ["CYP2C19", "CYP2C8", "CYP2B6", "CYP2A6"]})
    df["SMILES"] = df.index
    return df.dropna(subset=HOM, how="all")


class ZooLoss(MSE):
    """Primary heads per MODE; aux MSE; bound cols zero. See module docstring."""

    def __init__(self, mu_p=None, sd_p=None, mu_a=None, sd_a=None, mode="hinge",
                 n_aux=None, lo_sl=None, hi_sl=None, **kw):
        super().__init__(**kw)
        self.n_aux = NAUX if n_aux is None else n_aux
        self.lo_sl = LO_SL if lo_sl is None else lo_sl
        self.hi_sl = HI_SL if hi_sl is None else hi_sl
        mu_p = np.zeros(NPRIM) if mu_p is None else np.asarray(mu_p, dtype=np.float32)
        sd_p = np.ones(NPRIM) if mu_p is None else np.asarray(sd_p, dtype=np.float32)
        mu_a = np.zeros(self.n_aux) if mu_a is None else np.asarray(mu_a, dtype=np.float32)
        sd_a = np.ones(self.n_aux) if sd_a is None else np.asarray(sd_a, dtype=np.float32)
        self.mode = mode
        self.register_buffer("mu_p", torch.as_tensor(mu_p))
        self.register_buffer("sd_p", torch.as_tensor(sd_p))
        self.register_buffer("mu_a", torch.as_tensor(mu_a))
        self.register_buffer("sd_a", torch.as_tensor(sd_a))

    def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask,
                             *args):
        prim = preds[:, :NPRIM]
        yprim = targets[:, :NPRIM]
        lo = targets[:, self.lo_sl]
        hi = targets[:, self.hi_sl]
        hb = mask[:, self.lo_sl] & mask[:, self.hi_sl]
        if self.training:  # FFN transform identity in train: preds are z
            z = prim
            raw = prim * self.sd_p + self.mu_p
            aux_z = preds[:, NPRIM:NPRIM + self.n_aux]
            aux_t = targets[:, NPRIM:NPRIM + self.n_aux]
        else:              # eval: output_transform applied, preds RAW
            raw = prim
            z = (prim - self.mu_p) / self.sd_p
            aux_raw = preds[:, NPRIM:NPRIM + self.n_aux]
            aux_z = (aux_raw - self.mu_a) / self.sd_a
            aux_t = targets[:, NPRIM:NPRIM + self.n_aux]
        m = mask[:, :NPRIM]
        if self.mode == "mae":
            prim_loss = torch.where(m, (z - yprim).abs(), torch.zeros_like(z))
        elif self.mode == "mse":
            prim_loss = torch.where(m, (z - yprim) ** 2, torch.zeros_like(z))
        else:  # hinge / rank: squared hinge outside [lo,hi]; unbounded rows -> MSE
            lo_s = torch.where(hb, lo, raw)
            hi_s = torch.where(hb, hi, raw)
            hinge = (torch.clamp(raw - hi_s, min=0.0) ** 2
                     + torch.clamp(lo_s - raw, min=0.0) ** 2) / (self.sd_p ** 2)
            hasb = hb.any(dim=1, keepdim=True).expand(-1, NPRIM) & m
            prim_loss = torch.where(hasb, hinge, torch.where(m, (z - yprim) ** 2,
                                                             torch.zeros_like(z)))
            if self.mode == "rank":
                # pairwise logistic on z preds over labeled pairs per isoform
                rl = torch.zeros_like(z)
                for j in range(NPRIM):
                    idx = torch.nonzero(mask[:, j]).squeeze(1)
                    if idx.numel() < 2:
                        continue
                    yv = yprim[idx, j]
                    pv = z[idx, j]
                    s = torch.sign(yv[:, None] - yv[None, :])
                    pdiff = pv[:, None] - pv[None, :]
                    sl = torch.nn.functional.softplus(-s * pdiff)
                    pm = (s != 0).float()
                    row_loss = (sl * pm).sum(dim=1) / pm.sum(dim=1).clamp(min=1)
                    rl[idx, j] = row_loss
                prim_loss = prim_loss + RANK_W * rl
        aux = (aux_z - aux_t) ** 2
        bnd = torch.zeros_like(preds[:, NPRIM + self.n_aux:])
        return torch.cat([prim_loss, aux, bnd], dim=1)


def set_cols(hom=False):
    """Module-level column layout; bound cols always last (8, interleaved)."""
    global COLS, NALL, LO_SL, HI_SL, NAUXD, LABCOLS
    aux = ft_ext.AUX + (HOM if hom else [])
    NAUXD = len(aux)
    COLS = ft_ext.PRIMARY + aux + BOUNDC
    NALL = len(COLS)
    LO_SL = slice(NPRIM + NAUXD, NPRIM + NAUXD + 8, 2)
    HI_SL = slice(NPRIM + NAUXD + 1, NPRIM + NAUXD + 8, 2)
    LABCOLS = ft_ext.PRIMARY + aux


def build_points(df):
    pts = []
    for _, row in df.iterrows():
        t = [float(row[c]) if pd.notna(row[c]) else float("nan") for c in COLS]
        pts.append(data.MoleculeDatapoint.from_smi(row["SMILES"], t))
    return pts


def train_fold(tr_df, va_df, seed, full_ft, mode, auxw, pt_init=False):
    featurizer = ft_ext.featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_dset = data.MoleculeDataset(build_points(tr_df), featurizer)
    from sklearn.preprocessing import StandardScaler
    Yraw = np.asarray(tr_dset.Y, dtype=np.float64)
    scaler = StandardScaler().fit(Yraw)
    for j in range(NPRIM + NAUXD, NALL):
        scaler.mean_[j] = 0.0
        scaler.scale_[j] = 1.0
        scaler.var_[j] = 1.0
    tr_dset.Y = scaler.transform(Yraw)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=0)
    va_dset = data.MoleculeDataset(build_points(va_df), featurizer)
    va_dset.Y = scaler.transform(np.asarray(va_dset.Y, dtype=np.float64))
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH, num_workers=0,
                                      shuffle=False)
    st = torch.load(ft_ext.WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    if pt_init:
        ptp = os.path.join(CACHE, "pt_encoder.pt")
        mp.load_state_dict(torch.load(ptp, map_location="cpu",
                                      weights_only=True))
        print(f"[pt] encoder init from {ptp}", flush=True)
    else:
        mp.load_state_dict(st["state_dict"])
    if not full_ft:
        for p in mp.parameters():
            p.requires_grad = False
    wts = np.concatenate([np.ones(NPRIM), np.full(NAUXD, auxw), np.zeros(8)])
    mu_p = scaler.mean_[:NPRIM].astype(np.float32)
    sd_p = scaler.scale_[:NPRIM].astype(np.float32)
    mu_a = scaler.mean_[NPRIM:NPRIM + NAUXD].astype(np.float32)
    sd_a = scaler.scale_[NPRIM:NPRIM + NAUXD].astype(np.float32)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=2048, n_layers=2,
                           n_tasks=NALL, dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
                           criterion=ZooLoss(mu_p=mu_p, sd_p=sd_p, mu_a=mu_a,
                                             sd_a=sd_a, mode=mode, n_aux=NAUXD,
                                             lo_sl=LO_SL, hi_sl=HI_SL,
                                             task_weights=wts))
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    seed_everything(seed)
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    return model


def main(mode, auxw, seed=0, full_ft=False, smoke=False, hom=False,
     pt=False):
    global MAX_EPOCHS, PATIENCE, BATCH
    set_cols(hom)
    wtag = "" if abs(auxw - 0.3) < 1e-9 else f"_w{auxw:g}"
    sfx = "" if seed == 0 else f"_s{seed}"
    base = (f"homarm{wtag}" if hom and mode == "hinge"
            else f"zoo_{mode}{wtag}")
    if pt:
        base = "pt_" + base.replace("zoo_", "")
    tag = base + sfx
    if smoke:
        MAX_EPOCHS, PATIENCE, BATCH = 2, 2, 64
        tag = f"{tag}_smoke"
        ft_ext.EXT_CAP = 1200
    ft, test, allrows = ft_ext.load_tables()
    if hom:
        hdf = load_homolog()
        hdf = hdf[~hdf["SMILES"].isin(set(ft["SMILES"]) | set(test["SMILES"]))]
        for c in COLS:
            if c not in hdf:
                hdf[c] = np.nan
        allrows = pd.concat([allrows, hdf.assign(_src="homolog")],
                            ignore_index=True).drop_duplicates("SMILES")
    bb = bounds_by_smi()
    for c in BOUNDC:
        allrows[c] = allrows["SMILES"].map(bb[c])
    ch_mask = (allrows["_src"] == "challenge").values
    lab = allrows[LABCOLS].notna().any(axis=1)
    oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    folds = ft["fold"].values
    yfull = {iso: ft[iso].values.astype(float) for iso in ISO}
    lo_full = {iso: bb[f"{iso}_bndlo"].reindex(ft["SMILES"]).values for iso in ISO}
    hi_full = {iso: bb[f"{iso}_bndhi"].reindex(ft["SMILES"]).values for iso in ISO}
    per_fold = {iso: [] for iso in ISO}
    t0 = time.time()
    nfold = 1 if smoke else 5
    for f in range(nfold):
        va_idx = np.where(folds == f)[0]
        va_df = allrows[ch_mask].iloc[va_idx]
        tr_pool = allrows[ch_mask].drop(index=va_idx).reset_index(drop=True)
        tr_extra = allrows[~ch_mask & lab]
        rng = np.random.default_rng(1000 * seed + f)
        if len(tr_extra) > ft_ext.EXT_CAP:
            keep = rng.choice(len(tr_extra), ft_ext.EXT_CAP, replace=False)
            tr_extra = tr_extra.iloc[keep]
        lab_tr = tr_pool[LABCOLS].notna().any(axis=1).values
        vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC),
                        replace=False)
        fit_df = pd.concat([tr_pool.drop(index=vi), tr_extra], ignore_index=True)
        ev_df = tr_pool.iloc[vi].reset_index(drop=True)
        fit_df = fit_df.reset_index(drop=True)
        model = train_fold(fit_df, ev_df, seed, full_ft, mode, auxw, pt)
        P = ft_ext.predict(model, va_df)
        for j, iso in enumerate(ISO):
            oof.loc[va_idx, iso] = P[:, j]
        for iso in ISO:
            k = ISO.index(iso)
            m = ~np.isnan(yfull[iso][va_idx])
            s = soft_rae(yfull[iso][va_idx][m], P[m, k],
                         lo_full[iso][va_idx][m], hi_full[iso][va_idx][m])
            per_fold[iso].append(float(s))
        del model
        torch.cuda.empty_cache()
        print(f"[{tag}] fold {f} done {time.time()-t0:.0f}s",
              {i: round(per_fold[i][-1], 3) for i in ISO}, flush=True)
    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, f"ft_oof_{tag}.csv"), index=False)
    if smoke:
        print("SMOKE: pipeline check only", flush=True)
        return
    fit_all = pd.concat([allrows[ch_mask], allrows[~ch_mask & lab]],
                        ignore_index=True)
    rng = np.random.default_rng(1000 * seed + 99)
    lab_all = fit_all[LABCOLS].notna().any(axis=1).values
    vi = rng.choice(np.where(lab_all)[0], size=int(lab_all.sum() * VAL_FRAC),
                    replace=False)
    model = train_fold(fit_all.drop(index=vi).reset_index(drop=True),
                       fit_all.iloc[vi].reset_index(drop=True), seed, full_ft,
                       mode, auxw, pt)
    Pte = ft_ext.predict(model, test)
    tsub = pd.DataFrame(Pte[:, :NPRIM], columns=ISO)
    tsub.insert(0, "SMILES", test["SMILES"])
    tsub.to_csv(os.path.join(CACHE, f"ft_test_preds_{tag}.csv"), index=False)
    del model
    torch.cuda.empty_cache()
    ev = {}
    for iso in ISO:
        m = ~np.isnan(yfull[iso])
        pooled = soft_rae(yfull[iso][m], oof[iso].values[m],
                          lo_full[iso][m], hi_full[iso][m])
        off = max(abs(x - pooled) for x in per_fold[iso])
        ev[iso] = {"per_fold_st_rae": [round(x, 4) for x in per_fold[iso]],
                   "pooled_st_rae": round(float(pooled), 4),
                   "max_fold_off_pooled": round(float(off), 4),
                   "quarantine": bool(off > TRIPWIRE)}
        print(f"[{tag}]", iso, json.dumps(ev[iso]), flush=True)
    json.dump(ev, open(os.path.join(CACHE, f"zoo_eval_{tag}.json"), "w"), indent=2)
    print(f"WROTE ft_oof_{tag}.csv + ft_test_preds_{tag}.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hinge",
                    choices=["hinge", "mae", "mse", "rank"])
    ap.add_argument("--auxw", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-ft", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hom", action="store_true",
                    help="add 4 ChEMBL homolog aux heads (2C19/2C8/2B6/2A6)")
    ap.add_argument("--pt", action="store_true",
                    help="init encoder from cache/pt_encoder.pt (transductive PT)")
    a = ap.parse_args()
    main(a.mode, a.auxw, a.seed, a.full_ft, a.smoke, a.hom, a.pt)
