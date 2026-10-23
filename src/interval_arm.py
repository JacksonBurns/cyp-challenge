"""L3: interval-hinge training arm (hinge = the metric, as the loss).

Plan lever L3 (docs/IMPROVEMENT_PLAN.md). Field evidence (open_top.md
Scigantic 2x2: interval+uniform 0.6895 macro vs mae+invvar 0.7187 =
+0.0275; interval HURTS at inverse-variance, interaction -0.0510;
Magicano: uniform weights optimal by sweep; do NOT convert censored
bounds to point labels - hinge them).

Recipe = ft_ext.py (union table: 4 challenge pIC50 primary heads + 4
ChEMBL + 4 AID1851 aux heads; external rows join training labeled on aux
only; frozen CheMeLeon MP encoder + 2048x2 FFN) with the scored-head
loss swapped:
  - primary heads: SQUARED hinge OUTSIDE the per-compound credible
    interval [lo, hi] in z-space; zero cost inside (that IS soft-RAE);
    rows without bounds (TDI-condition targets, external rows) fall back
    to plain MSE via mask (chemprop nan_to_num makes NaN bounds 0.0;
    the pre-nan_to_num mask carries the truth).
  - UNIFORM primary sample weights (ft_ext uses inverse-count; the arm
    uses flat 1.0 per the 2x2 evidence). Aux 0.3 as in ft_ext.
  - bounds ride in natively as 8 dummy target columns {iso}_bnd{lo,hi}
    (interleaved lo,hi per isoform, cols 8-15), task-weight 0.

Gate (plan): per-fold ST-RAE means + pooled; any fold >0.085 off pooled
= quarantine. 3 seeds FIRST to measure our seed-noise floor (field
floor: 0.05 Pearson on 1A2/2D6, 0.012-0.013 on 2C9/3A4). Promote to the
pool only above the floor; 2D6 needs 4/5 folds. Then re-run L1 on cp19.

Writes cache/ft_oof_intarm{_sN}.csv + cache/ft_test_preds_intarm{_sN}.csv
+ cache/intarm_eval_{tag}.json.

Run (chemeleon env, repo root; ONE GPU job at a time):
  ~/miniforge3/envs/chemeleon/bin/python src/interval_arm.py --seed 0
  ~/miniforge3/envs/chemeleon/bin/python src/interval_arm.py --seed 1
  ~/miniforge3/envs/chemeleon/bin/python src/interval_arm.py --seed 2
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
import torch.nn.functional as F
from chemprop import data, models, nn
from chemprop.nn.metrics import MSE
from lightning import pytorch as pl, seed_everything
from lightning.pytorch.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import ft_ext  # noqa: E402  (import-safe: no module-level side effects)


def soft_rae(y, yhat, lo, hi):
    """Same as src/run_regression.py:soft_rae (inlined - run_regression
    imports lightgbm, absent in the chemeleon env)."""
    err = np.clip(yhat - hi, 0, None) + np.clip(lo - yhat, 0, None)
    base = np.clip(np.mean(y) - hi, 0, None) + np.clip(lo - np.mean(y), 0, None)
    return err.sum() / base.sum()

ISO = ft_ext.ISO
NPRIM = 4
NAUX = 4
BOUNDC = [f"{i}_bnd{t}" for i in ISO for t in ("lo", "hi")]  # interleaved
COLS = ft_ext.COLS + BOUNDC  # 4 prim + 4 aux + 8 bounds = 16
NALL = len(COLS)
# BOUNDC interleaved: cols 8,10,12,14 = lo; 9,11,13,15 = hi
LO_SL = slice(NPRIM + NAUX, NPRIM + NAUX + 8, 2)
HI_SL = slice(NPRIM + NAUX + 1, NPRIM + NAUX + 8, 2)
BATCH, VAL_FRAC, MAX_EPOCHS, PATIENCE = 128, 0.1, 40, 8
AUX_W = 0.3
TRIPWIRE = 0.085


class IntervalHinge(MSE):
    """Primary cols 0-3: SQUARED hinge outside the RAW credible interval
    [lo,hi]. Bound cols 8-15 ride through normalize_targets RAW (train_fold
    forces their scaler rows to mean=0/scale=1). In TRAIN mode the FFN
    output_transform is identity, so preds are z-space: we un-z to raw with
    the primary label stats (mu_p, sd_p), hinge against raw bounds, divide
    by sd_p^2 to stay z-comparable with the aux MSE terms. Rows whose
    bounds are absent (mask False on a bound col) fall back to plain
    z-space MSE. Aux cols 4-7: MSE. Bound cols: zero. Targets arrive
    nan_to_num(0)-ed; `mask` (pre-fill) carries the truth."""

    def __init__(self, mu_p=None, sd_p=None, **kw):
        super().__init__(**kw)
        mu_p = np.zeros(NPRIM) if mu_p is None else np.asarray(mu_p, dtype=np.float32)
        sd_p = np.ones(NPRIM) if sd_p is None else np.asarray(sd_p, dtype=np.float32)
        self.register_buffer("mu_p", torch.as_tensor(mu_p))
        self.register_buffer("sd_p", torch.as_tensor(sd_p))

    def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask,
                             *args):
        prim = preds[:, :NPRIM]
        yprim = targets[:, :NPRIM]
        lo = targets[:, LO_SL]
        hi = targets[:, HI_SL]
        hb = mask[:, LO_SL] & mask[:, HI_SL]
        if self.training:  # FFN transform identity in train: preds are z
            z = prim
            raw = prim * self.sd_p + self.mu_p
        else:              # eval: output_transform applied, preds RAW
            raw = prim
            z = (prim - self.mu_p) / self.sd_p
        lo_s = torch.where(hb, lo, raw)
        hi_s = torch.where(hb, hi, raw)
        hinge = (torch.clamp(raw - hi_s, min=0.0) ** 2
                 + torch.clamp(lo_s - raw, min=0.0) ** 2) / (self.sd_p ** 2)
        m = mask[:, :NPRIM]
        hasb = hb.any(dim=1, keepdim=True).expand(-1, NPRIM) & m
        mse = (z - yprim) ** 2
        prim_loss = torch.where(hasb, hinge, mse)
        aux = (z - targets[:, NPRIM:NPRIM + NAUX]) ** 2
        bnd = torch.zeros_like(preds[:, NPRIM + NAUX:])
        return torch.cat([prim_loss, aux, bnd], dim=1)


def bounds_by_smi():
    DATA = os.path.join(HERE, "..", "data")
    inh = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv"))
    inh = inh.drop_duplicates("SMILES")
    d = {}
    for iso in ISO:
        d[f"{iso}_bndlo"] = inh.set_index("SMILES")[f"{iso}_pIC50_direct_inhibition_conf_low"]
        d[f"{iso}_bndhi"] = inh.set_index("SMILES")[f"{iso}_pIC50_direct_inhibition_conf_high"]
    bb = pd.DataFrame(d)  # index = SMILES
    return bb[~bb.index.duplicated(keep="first")]


def build_points_ext(df):
    pts = []
    for _, row in df.iterrows():
        t = [float(row[c]) if pd.notna(row[c]) else float("nan") for c in COLS]
        pts.append(data.MoleculeDatapoint.from_smi(row["SMILES"], t))
    return pts


def train_fold(tr_df, va_df, seed, full_ft):
    featurizer = ft_ext.featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_dset = data.MoleculeDataset(build_points_ext(tr_df), featurizer)
    # fit a scaler on RAW targets, pin bound cols to identity (raw bounds
    # pass through), apply manually: primary/aux cols z-scored, bounds raw
    from sklearn.preprocessing import StandardScaler
    Yraw = np.asarray(tr_dset.Y, dtype=np.float64)
    scaler = StandardScaler().fit(Yraw)
    for j in range(NPRIM + NAUX, NALL):
        scaler.mean_[j] = 0.0
        scaler.scale_[j] = 1.0
        scaler.var_[j] = 1.0
    tr_dset.Y = scaler.transform(Yraw)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=0)
    va_dset = data.MoleculeDataset(build_points_ext(va_df), featurizer)
    va_dset.Y = scaler.transform(np.asarray(va_dset.Y, dtype=np.float64))
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH, num_workers=0,
                                      shuffle=False)
    st = torch.load(ft_ext.WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    if not full_ft:
        for p in mp.parameters():
            p.requires_grad = False
    wts = np.concatenate([np.ones(NPRIM), np.full(NAUX, AUX_W), np.zeros(8)])
    mu_p = scaler.mean_[:NPRIM].astype(np.float32)
    sd_p = scaler.scale_[:NPRIM].astype(np.float32)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=2048, n_layers=2,
                           n_tasks=NALL, dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
                           criterion=IntervalHinge(mu_p=mu_p, sd_p=sd_p, task_weights=wts))
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    seed_everything(seed)
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    return model


def main(seed, full_ft=False, smoke=False):
    global MAX_EPOCHS, PATIENCE, BATCH
    sfx = "" if seed == 0 else f"_s{seed}"
    tag = f"intarm{sfx}"
    if smoke:
        MAX_EPOCHS, PATIENCE, BATCH = 2, 2, 64
        tag = "intarm_smoke"
        ft_ext.EXT_CAP = 1200
    ft, test, allrows = ft_ext.load_tables()
    bb = bounds_by_smi()
    for c in BOUNDC:
        allrows[c] = allrows["SMILES"].map(bb[c])
    ch_mask = (allrows["_src"] == "challenge").values
    lab = allrows[ft_ext.COLS].notna().any(axis=1)
    oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    folds = ft["fold"].values
    yfull = {iso: ft[iso].values.astype(float) for iso in ISO}
    lo_full = {iso: bb[f"{iso}_bndlo"].reindex(ft["SMILES"]).values
               for iso in ISO}
    hi_full = {iso: bb[f"{iso}_bndhi"].reindex(ft["SMILES"]).values
               for iso in ISO}
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
        lab_tr = tr_pool[ft_ext.COLS].notna().any(axis=1).values
        vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC),
                        replace=False)
        fit_df = pd.concat([tr_pool.drop(index=vi), tr_extra], ignore_index=True)
        ev_df = tr_pool.iloc[vi].reset_index(drop=True)
        fit_df = fit_df.reset_index(drop=True)
        model = train_fold(fit_df, ev_df, seed, full_ft)
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
        print(f"fold {f} done {time.time()-t0:.0f}s",
              {i: round(per_fold[i][-1], 3) for i in ISO}, flush=True)
    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, f"ft_oof_{tag}.csv"), index=False)
    if smoke:
        print("SMOKE: skipping test-preds model + eval json (pipeline check only)",
              flush=True)
        for iso in ISO:
            m = ~np.isnan(yfull[iso])
            pooled = soft_rae(yfull[iso][m], oof[iso].values[m],
                              lo_full[iso][m], hi_full[iso][m])
            print(iso, "fold0 st_rae", round(per_fold[iso][0], 4),
                  "sane:", bool(np.isfinite(per_fold[iso][0])), flush=True)
        return
    # test preds: full-data model (ft_ext pattern)
    fit_all = pd.concat([allrows[ch_mask], allrows[~ch_mask & lab]],
                        ignore_index=True)
    rng = np.random.default_rng(1000 * seed + 99)
    lab_all = fit_all[ft_ext.COLS].notna().any(axis=1).values
    vi = rng.choice(np.where(lab_all)[0], size=int(lab_all.sum() * VAL_FRAC),
                    replace=False)
    model = train_fold(fit_all.drop(index=vi).reset_index(drop=True),
                       fit_all.iloc[vi].reset_index(drop=True), seed, full_ft)
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
        print(iso, json.dumps(ev[iso]), flush=True)
    json.dump(ev, open(os.path.join(CACHE, f"intarm_eval_{tag}.json"), "w"),
              indent=2)
    print("WROTE", f"ft_oof_{tag}.csv + ft_test_preds_{tag}.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-ft", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, 2 epochs, tiny external cap - pipeline check only")
    a = ap.parse_args()
    main(a.seed, a.full_ft, a.smoke)
