"""CYP-UMTM: unified multi-task multi-view LightningModule (FINAL_PUSH_PLAN 1.2/1.4).

One chemprop MPNN, single FFN over shared MP state, 22 NaN-masked heads:
  direct x4 (MSE, inverse-count mean-normalized to 1.0) / tdic x4 (MSE grp .5) /
  is_TDI x2 (BCE grp .5, stop-grad STACK over Z + detached potency-head outputs) /
  log2fc x4, chembl x4, pubchem x4 (MSE grp .3). Weights per umtm_data.
  --weights {plan,ext,primary}:
    plan    = full spec (production)
    ext     = ft_ext reproduction: direct + chembl/pubchem .3, new heads weight 0,
              rows = challenge + capped ext (no sc_only) -> gate vs ft_oof_ext.csv
    primary = direct only, challenge rows only -> gate vs ft_oof_fullft.csv

Training regime copied from ft_ext (proven lineage): targets z-normed by FIT-set
stats, RegressionFFN hidden 2048 x2 dropout .1, batch 128, MPNN default LR
(2-ep warmup 1e-4->1e-3->1e-4), full fine-tune, EarlyStopping val_loss p8, max
40 ep, same fold/val/ext-subsample rng streams as ft_ext.main().

Unit convention (decisive, probed on chemprop 2.3.1): UnscaleTransform is a
NO-OP in train mode, so chemprop's own loss is z-space; in eval mode predictor
emits original units (ft_ext silently validates against a unit-mismatched
val_loss). We bypass predictor.forward: raw = predictor.ffn(Z) is z-space in
BOTH modes, losses always on z-space (identical to ft_ext training loss), and
exports unscale explicitly via scaler buffers. Binary heads keep mean 0/std 1
(identity scaler cols) so raw[:,BIN] ARE logits in every mode; UnscaleTransform
gets identity mean/scale there too (defense in depth).

Lineages (bare-encoder ckpt schema = ft_ext --pretrained):
  L1 ~/chemeleon_nazarov/chemeleon_mp.pt   L2 cache/admecd_ckpt_chm.pt
  L3 cache/admecd_ckpt_med.pt              L4 from-scratch D-MPNN d_h 300
Folds: --fold reg (seed 0, legacy ft_data; REQUIRED for gate reproduction) or
tdi (seed 7, production). Writes cache/umtm_oof_<tag>.csv (6145 challenge rows x
22 heads) + cache/umtm_test_<tag>.csv (750 rows). tag = lineage[+mode][+_s<seed>].
Run (chemeleon env, ONE job at a time on the single shared GPU):
  ~/miniforge3/envs/chemeleon/bin/python src/umtm_module.py --lineage L2 --seed 0
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as tnn
import torch.nn.functional as F
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from umtm_data import CHM, DIRECT, HEADS, ISO, PUB, SC, TDIB, TDIC, per_task_weights  # noqa: E402

ROOT = os.path.join(HERE, "..")
CACHE = os.path.join(ROOT, "cache")
CHM_W = os.path.expanduser("~/chemeleon_nazarov/chemeleon_mp.pt")
LINEAGE_CKPT = {"L1": CHM_W, "L2": os.path.join(CACHE, "admecd_ckpt_chm.pt"),
                "L3": os.path.join(CACHE, "admecd_ckpt_med.pt"), "L4": None}
MAX_EPOCHS, PATIENCE, BATCH, VAL_FRAC, EXT_CAP = 40, 8, 128, 0.1, 9000
D_H_L4, FFN_HID = 300, 2048

BIN_IDX = [HEADS.index(h) for h in TDIB]
DIR_IDX = [HEADS.index(h) for h in DIRECT]
TDIC_IDX = [HEADS.index(h) for h in TDIC]
W_BIN = 0.25  # 0.5 group weight / 2 heads


def gate_weights(mode, counts):
    """per-task MSE weights in HEADS order for gate modes (plan uses
    umtm_data.per_task_weights). ext == ft_ext recipe: direct inverse-count +
    flat 0.3/head on chembl/pubchem, everything else zero."""
    if mode == "plan":
        raise ValueError("plan mode must use per_task_weights")
    w = np.zeros(len(HEADS))
    prim = 1.0 / np.maximum(counts[:4], 1)
    w[:4] = prim / prim.mean()
    if mode == "ext":
        w[[HEADS.index(h) for h in CHM + PUB]] = 0.3
    return w


def head_weights(mode, fit_df):
    """returns (w_mse 22-float ndarray, ) with direct inverse-count recipe; plan
    mode uses umtm_data.per_task_weights for the authoritative spec numbers."""
    counts = fit_df[HEADS].notna().sum().values.astype(float)
    if mode == "plan":
        w = per_task_weights(fit_df)
    else:
        w = gate_weights(mode, counts)
    w = np.asarray(w, dtype=float)
    w[BIN_IDX] = 0.0  # bin heads excluded from the MSE metric, handled by BCE
    return w


def make_scaler(fit_df):
    """z-stats over FIT rows; binary head cols forced to identity (logits)."""
    Y = fit_df[HEADS].to_numpy(float)
    mean = np.nanmean(Y, axis=0)
    std = np.nanstd(Y, axis=0)
    std = np.where(std > 0, std, 1.0)
    mean[BIN_IDX], std[BIN_IDX] = 0.0, 1.0
    sc = StandardScaler()
    sc.mean_, sc.scale_, sc.var_, sc.n_features_in_ = mean, std, std ** 2, len(HEADS)
    return sc


def build_points(df):
    return [data.MoleculeDatapoint.from_smi(
        r["SMILES"], [float(r[c]) if pd.notna(r[c]) else float("nan") for c in HEADS])
        for _, r in df.iterrows()]


class UMTM(models.MPNN):
    def __init__(self, mp, agg, ffn, stacked=True):
        super().__init__(mp, agg, ffn, batch_norm=False)
        self.stacked = stacked
        self._tr_sq = self._tr_m = 0
        self._va_err = self._va_n = self._va_bce = self._va_bn = 0
        if stacked:
            d_in = mp.output_dim + len(DIR_IDX) + len(TDIC_IDX)
            self.stack = tnn.Sequential(tnn.Linear(d_in, 256), tnn.ReLU(),
                                        tnn.Dropout(0.1), tnn.Linear(256, len(BIN_IDX)))

    def outputs(self, bmg, V_d, X_d):
        """mode-invariant: raw z-space ffn outputs + bin logits (z==orig there)."""
        Z = self.fingerprint(bmg, V_d, X_d)
        raw = self.predictor.ffn(Z)  # bypass output_transform (train-mode no-op anyway)
        if self.stacked:
            cat = torch.cat([Z, raw[:, DIR_IDX].detach(), raw[:, TDIC_IDX].detach()], dim=1)
            bin_logits = self.stack(cat)
        else:
            bin_logits = raw[:, BIN_IDX]
        return raw, bin_logits

    def _loss_parts(self, raw, bin_logits, targets, w_bin):
        mask = targets.isfinite()
        t = targets.nan_to_num(nan=0.0)
        wm = self._wm_t.to(t.device)
        sq = ((raw - t) ** 2) * mask * wm
        # chemprop MSE normalization (probed): total_loss += (sq*w*mask).sum(),
        # num_samples += mask.sum() -> divide by ACTIVE CELL count over the
        # active (nonzero-weight) columns, so ext mode == ft_ext loss exactly.
        active = mask[:, self._active_cols].sum().clamp(min=1)
        l_mse = sq.sum() / active
        mb = mask[:, BIN_IDX]
        lb = F.binary_cross_entropy_with_logits(bin_logits, t[:, BIN_IDX], reduction="none")
        # gate modes (ext/primary) pass w_bin=0: bin heads fully inactive so
        # the loss == ft_ext recipe exactly (stack exists but gets no gradient).
        l_bce = (lb * mb).sum() * w_bin / mb.sum().clamp(min=1)
        return sq, mb, l_mse, l_bce

    def training_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, weights, lt, gt = batch
        raw, bl = self.outputs(bmg, V_d, X_d)
        sq, mb, l_mse, l_bce = self._loss_parts(raw, bl, targets, self._w_bin)
        self._tr_sq += sq.sum().item()
        self._tr_m += int(targets.shape[0])
        l = l_mse + l_bce
        self.log("train_loss", l, on_epoch=True, prog_bar=True, batch_size=targets.shape[0])
        return l

    def validation_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, weights, lt, gt = batch
        raw, bl = self.outputs(bmg, V_d, X_d)
        mask = targets.isfinite()
        t = targets.nan_to_num(nan=0.0)
        # ES loss == the training loss formula restricted to ev (challenge) rows:
        # weighted z-MSE over active columns / active cells + w_bin * mean masked
        # BCE. In ext/primary gate modes (w_bin=0, only primary aux columns
        # active, and ev rows carry no aux labels) this is byte-equivalent to
        # ft_ext's val_loss, so EarlyStopping stops at the same place. External
        # rows never validate (ev_df is always challenge-only).
        wm = self._wm_t.to(t.device)
        sq = ((raw - t) ** 2) * mask * wm
        active = mask[:, self._active_cols].sum().clamp(min=1)
        self._va_err = getattr(self, "_va_err", 0.0) + sq.sum().item()
        self._va_n = getattr(self, "_va_n", 0) + int(active.item())
        mb = mask[:, BIN_IDX]
        lb = F.binary_cross_entropy_with_logits(bl, t[:, BIN_IDX], reduction="none")
        self._va_bce = getattr(self, "_va_bce", 0.0) + float((lb * mb).sum().item())
        self._va_bn = getattr(self, "_va_bn", 0) + int(mb.sum().item())
        # per-head z-space MSE accumulation for --report curves
        for j in range(len(HEADS)):
            m = mask[:, j]
            if m.sum() == 0:
                continue
            e = ((raw[:, j] - t[:, j]) ** 2)[m]
            cur = getattr(self, f"_vh_{j}", None)
            setattr(self, f"_vh_{j}", (cur[0] + e.sum().item(), cur[1] + int(m.sum())) if cur
                    else (e.sum().item(), int(m.sum())))

    def on_train_epoch_start(self):
        self._tr_sq = 0
        self._tr_m = 0

    def on_validation_epoch_start(self):
        self._va_err = self._va_n = self._va_bce = self._va_bn = 0
        for j in range(len(HEADS)):
            if hasattr(self, f"_vh_{j}"):
                delattr(self, f"_vh_{j}")

    def on_validation_epoch_end(self):
        n = max(self._va_n, 1)
        val = self._va_err / n + self._w_bin * self._va_bce / max(self._va_bn, 1)
        self.log("val_loss", val, prog_bar=True)
        parts = [f"ep{self.current_epoch + 1} train_mse={self._tr_sq / max(self._tr_m, 1):.4f}"
                 f" val={val:.4f}"]
        for j, h in enumerate(HEADS):
            if hasattr(self, f"_vh_{j}"):
                s, m = getattr(self, f"_vh_{j}")
                parts.append(f"{h}={s / m:.4f}")
        print(" ".join(parts), flush=True)

    def predict_frame(self, smiles_df):
        feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
        pts = [data.MoleculeDatapoint.from_smi(s, [float("nan")] * len(HEADS))
               for s in smiles_df["SMILES"]]
        dset = data.MoleculeDataset(pts, feat)
        loader = data.build_dataloader(dset, batch_size=BATCH, num_workers=0, shuffle=False)
        self.eval()
        outs = []
        with torch.no_grad():
            for batch in loader:
                bmg, V_d, X_d = batch[0], batch[1], batch[2]
                raw, bl = self.outputs(bmg, V_d, X_d)
                y = raw * self._sc_scale + self._sc_mean  # explicit unscale
                y[:, BIN_IDX] = torch.sigmoid(bl)
                outs.append(y.detach().cpu().numpy())
        P = np.vstack(outs)
        # alignment guarantee: one row per input SMILES (MoleculeDataset keeps
        # unfeaturizable rows as zero graphs, but assert or a silent drop would
        # corrupt OOF row alignment)
        assert P.shape[0] == len(smiles_df), f"predict_frame rows {P.shape[0]} != {len(smiles_df)}"
        return P


def build_model(lineage, scaler, w_mse, stacked, w_bin=W_BIN):
    if lineage == "L4":
        feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
        mp = nn.BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=D_H_L4,
                                   bias=True, depth=4, dropout=0.0, undirected=True)
    else:
        st = torch.load(LINEAGE_CKPT[lineage], map_location="cpu", weights_only=False)
        mp = nn.BondMessagePassing(**st["hyper_parameters"])
        mp.load_state_dict(st["state_dict"])
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=FFN_HID, n_layers=2,
                           n_tasks=len(HEADS), dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    model = UMTM(mp, nn.MeanAggregation(), ffn, stacked=stacked)
    model._w_mse = w_mse
    model._w_bin = float(w_bin)
    model.register_buffer("_wm_t", torch.tensor(w_mse, dtype=torch.float))
    model.register_buffer("_active_cols", torch.tensor(w_mse > 0))
    model.register_buffer("_sc_mean", torch.tensor(scaler.mean_, dtype=torch.float).unsqueeze(0))
    model.register_buffer("_sc_scale", torch.tensor(scaler.scale_, dtype=torch.float).unsqueeze(0))
    return model


def train_model(fit_df, ev_df, lineage, tag, seed, weights, stacked, max_epochs=MAX_EPOCHS):
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    scaler = make_scaler(fit_df)
    tr_dset = data.MoleculeDataset(build_points(fit_df), feat)
    tr_dset.normalize_targets(scaler)
    ev_dset = data.MoleculeDataset(build_points(ev_df), feat)
    ev_dset.normalize_targets(scaler)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=0)
    ev_loader = data.build_dataloader(ev_dset, batch_size=BATCH, num_workers=0, shuffle=False)
    # gate modes (ext/primary): bin heads fully inactive - no BCE gradient and
    # no stack module (closest possible shape match to the legacy recipe).
    w_bin_eff = W_BIN if weights == "plan" else 0.0
    stacked_eff = bool(stacked and weights == "plan")
    model = build_model(lineage, scaler, head_weights(weights, fit_df), stacked_eff, w_bin_eff)
    print(f"[{tag}] trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M",
          flush=True)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    seed_everything(seed)
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=ev_loader)
    return model


def fold_inputs(master, weights, fold_col, seed, f):
    """one fold's (fit_df, ev_df, va_idx) with ft_ext rng semantics (1000*seed+f)."""
    ch = master[master.kind == "challenge"].reset_index(drop=True)
    va_idx = np.where(ch[fold_col].values == f)[0]
    tr_pool = ch.drop(index=va_idx).reset_index(drop=True)
    rng = np.random.default_rng(1000 * seed + f)
    ext = master[master.kind == "ext"]
    sc_only = master[master.kind == "sc_only"]
    if weights == "primary":
        ext, sc_only = ext.iloc[0:0], sc_only.iloc[0:0]  # challenge rows only
    elif weights == "ext":
        sc_only = sc_only.iloc[0:0]  # ft_ext pool: challenge + chembl/pubchem only
    if len(ext) > EXT_CAP:
        ext = ext.iloc[rng.choice(len(ext), EXT_CAP, replace=False)]
    # ft_ext semantics: rng draws ext subsample BEFORE the val split (same stream)
    lab_tr = tr_pool[HEADS].notna().any(axis=1).values
    vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC), replace=False)
    fit_df = pd.concat([tr_pool.drop(index=vi), ext, sc_only], ignore_index=True)
    return fit_df, tr_pool.iloc[vi], va_idx


def run(lineage, seed, weights, fold, smoke=False, stacked=True, no_test=False):
    t0 = time.time()
    s = "" if seed == 0 else f"_s{seed}"
    tag = f"{lineage}{'' if weights == 'plan' else '_' + weights}{s}" + ("_smoke" if smoke else "")
    master = pd.read_csv(os.path.join(CACHE, "umtm_master.csv"))
    ch = master[master.kind == "challenge"].reset_index(drop=True)
    fold_col = "fold_" + fold
    print(f"[{tag}] rows: challenge {len(ch)}; fold col {fold_col}; weights={weights}", flush=True)

    oof = pd.DataFrame(np.nan, index=range(len(ch)), columns=HEADS)
    for f in ([0] if smoke else range(5)):
        fit_df, ev_df, va_idx = fold_inputs(master, weights, fold_col, seed, f)
        model = train_model(fit_df, ev_df, lineage, f"{tag} fold{f}", seed, weights,
                            stacked, max_epochs=2 if smoke else MAX_EPOCHS)
        # ft_ext semantics: ev_df is only the EarlyStopping split; the OOF rows
        # are predictions on the fold's own rows (va_idx), predicted after training.
        oof.iloc[va_idx, :] = model.predict_frame(ch.iloc[va_idx][["SMILES"]])
        del model
        torch.cuda.empty_cache()
        print(f"[{tag}] fold {f} done {time.time()-t0:.0f}s", flush=True)
        if smoke:
            print("SMOKE OK (fold 0 only; no files written)", flush=True)
            return
    oof.insert(0, "SMILES", ch.SMILES.values)
    oof.insert(1, "fold", ch[fold_col].values)
    # both fold vectors ride along: production runs use fold_reg, but the TDI
    # nested-threshold eval must nest on seed-7 folds to match legacy protocol.
    if "fold_tdi" in ch.columns and fold_col != "fold_tdi":
        oof.insert(2, "fold_tdi", ch["fold_tdi"].values)
    oof.to_csv(os.path.join(CACHE, f"umtm_oof_{tag}.csv"), index=False)
    if no_test:  # gate runs: OOF only, skip the all-data model (ft_ext --skip-folds mirror)
        print("DONE-OOF-ONLY", tag, f"{time.time() - t0:.0f}s", flush=True)
        return

    # all-data model for test predictions (ft_ext rng semantics: default_rng(99+seed))
    rng = np.random.default_rng(99 + seed)
    lab = ch[HEADS].notna().any(axis=1).values
    ext = master[master.kind == "ext"]
    sc_only = master[master.kind == "sc_only"]
    if weights == "primary":
        ext, sc_only = ext.iloc[0:0], sc_only.iloc[0:0]
    elif weights == "ext":
        sc_only = sc_only.iloc[0:0]
    if len(ext) > EXT_CAP:
        ext = ext.iloc[rng.choice(len(ext), EXT_CAP, replace=False)]
    vi = rng.choice(np.where(lab)[0], size=int(lab.sum() * VAL_FRAC), replace=False)
    fit_df = pd.concat([ch.drop(index=vi), ext, sc_only], ignore_index=True)
    model = train_model(fit_df, ch.iloc[vi], lineage, f"{tag} all", seed, weights, stacked)
    test = pd.read_csv(os.path.join(CACHE, "umtm_test_smiles.csv"))
    P = model.predict_frame(test)
    tdf = pd.DataFrame(P, columns=HEADS)
    tdf.insert(0, "SMILES", test.SMILES.values)
    tdf.to_csv(os.path.join(CACHE, f"umtm_test_{tag}.csv"), index=False)
    print("DONE", tag, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", required=True, choices=["L1", "L2", "L3", "L4"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="plan", choices=["plan", "ext", "primary"])
    ap.add_argument("--fold", default="reg", choices=["reg", "tdi"])
    ap.add_argument("--smoke", action="store_true", help="fold 0 only, 2 epochs, no outputs")
    ap.add_argument("--no-test", action="store_true", help="OOF only, skip all-data model + test preds")
    ap.add_argument("--no-stack", action="store_true", help="bin heads plain from FFN (ablation)")
    a = ap.parse_args()
    run(a.lineage, a.seed, a.weights, a.fold, smoke=a.smoke, stacked=not a.no_stack,
        no_test=a.no_test)
