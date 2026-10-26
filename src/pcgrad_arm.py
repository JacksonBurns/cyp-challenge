"""PCGrad gradient-surgery arm (Oct 25 directive: multi-objective balancing).

Motivation: mv/mvx multi-view arms give 2D6 ST-RAE 0.97-1.01 (starved) while
cls-weight 0.3 did NOT fix it (Sec 33). Hypothesis: the aux/cls task gradients
CONFLICT with the 2D6 primary gradient in the shared FFN trunk; scalar
reweighting cannot fix direction conflicts, only relative magnitudes. PCGrad
(Yu et al. 2020) projects each auxiliary task gradient onto the normal plane of
the primary gradient when they conflict (dot < 0), removing the conflicting
component while keeping the cooperative one.

Design:
- Same data/arch as loss_zoo --emx (PRIMARY x4 hinge + AUX(ChEMBL ext) +
  TDIC x4 + L2FC x4 + EMX x4 MSE + is_TDI x4 BCE heads).
- Task GROUPS for surgery: prim (4 hinge heads, anchor), aux (all MSE aux
  heads), cls (all is_TDI BCE heads). Surgery on the shared FFN trunk
  (+ encoder params if --fullft). Task heads receive ordinary gradients.
- Manual loop (chemprop's criterion receives no parameter access): replicate
  ZooLoss math from loss_zoo.ZooLoss._calc_unreduced_loss.
- Validation: per-epoch macro soft_rae on held-out challenge rows
  (VAL_FRAC subsample, same split logic as loss_zoo). Early stopping.

Gate (LEDGER2): 2D6 standalone ST-RAE improves vs mv control (0.97+) AND
macro not worse than mv (0.84) -> register ft_pcgrad family, hinge_blend
pool CV gain. 3 seeds only if gate passes.

Tag: pcgrad  -> cache/ft_oof_pcgrad{_sN}.csv, cache/ft_test_preds_pcgrad{_sN}.csv
Run (chemeleon env, ONE GPU job at a time):
  ~/miniforge3/envs/chemeleon/bin/python src/pcgrad_arm.py            # full
  ~/miniforge3/envs/chemeleon/bin/python src/pcgrad_arm.py --smoke    # 1 fold, 2 ep
  ~/miniforge3/envs/chemeleon/bin/python src/pcgrad_arm.py --nopcgrad # control (plain sum)
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from chemprop import data, models, nn
from lightning import seed_everything
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import ft_ext  # noqa: E402
import loss_zoo as L  # noqa: E402
from interval_arm import bounds_by_smi, soft_rae, ISO, NPRIM, BOUNDC  # noqa: E402

BATCH = 64
MAX_EPOCHS = 50
PATIENCE = 6
VAL_FRAC = 0.1
L2W = 1e-4


def make_model(fit_df, full_ft, device):
    """Build MPNN (chemprop-init) + FFN over L.COLS; returns model, scaler,
    mu/sd tensors, and shared-param list for surgery."""
    featurizer = ft_ext.featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_dset = data.MoleculeDataset(L.build_points(fit_df), featurizer)
    Yraw = np.asarray(tr_dset.Y, dtype=np.float64)
    scaler = StandardScaler().fit(Yraw)
    n_aux, n_cls = L.NAUXD, L.NCLSD
    for j in range(NPRIM + n_aux, L.NALL):
        scaler.mean_[j] = 0.0
        scaler.scale_[j] = 1.0
        scaler.var_[j] = 1.0
    tr_dset.Y = scaler.transform(Yraw)
    st = torch.load(ft_ext.WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    if not full_ft:
        for p in mp.parameters():
            p.requires_grad = False
    mu_p = torch.tensor(scaler.mean_[:NPRIM], dtype=torch.float32)
    sd_p = torch.tensor(scaler.scale_[:NPRIM], dtype=torch.float32)
    mu_a = torch.tensor(scaler.mean_[NPRIM:NPRIM + n_aux], dtype=torch.float32)
    sd_a = torch.tensor(scaler.scale_[NPRIM:NPRIM + n_aux], dtype=torch.float32)
    scaler_t = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=2048,
                           n_layers=2, n_tasks=L.NALL, dropout=0.1,
                           output_transform=scaler_t)
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    lo_sl, hi_sl, cls_sl = L.LO_SL, L.HI_SL, L.CLS_SL
    return (model.to(device), scaler, tr_dset, mu_p.to(device),
            sd_p.to(device), mu_a.to(device), sd_a.to(device),
            lo_sl, hi_sl, cls_sl, n_aux, n_cls)


def group_losses(z, targets, mask, mu_p, sd_p, mu_a, sd_a,
                 lo_sl, hi_sl, cls_sl, n_aux, n_cls,
                 auxw, clsw, primw=1.0):
    """Replicates ZooLoss pieces; returns (prim, aux, cls) scalar losses."""
    raw = z[:, :NPRIM] * sd_p + mu_p
    lo = targets[:, lo_sl]
    hi = targets[:, hi_sl]
    hb = mask[:, lo_sl] & mask[:, hi_sl]
    m = mask[:, :NPRIM]
    lo_s = torch.where(hb, lo, raw)
    hi_s = torch.where(hb, hi, raw)
    hinge = (torch.clamp(raw - hi_s, min=0.0) ** 2
             + torch.clamp(lo_s - raw, min=0.0) ** 2) / (sd_p ** 2)
    hasb = hb.any(dim=1, keepdim=True).expand(-1, NPRIM) & m
    yp = targets[:, :NPRIM]
    zprim = z[:, :NPRIM]
    prim = torch.where(hasb, hinge,
                       torch.where(m, (zprim - yp) ** 2, torch.zeros_like(zprim)))
    aux = (z[:, NPRIM:NPRIM + n_aux] - targets[:, NPRIM:NPRIM + n_aux]) ** 2
    aux = torch.where(mask[:, NPRIM:NPRIM + n_aux], aux, torch.zeros_like(aux))
    cls = torch.zeros(z.shape[0], n_cls, device=z.device)
    if n_cls:
        lg = z[:, cls_sl]
        ct = targets[:, cls_sl]
        cm = mask[:, cls_sl]
        ce = torch.nn.functional.binary_cross_entropy_with_logits(
            lg, ct, reduction="none")
        cls = torch.where(cm, ce, torch.zeros_like(ce))
    return (prim.mean() * primw, aux.mean() * auxw,
            cls.sum() / cm.sum().clamp(min=1) if n_cls else cls.sum() * 0)


def pcgrad_combine(losses, params, anchor=0):
    """PCGrad: project each non-anchor gradient onto normal plane of anchor.
    Call BEFORE the total-loss backward (graph intact). Returns list of
    combined grads (one per param) + diagnostics dict."""
    grads = []
    for i, l in enumerate(losses):
        g = torch.autograd.grad(l, params, retain_graph=True,
                                allow_unused=True)
        grads.append([torch.zeros_like(p) if gk is None else gk
                      for p, gk in zip(params, g)])
    flat = [torch.cat([gk.reshape(-1) for gk in g]) for g in grads]
    base = flat[anchor]
    bn2 = base.dot(base).clamp(min=1e-12)
    diags = []
    combined = base.clone()
    for i, g in enumerate(flat):
        if i == anchor:
            continue
        d = g.dot(base)
        diags.append(float(d / (g.norm() * base.norm() + 1e-12)))
        if d < 0:
            g = g - (d / bn2) * base
        combined = combined + g
    out, off = [], 0
    for p in params:
        n = p.numel()
        out.append(combined[off:off + n].view_as(p))
        off += n
    return out, {"cos_prim_aux": diags[0] if diags else float("nan"),
                 "cos_prim_cls": diags[1] if len(diags) > 1 else float("nan")}


@torch.no_grad()
def predict(model, df, n_all):
    featurizer = ft_ext.featurizers.SimpleMoleculeMolGraphFeaturizer()
    pts = [data.MoleculeDatapoint.from_smi(
        s, [float("nan")] * n_all) for s in df["SMILES"]]
    loader = data.build_dataloader(
        data.MoleculeDataset(pts, featurizer), batch_size=BATCH,
        num_workers=0, shuffle=False)
    model.eval()
    outs = []
    for batch in loader:
        bmg, V_d, X_d = batch[0], batch[1], batch[2]
        bmg.to(next(model.parameters()).device)
        y = model(bmg, V_d, X_d)
        outs.append(np.atleast_2d(y.detach().cpu().numpy()))
    return np.vstack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auxw", type=float, default=0.3)
    ap.add_argument("--clsw", type=float, default=1.0)
    ap.add_argument("--fullft", action="store_true")
    ap.add_argument("--nopcgrad", action="store_true",
                    help="control: plain sum, no surgery")
    a = ap.parse_args()
    tag = "pcgrad" + ("_ctl" if a.nopcgrad else "")
    tag += "" if a.seed == 0 else f"_s{a.seed}"
    if a.smoke:
        tag += "_smoke"
    L.set_cols(emx=True)
    n_aux, n_cls = L.NAUXD, L.NCLSD
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ft_ext.EXT_CAP = 1200 if a.smoke else ft_ext.EXT_CAP
    nfold = 1 if a.smoke else 5
    maxep = 2 if a.smoke else MAX_EPOCHS
    ft, test, allrows = ft_ext.load_tables()
    av = L.load_aux_views()
    av = av[~av["SMILES"].isin(set(test["SMILES"]))]
    keep = [c for c in av.columns if c in L.COLS or c == "SMILES"]
    allrows = allrows.merge(av[keep], on="SMILES", how="left")
    bb = bounds_by_smi()
    for c in BOUNDC:
        allrows[c] = allrows["SMILES"].map(bb[c])
    ch_mask = (allrows["_src"] == "challenge").values
    lab = allrows[L.LABCOLS].notna().any(axis=1)
    folds = ft["fold"].values
    yfull = {iso: ft[iso].values.astype(float) for iso in ISO}
    lo_full = {iso: bb[f"{iso}_bndlo"].reindex(ft["SMILES"]).values for iso in ISO}
    hi_full = {iso: bb[f"{iso}_bndhi"].reindex(ft["SMILES"]).values for iso in ISO}
    oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    oof_cls = np.full((len(ft), n_cls), np.nan)
    test_preds = []
    per_fold = {iso: [] for iso in ISO}
    seed_everything(a.seed)
    t0, hist = time.time(), []
    for f in range(nfold):
        va_idx = np.where(folds == f)[0]
        va_df = allrows[ch_mask].iloc[va_idx]
        tr_pool = allrows[ch_mask].drop(index=va_idx).reset_index(drop=True)
        tr_extra = allrows[~ch_mask & lab]
        rng = np.random.default_rng(1000 * a.seed + f)
        if len(tr_extra) > ft_ext.EXT_CAP:
            keep = rng.choice(len(tr_extra), ft_ext.EXT_CAP, replace=False)
            tr_extra = tr_extra.iloc[keep]
        lab_tr = tr_pool[L.LABCOLS].notna().any(axis=1).values
        vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC),
                        replace=False)
        fit_df = pd.concat([tr_pool.drop(index=vi), tr_extra],
                           ignore_index=True).reset_index(drop=True)
        ev_df = tr_pool.iloc[vi].reset_index(drop=True)
        (model, scaler, tr_dset, mu_p, sd_p, mu_a, sd_a,
         lo_sl, hi_sl, cls_sl, na, nc) = make_model(fit_df, a.fullft, device)
        loader = data.build_dataloader(tr_dset, batch_size=BATCH,
                                       num_workers=0)
        # shared params for surgery: FFN trunk (+ encoder if fullft)
        shared = list(model.predictor.ffn.parameters())
        if a.fullft:
            shared += [p for p in model.message_passing.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([
            {"params": model.predictor.ffn.parameters(), "weight_decay": L2W},
            {"params": model.predictor.output_transform.parameters(),
             "weight_decay": 0.0},
            {"params": model.message_passing.parameters(), "weight_decay": L2W},
        ], lr=1e-3)
        best, best_ep, bad = np.inf, 0, 0
        for ep in range(maxep):
            model.train()
            ep_diag = []
            for batch in loader:
                bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
                mask = targets.isfinite()
                targets = targets.nan_to_num(nan=0.0)
                bmg.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                Z = model.fingerprint(bmg, V_d, X_d)
                z = model.predictor.ffn(Z)  # z-space preds (train-mode identity)
                lp, la, lc = group_losses(
                    z, targets, mask, mu_p, sd_p, mu_a, sd_a,
                    lo_sl, hi_sl, cls_sl, na, nc, a.auxw, a.clsw)
                opt.zero_grad(set_to_none=True)
                if a.nopcgrad:
                    (lp + la + lc).backward()
                else:
                    comb, dg = pcgrad_combine([lp, la, lc], shared, anchor=0)
                    (lp + la + lc).backward()  # head grads (graph intact)
                    for p, gk in zip(shared, comb):  # override shared
                        p.grad = gk.clone()
                    ep_diag.append(dg)
                opt.step()
            # epoch validation on ev_df
            Pf = predict(model, ev_df, L.NALL)
            ev_iso = ev_df[ISO].values.astype(float)
            smacro = float(np.mean([soft_rae(
                ev_iso[:, k][~np.isnan(ev_iso[:, k])],
                Pf[~np.isnan(ev_iso[:, k]), k],
                bb[f"{ISO[k]}_bndlo"].reindex(ev_df["SMILES"]).values[
                    ~np.isnan(ev_iso[:, k])],
                bb[f"{ISO[k]}_bndhi"].reindex(ev_df["SMILES"]).values[
                    ~np.isnan(ev_iso[:, k])]) for k in range(4)]))
            hist.append(smacro)
            if smacro < best - 1e-4:
                best, best_ep, bad = smacro, ep, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
            diag = (ep_diag[-1] if ep_diag else {})
            print(f"[{tag}] fold {f} ep {ep} val_macro={smacro:.4f} "
                  f"best={best:.4f} cos_pa={diag.get('cos_prim_aux', float('nan')):.3f} "
                  f"cos_pc={diag.get('cos_prim_cls', float('nan')):.3f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
        model.load_state_dict(best_state)
        Pf = predict(model, va_df, L.NALL)
        # Pf columns are RAW: model(...) in eval applies output_transform
        if not a.smoke:
            Pte = predict(model, test, L.NALL)
            test_preds.append(Pte)
        for k, iso in enumerate(ISO):
            oof.loc[va_idx, iso] = Pf[:, k]
        if n_cls:
            oof_cls[va_idx] = 1.0 / (1.0 + np.exp(-Pf[:, NPRIM + na:
                                                       NPRIM + na + n_cls]))
        for k, iso in enumerate(ISO):
            m = ~np.isnan(yfull[iso][va_idx])
            s = soft_rae(yfull[iso][va_idx][m], Pf[m, k],
                         lo_full[iso][va_idx][m], hi_full[iso][va_idx][m])
            per_fold[iso].append(float(s))
        del model, opt, best_state
        torch.cuda.empty_cache()
        print(f"[{tag}] fold {f} DONE {time.time()-t0:.0f}s",
              {i: round(per_fold[i][-1], 4) for i in ISO}, flush=True)
    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, f"ft_oof_{tag}.csv"), index=False)
    print(f"[{tag}] POOLED soft_rae " +
          " ".join(f"{i}={np.mean(per_fold[i]):.4f}" for i in ISO), flush=True)
    if a.smoke:
        print("SMOKE: pipeline check only", flush=True)
        return
    # ensemble-average fold-model test predictions (seed tag per run;
    # loss_zoo seed-averaging downstream consumes ft_test_preds_<tag>)
    Pte = np.mean(np.stack(test_preds), axis=0)
    tp = pd.DataFrame(Pte[:, :NPRIM], columns=ISO)
    tp.insert(0, "SMILES", test["SMILES"])
    tp.to_csv(os.path.join(CACHE, f"ft_test_preds_{tag}.csv"), index=False)
    if n_cls:
        tp_cls = pd.DataFrame(
            1.0 / (1.0 + np.exp(-Pte[:, NPRIM + na:NPRIM + na + n_cls])),
            columns=L.COLS[NPRIM + na:NPRIM + na + n_cls])
        tp_cls.insert(0, "SMILES", test["SMILES"])
        tp_cls.to_csv(os.path.join(CACHE, f"tdi_probs_{tag}.csv"),
                      index=False)
    if n_cls:
        from sklearn.metrics import matthews_corrcoef as _mcc
        co = ft[["SMILES"]].copy()
        truth = allrows[ch_mask]
        clsh_names = L.CLSH4
        for j, c in enumerate(clsh_names):
            yv = truth[c].values.astype(float)
            co[c] = oof_cls[:, j]
            m = ~np.isnan(yv) & ~np.isnan(oof_cls[:, j])
            pred = (oof_cls[m, j] > 0.5).astype(int)
            sc = _mcc(yv[m].astype(int), pred)
            print(f"[{tag}] TDI head {c} OOF n={int(m.sum())} "
                  f"pos={int(yv[m].sum())} MCC={sc:.4f}", flush=True)
            co[f"{c}_truth"] = yv
        co.to_csv(os.path.join(CACHE, f"joint_oof_tdi_{tag}.csv"),
                  index=False)
    print(f"[{tag}] ALL DONE {time.time()-t0:.0f}s "
          f"hist_tail={[round(h, 4) for h in hist[-6:]]}", flush=True)


if __name__ == "__main__":
    main()
