"""UQ-weighted loss arm (Oct 25 directive, follow-on to uq_detect gate PASS).

uq_detect.py: cross-family OOF disagreement flags high-residual molecules
(AUC .62-.66, top-decile lift 1.32-1.52x). This arm tests the CAUSAL use:
downweight flagged training rows in the prim (per-iso) loss.

Design (pre-registered):
 - row weight w[i] = W_DW if per-molecule mean UQ >= q-quantile, else 1.0;
   rides chemprop MoleculeDatapoint.weight into the batch `weights` tensor.
 - prim-only loss identical to pcgrad_arm (hinge w/ bounds, MSE w/o),
   normalized weighting: (cell*w).sum() / (mask*w).sum().
 - frozen encoder (chemprop PT init), same folds/val-split/early-stop.
 - GATE: macro val soft_rae gain vs pooled baseline seeds > 0.005.

Run: baseline first (--baseline), then weighted; compare MACRO lines.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from chemprop import data, models, nn
from lightning import seed_everything
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loss_zoo as L  # noqa: E402
import ft_ext  # noqa: E402
from interval_arm import bounds_by_smi, soft_rae, ISO, NPRIM, BOUNDC  # noqa: E402

CACHE = L.CACHE
BATCH = 64
MAX_EPOCHS = 50
L2W = 1e-5
VAL_FRAC = 0.1


def build_points_w(df, wvec):
    pts = []
    for i, (_, row) in enumerate(df.iterrows()):
        t = [float(row[c]) if pd.notna(row[c]) else float("nan")
             for c in L.COLS]
        pts.append(data.MoleculeDatapoint.from_smi(
            row["SMILES"], t, weight=float(wvec[i])))
    return pts


def make_model(fit_df, wvec, full_ft, device):
    featurizer = ft_ext.featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_dset = data.MoleculeDataset(build_points_w(fit_df, wvec), featurizer)
    Yraw = np.asarray(tr_dset.Y, dtype=np.float64)
    scaler = StandardScaler().fit(Yraw)
    for j in range(NPRIM, L.NALL):
        scaler.mean_[j] = 0.0
        scaler.scale_[j] = 1.0
        scaler.var_[j] = 1.0
    tr_dset.Y = scaler.transform(Yraw)
    st = torch.load(ft_ext.WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    for p in mp.parameters():
        p.requires_grad = bool(full_ft)
    mu_p = torch.tensor(scaler.mean_[:NPRIM], dtype=torch.float32)
    sd_p = torch.tensor(scaler.scale_[:NPRIM], dtype=torch.float32)
    scaler_t = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=2048,
                           n_layers=2, n_tasks=L.NALL, dropout=0.1,
                           output_transform=scaler_t)
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    return (model.to(device), scaler, tr_dset, mu_p.to(device),
            sd_p.to(device))


def prim_loss(z, targets, mask, w, mu_p, sd_p):
    """prim cells (B, NPRIM); w: (B,) row weights; targets standardized."""
    raw = z[:, :NPRIM] * sd_p + mu_p
    m = mask[:, :L.NALL][:, :NPRIM]
    lo = targets[:, L.LO_SL]
    hi = targets[:, L.HI_SL]
    hb = mask[:, L.LO_SL] & mask[:, L.HI_SL]
    lo_s = torch.where(hb, lo, raw)
    hi_s = torch.where(hb, hi, raw)
    hinge = (torch.clamp(raw - hi_s, min=0.0) ** 2
             + torch.clamp(lo_s - raw, min=0.0) ** 2) / (sd_p ** 2)
    mse = (z[:, :NPRIM] - targets[:, :NPRIM]) ** 2
    cell = torch.where(hb, hinge, mse)
    cell = torch.where(m, cell * w[:, None], torch.zeros_like(cell))
    return cell.sum() / m.sum().clamp(min=1)


def predict(model, df):
    featurizer = ft_ext.featurizers.SimpleMoleculeMolGraphFeaturizer()
    pts = [data.MoleculeDatapoint.from_smi(
        s, [float("nan")] * L.NALL) for s in df["SMILES"]]
    loader = data.build_dataloader(
        data.MoleculeDataset(pts, featurizer), batch_size=BATCH,
        num_workers=0, shuffle=False)
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d = batch[0], batch[1], batch[2]
            bmg.to(next(model.parameters()).device)
            y = model(bmg, V_d, X_d)
            outs.append(np.atleast_2d(y.detach().cpu().numpy()))
    return np.vstack(outs)[:, :NPRIM]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--q", type=float, default=0.9)
    ap.add_argument("--wdw", type=float, default=0.5)
    ap.add_argument("--iso", default="all",
                    help="'all' = mean UQ across iso; iso name = that col")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--fullft", action="store_true")
    a = ap.parse_args()
    tag = "uqw_ctl" if a.baseline else f"uqw_q{int(a.q*100)}_w{int(a.wdw*100)}_{a.iso}"
    if a.seed:
        tag += f"_s{a.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft, test, allrows = ft_ext.load_tables()
    bb = bounds_by_smi()
    for c in BOUNDC:
        allrows[c] = allrows["SMILES"].map(bb[c])
    ch_mask = (allrows["_src"] == "challenge").values
    folds = ft["fold"].values
    uq = None
    if not a.baseline:
        uq = pd.read_csv(os.path.join(CACHE, "uq_cross_spread.csv"),
                         index_col=0)
        col = (uq[[f"{i}_uq" for i in ISO]].mean(axis=1)
               if a.iso == "all" else uq[f"{a.iso}_uq"])
        thr = col.quantile(a.q)
        print(f"UQ down rows: {int((col >= thr).sum())} at q={a.q}",
              flush=True)

    def roww_for(df):
        if a.baseline:
            return np.ones(len(df), dtype=np.float32)
        s = col.reindex(pd.Index(df["SMILES"])).values
        return np.where(np.nan_to_num(s, nan=-np.inf) >= thr,
                        a.wdw, 1.0).astype(np.float32)

    seed_everything(a.seed)
    res, t0 = [], time.time()
    for f in range(5):
        va_idx = np.where(folds == f)[0]
        va_df = allrows[ch_mask].iloc[va_idx]
        tr_pool = allrows[ch_mask].drop(index=va_idx).reset_index(drop=True)
        rng = np.random.default_rng(1000 * a.seed + f)
        lab_tr = tr_pool[L.LABCOLS].notna().any(axis=1).values
        vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC),
                        replace=False)
        fit_df = tr_pool.drop(index=vi).reset_index(drop=True)
        ev_df = tr_pool.iloc[vi].reset_index(drop=True)
        wvec = roww_for(fit_df)
        (model, scaler, tr_dset, mu_p, sd_p) = make_model(
            fit_df, wvec, a.fullft, device)
        loader = data.build_dataloader(tr_dset, batch_size=BATCH,
                                       num_workers=0)
        opt = torch.optim.AdamW(
            [{"params": model.predictor.ffn.parameters(), "weight_decay": L2W},
             {"params": model.predictor.output_transform.parameters(),
              "weight_decay": 0.0},
             {"params": [p for p in model.message_passing.parameters()
                         if p.requires_grad], "weight_decay": L2W}], lr=1e-3)
        best, best_ep, bad = np.inf, 0, 0
        best_state = None
        for ep in range(MAX_EPOCHS):
            model.train()
            for batch in loader:
                bmg, V_d, X_d, targets, weights, lt_m, gt_m = batch
                mask = targets.isfinite()
                targets = targets.nan_to_num(nan=0.0)
                bmg.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                w = weights.to(device).clamp(min=1e-3)
                z = model.predictor.ffn(model.fingerprint(bmg, V_d, X_d))
                loss = prim_loss(z, targets, mask, w, mu_p, sd_p)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            Pf = predict(model, ev_df)
            yv = ev_df[L.LABCOLS].values.astype(float)
            lo_e = ev_df[[f"{i}_bndlo" for i in ISO]].values
            hi_e = ev_df[[f"{i}_bndhi" for i in ISO]].values
            scores = []
            for k in range(NPRIM):
                m = np.isfinite(yv[:, k])
                if m.sum() < 8:
                    continue
                scores.append(float(np.mean(
                    soft_rae(yv[m, k], Pf[m, k], lo_e[m, k], hi_e[m, k]))))
            smacro = float(np.mean(scores))
            if smacro < best - 1e-4:
                best, best_ep, bad = smacro, ep, 0
                best_state = {kk: vv.cpu().clone()
                              for kk, vv in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break
        model.load_state_dict(best_state)
        Pf = predict(model, va_df)
        yva = va_df[L.LABCOLS].values.astype(float)
        lo_v = va_df[[f"{i}_bndlo" for i in ISO]].values
        hi_v = va_df[[f"{i}_bndhi" for i in ISO]].values
        out = {}
        for k, iso in enumerate(ISO):
            m = np.isfinite(yva[:, k])
            out[iso] = float(np.mean(
                soft_rae(yva[m, k], Pf[m, k], lo_v[m, k], hi_v[m, k])))
        res.append(out)
        print(f"[{tag}] f{f} best_ep={best_ep} " +
              " ".join(f"{i}={out[i]:.4f}" for i in ISO), flush=True)
    macro = {iso: float(np.mean([r[iso] for r in res])) for iso in ISO}
    print(f"[{tag}] MACRO " +
          " ".join(f"{i}={macro[i]:.4f}" for i in ISO) +
          f" | macroST {np.mean(list(macro.values())):.4f} "
          f"t={time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(CACHE, f"uqw_{tag}.json"), "w") as fh:
        json.dump({"folds": res, "macro": macro, "q": a.q, "wdw": a.wdw,
                   "iso": a.iso, "baseline": a.baseline, "seed": a.seed}, fh)


if __name__ == "__main__":
    main()
