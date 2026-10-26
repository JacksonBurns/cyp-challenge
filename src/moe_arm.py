"""Gated MoE pIC50 arm (Oct 25-26: 'the TDI signal should help inform the
pIC50 signal - push the multitask setting harder').

Heads (loss_zoo emx layout): prim pIC50 x4 (hinge), TDIC pIC50
TDI-condition x4 (MSE), is_TDI x4 (BCE, PCGrad-surgered), aux.
INFERENCE MIXTURE (the point): p_k = sigmoid(is_TDI head_k);
    pred_k = prim_k + alpha * p_k * (tdic_k - prim_k)
Gate LEARNED (pcg1 TDI heads: 3A4 OOF MCC 0.34), experts LEARNED
(prim = direct-inhibition chemistry, tdic = TDI-condition chemistry),
association REAL (3A4 TDI+ pIC50 4.70 vs 3.80; 33% of rows).
Post-hoc calibration FAILED (honest CV): the conditioning must be learned
in-model. This is the learned version.

Training = pcgrad_arm recipe (prim hinge + aux MSE + cls BCE, PCGrad
surgery on FFN trunk). Eval = 5-fold OOF + alpha sweep per isoform.
Gate: 2D6 alpha-best gain > 0 with >=4/5 folds AND 3A4 not worse than
prim alone; L4 bootstrap before any ship.

Run (chemeleon env, ONE GPU job at a time):
  ~/miniforge3/envs/chemeleon/bin/python src/moe_arm.py --smoke
  ~/miniforge3/envs/chemeleon/bin/python src/moe_arm.py
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from chemprop import data
from lightning import seed_everything

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import ft_ext  # noqa: E402
import loss_zoo as L  # noqa: E402
import pcgrad_arm as PG  # noqa: E402
from interval_arm import bounds_by_smi, soft_rae, ISO, NPRIM, BOUNDC  # noqa: E402

MAX_EPOCHS = 50
PATIENCE = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auxw", type=float, default=0.3)
    ap.add_argument("--clsw", type=float, default=1.0)
    ap.add_argument("--fullft", action="store_true")
    a = ap.parse_args()
    tag = "moe" + ("" if a.seed == 0 else f"_s{a.seed}")
    if a.smoke:
        tag += "_smoke"
    L.set_cols(emx=True)
    n_aux, n_cls = L.NAUXD, L.NCLSD
    tdic0 = NPRIM + 8          # chembl4+pubchem4 then TDIC x4
    cls0 = NPRIM + n_aux       # is_TDI x4
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
    oof_full = np.full((len(ft), L.NALL), np.nan)
    test_preds = []
    per_fold_m = {iso: [] for iso in ISO}   # mixture (alpha=best)
    per_fold_p = {iso: [] for iso in ISO}   # prim only
    seed_everything(a.seed)
    t0 = time.time()
    for f in range(nfold):
        va_idx = np.where(folds == f)[0]
        va_df = allrows[ch_mask].iloc[va_idx]
        tr_pool = allrows[ch_mask].drop(index=va_idx).reset_index(drop=True)
        tr_extra = allrows[~ch_mask & lab]
        rng = np.random.default_rng(2000 * a.seed + f)
        if len(tr_extra) > ft_ext.EXT_CAP:
            kk = rng.choice(len(tr_extra), ft_ext.EXT_CAP, replace=False)
            tr_extra = tr_extra.iloc[kk]
        lab_tr = tr_pool[L.LABCOLS].notna().any(axis=1).values
        vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * PG.VAL_FRAC),
                        replace=False)
        fit_df = pd.concat([tr_pool.drop(index=vi), tr_extra],
                           ignore_index=True).reset_index(drop=True)
        ev_df = tr_pool.iloc[vi].reset_index(drop=True)
        (model, scaler, tr_dset, mu_p, sd_p, mu_a, sd_a,
         lo_sl, hi_sl, cls_sl, na, nc) = PG.make_model(fit_df, a.fullft, device)
        loader = data.build_dataloader(tr_dset, batch_size=PG.BATCH,
                                       num_workers=0)
        shared = list(model.predictor.ffn.parameters())
        if a.fullft:
            shared += [p for p in model.message_passing.parameters()
                       if p.requires_grad]
        opt = torch.optim.AdamW([
            {"params": model.predictor.ffn.parameters(), "weight_decay": PG.L2W},
            {"params": model.predictor.output_transform.parameters(),
             "weight_decay": 0.0},
            {"params": model.message_passing.parameters(), "weight_decay": PG.L2W},
        ], lr=1e-3)
        best, best_ep, bad = np.inf, 0, 0
        best_state = None
        for ep in range(maxep):
            model.train()
            for batch in loader:
                bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
                mask = targets.isfinite()
                targets = targets.nan_to_num(nan=0.0)
                bmg.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                Z = model.fingerprint(bmg, V_d, X_d)
                z = model.predictor.ffn(Z)
                lp, la, lc = PG.group_losses(
                    z, targets, mask, mu_p, sd_p, mu_a, sd_a,
                    lo_sl, hi_sl, cls_sl, na, nc, a.auxw, a.clsw)
                opt.zero_grad(set_to_none=True)
                comb, _ = PG.pcgrad_combine([lp, la, lc], shared, anchor=0)
                (lp + la + lc).backward()
                for p, gk in zip(shared, comb):
                    p.grad = gk.clone()
                opt.step()
            Pf = PG.predict(model, ev_df, L.NALL)
            ev_iso = ev_df[ISO].values.astype(float)
            smacro = float(np.mean([soft_rae(
                ev_iso[:, k][~np.isnan(ev_iso[:, k])],
                Pf[~np.isnan(ev_iso[:, k]), k],
                bb[f"{ISO[k]}_bndlo"].reindex(ev_df["SMILES"]).values[
                    ~np.isnan(ev_iso[:, k])],
                bb[f"{ISO[k]}_bndhi"].reindex(ev_df["SMILES"]).values[
                    ~np.isnan(ev_iso[:, k])]) for k in range(4)]))
            if smacro < best - 1e-4:
                best, best_ep, bad = smacro, ep, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
            print(f"[{tag}] fold {f} ep {ep} val_macro={smacro:.4f} "
                  f"best={best:.4f} t={time.time()-t0:.0f}s", flush=True)
        model.load_state_dict(best_state)
        Pf = PG.predict(model, va_df, L.NALL)
        oof_full[va_idx] = Pf[:, :L.NALL]
        if not a.smoke:
            Pte = PG.predict(model, test, L.NALL)
            test_preds.append(Pte[:, :L.NALL])
        # per-fold prim vs mixture (alpha fixed 0.5 eval; sweep at pooled)
        p_gate = 1.0 / (1.0 + np.exp(-Pf[:, cls0:cls0 + 4]))
        for k, iso in enumerate(ISO):
            m = ~np.isnan(yfull[iso][va_idx])
            prim = Pf[m, k]
            tdic = Pf[m, tdic0 + k]
            mix = prim + 0.5 * p_gate[m, k] * (tdic - prim)
            s_p = soft_rae(yfull[iso][va_idx][m], prim,
                           lo_full[iso][va_idx][m], hi_full[iso][va_idx][m])
            s_m = soft_rae(yfull[iso][va_idx][m], mix,
                           lo_full[iso][va_idx][m], hi_full[iso][va_idx][m])
            per_fold_p[iso].append(float(s_p))
            per_fold_m[iso].append(float(s_m))
        del model, opt, best_state
        torch.cuda.empty_cache()
        print(f"[{tag}] fold {f} DONE {time.time()-t0:.0f}s "
              f"prim={{{', '.join(f'{i}:{per_fold_p[i][-1]:.4f}' for i in ISO)}}} "
              f"mix={{{', '.join(f'{i}:{per_fold_m[i][-1]:.4f}' for i in ISO)}}}",
              flush=True)

    # pooled alpha sweep
    p_gate = 1.0 / (1.0 + np.exp(-oof_full[:, cls0:cls0 + 4]))
    print(f"[{tag}] POOLED alpha sweep:", flush=True)
    res = {}
    for k, iso in enumerate(ISO):
        yv = yfull[iso]
        m = ~np.isnan(yv) & ~np.isnan(oof_full[:, k]) & ~np.isnan(lo_full[iso]) & ~np.isnan(hi_full[iso])
        prim, tdic, pg = oof_full[m, k], oof_full[m, tdic0 + k], p_gate[m, k]
        lo, hi = lo_full[iso][m], hi_full[iso][m]
        row = {"prim": round(soft_rae(yv[m], prim, lo, hi), 4)}
        best_a, best_s = 0.0, row["prim"]
        for alpha in [0.25, 0.5, 0.75, 1.0]:
            s = soft_rae(yv[m], prim + alpha * pg * (tdic - prim), lo, hi)
            row[f"a{alpha}"] = round(s, 4)
            if s < best_s:
                best_a, best_s = alpha, s
        row["best_alpha"] = best_a
        row["folds_pos_mix"] = int(sum(
            1 for sp, sm in zip(per_fold_p[iso], per_fold_m[iso]) if sm < sp))
        res[iso] = row
        print(f"[{tag}] {iso} " + " ".join(f"{k2}={v2}" for k2, v2 in row.items()),
              flush=True)

    if a.smoke:
        print("SMOKE: pipeline check only", flush=True)
        return
    # TDI gate head MCC (learned gate quality)
    from sklearn.metrics import matthews_corrcoef as _mcc
    truth = allrows[ch_mask]
    for j, c in enumerate(L.CLSH4):
        yv = truth[c].values.astype(float)
        m = ~np.isnan(yv) & ~np.isnan(oof_full[:, cls0 + j])
        if m.sum() < 20:
            continue
        pred = (oof_full[m, cls0 + j] > 0.0).astype(int)
        print(f"[{tag}] gate head {c} n={int(m.sum())} "
              f"MCC={_mcc(yv[m].astype(int), pred):.4f}", flush=True)
    Pte = np.mean(np.stack(test_preds), axis=0)
    tp = pd.DataFrame(Pte[:, :NPRIM], columns=ISO)
    tp["gate1A2"], tp["gate2C9"] = (
        1 / (1 + np.exp(-Pte[:, cls0])), 1 / (1 + np.exp(-Pte[:, cls0 + 1])))
    tp["gate2D6"], tp["gate3A4"] = (
        1 / (1 + np.exp(-Pte[:, cls0 + 2])), 1 / (1 + np.exp(-Pte[:, cls0 + 3])))
    # predicted assay bounds (learned bound heads, raw scale via
    # output_transform): enables test-side clip-into-interval calibration
    b0 = NPRIM + n_aux + n_cls
    for k, iso in enumerate(ISO):
        tp[f"{iso}_plo"] = Pte[:, b0 + 2 * k]
        tp[f"{iso}_phi"] = Pte[:, b0 + 2 * k + 1]
    tp.insert(0, "SMILES", test["SMILES"])
    tp.to_csv(os.path.join(CACHE, f"ft_test_preds_{tag}.csv"), index=False)
    oof_df = pd.DataFrame(oof_full[:, :NPRIM], columns=ISO)
    for k, iso in enumerate(ISO):
        oof_df[f"{iso}_plo"] = oof_full[:, b0 + 2 * k]
        oof_df[f"{iso}_phi"] = oof_full[:, b0 + 2 * k + 1]
    oof_df["SMILES"] = ft["SMILES"].values
    oof_df.to_csv(os.path.join(CACHE, f"ft_oof_{tag}.csv"), index=False)
    print(f"[{tag}] ALL DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
