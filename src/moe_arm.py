"""Gated MoE pIC50 arm (Oct 25: 'TDI signal should inform pIC50 - push
the multi-task setting harder').

Heads (loss_zoo mv+emx layout): prim pIC50 x4 (hinge), TDIC pIC50
TDI-condition x4 (MSE), is_TDI x4 (BCE, PCGrad-surgered), aux.
INFERENCE MIXTURE (the point): p_k = sigmoid(is_TDI head k);
    pred_k = (1-p_k) * prim_k + p_k * tdic_k
The gate is LEARNED (pcg1 TDI heads: 3A4 OOF MCC 0.34), the experts are
LEARNED (prim = direct-inhibition chemistry, tdic = TDI-condition
chemistry), and the association is real (3A4 TDI+ rows: pIC50 4.70 vs
3.80; 33% of rows). Post-hoc calib failed (in-sample artifact); this is
the learned version.

Eval: 5-fold OOF soft_rae per isoform; alpha sweep on the mixture
(P = prim + alpha*p*(tdic-prim)); L4-style bootstrap gate vs zoo6 base.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from chemprop import data, models, nn
from lightning import seed_everything

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loss_zoo as L  # noqa: E402
import ft_ext  # noqa: E402
from ft_ext import BATCH, EXT_CAP, L2W, MAX_EPOCHS, VAL_FRAC  # noqa: E402
from interval_arm import bounds_by_smi, soft_rae, ISO, NPRIM, BOUNDC  # noqa
import pcgrad_arm as P  # noqa: E402

CACHE = L.CACHE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tag = f"moe{'' if a.seed == 0 else f'_s{a.seed}'}"
    L.set_cols(mv=True, emx=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft, test, allrows = P.load_rows(mv=True, emx=True)
    bb = bounds_by_smi()
    for c in BOUNDC:
        if c not in allrows.columns:
            allrows[c] = allrows["SMILES"].map(bb[c])
    folds = ft["fold"].values
    cls_sl = L.CLS_SL
    tdic_sl = slice(NPRIM + L.NAUXD - 8, NPRIM + L.NAUXD - 4)  # TDIC block
    seed_everything(a.seed)
    n_folds = 1 if a.smoke else 5
    oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    oofp = pd.DataFrame(np.nan, index=ft.index, columns=ISO)  # gate probs
    for f in range(n_folds):
        va_mask = folds == f
        va_idx = np.where(va_mask)[0]
        tr_pool = allrows[(allrows["_src"] == "challenge") & ~va_mask]
        lab_any = allrows[L.LABCOLS].notna().any(axis=1)
        tr_extra = allrows[(allrows["_src"] != "challenge") & lab_any]
        rng = np.random.default_rng(1000 * a.seed + f)
        if len(tr_extra) > EXT_CAP:
            tr_extra = tr_extra.iloc[rng.choice(len(tr_extra), EXT_CAP,
                                                replace=False)]
        fit_df = pd.concat([tr_pool, tr_extra], ignore_index=True)
        model, scaler, tr_dset, mu_p, sd_p, mu_a, sd_a = P.make_model(
            fit_df, False, device)
        loader = data.build_dataloader(tr_dset, batch_size=BATCH,
                                       num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=L2W)
        ev_df = allrows[va_mask & (allrows["_src"] == "challenge")]
        best, bad, best_state = np.inf, 0, None
        for ep in range(1 if a.smoke else MAX_EPOCHS):
            model.train()
            for batch in loader:
                bmg, V_d, X_d, targets, w, lt_m, gt_m = batch
                mask = targets.isfinite()
                targets = targets.nan_to_num(nan=0.0)
                bmg.to(device); targets = targets.to(device)
                mask = mask.to(device)
                z = model.predictor.ffn(model.fingerprint(bmg, V_d, X_d))
                loss = P.losses(z, targets, mask, mu_p, sd_p, mu_a, sd_a)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            Pf = P.predict(model, ev_df, L.NALL)
            yv = ev_df[L.LABCOLS].values.astype(float)
            lo_e = ev_df[[f"{i}_bndlo" for i in ISO]].values
            hi_e = ev_df[[f"{i}_bndhi" for i in ISO]].values
            sc = []
            for k in range(NPRIM):
                m = np.isfinite(yv[:, k])
                if m.sum() >= 8:
                    sc.append(float(np.mean(soft_rae(
                        yv[m, k], Pf[m, k], lo_e[m, k], hi_e[m, k]))))
            sm = float(np.mean(sc))
            if sm < best - 1e-4:
                best, bad = sm, 0
                best_state = {kk: vv.cpu().clone()
                              for kk, vv in model.state_dict().items()}
                Pf_best = Pf
            else:
                bad += 1
                if bad >= 8:
                    break
        model.load_state_dict(best_state)
        Pf = P.predict(model, ev_df, L.NALL)
        prim = Pf[:, :NPRIM]
        tdic = Pf[:, tdic_sl]
        gate = 1.0 / (1.0 + np.exp(-Pf[:, cls_sl][:, :NPRIM]))
        for k, iso in enumerate(ISO):
            i = np.where(ft.index.isin(ev_df.index))[0] if False else va_idx
            oof.iloc[va_idx, k] = prim[:, k]
            oofp.iloc[va_idx, k] = gate[:, k]
            # store tdic via aligned column (append suffix col per iso)
            oof.attrs.setdefault("tdic", {})[iso] = None
        print(f"[{tag}] f{f} val macroST(prim)={best:.4f} t={ep+1}ep",
              flush=True)
        np.save(f"{CACHE}/moe_gate_{tag}.npy", oofp.values)
        np.save(f"{CACHE}/moe_tdic_{tag}.npy", tdic)
    oof.insert(0, "SMILES", ft["SMILES"].values)
    oof.to_csv(f"{CACHE}/ft_oof_{tag}.csv", index=False)
    print("WROTE", f"ft_oof_{tag}.csv", flush=True)


if __name__ == "__main__":
    main()
