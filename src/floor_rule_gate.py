"""R5: floor-hazard RULE, test-applicable, honest CV gate.

The post-processing route (R3/gate C) shifted rows flagged by an OOF
hazard head and was dead. R5 differs: the hazard head is trained ONLY on
train-fold labels, applied to held-out rows exactly as it would be at
test time (no OOF leakage). Grid = (thr-quantile, shift d), chosen on the
train folds (inner CV), then applied once to the held-out fold and scored.

If gain > 0 on >=4/5 folds for any isoform, this is a shippable rule.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CACHE = "/home/jackson/cyp-challenge/cache"


def soft_rae(y, yhat, lo, hi):
    err = np.clip(yhat - hi, 0, None) + np.clip(lo - yhat, 0, None)
    return np.nanmean(err)


def blend_oof(iso, W, smi):
    P = np.zeros(len(smi))
    tot = np.zeros(len(smi))
    for fam, w in W[iso]["w"].items():
        if not isinstance(w, (int, float)) or abs(w) < 1e-9:
            continue
        v = H.raw_oof_fam(fam)[iso].reindex(smi).values.astype(float)
        ok = ~np.isnan(v)
        P[ok] += w * v[ok]
        tot[ok] += w
    P[tot > 0] /= tot[tot > 0]
    return P


def main():
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
    Afe = np.load(os.path.join(CACHE, "adme_aux", "adme_feats_tr.npy"))
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    smi = H.Xtr["SMILES"].values
    folds = ft["fold"].reindex(smi).values
    Xbase = np.nan_to_num(H.Xtr.drop(columns=["SMILES"], errors="ignore").values.astype(float), nan=0.0)
    Xa = np.hstack([Xbase, np.nan_to_num(Afe, nan=0.0)])
    GRID = [(t, d) for t in (0.90, 0.95, 0.98) for d in (0.5, 1.0, 1.5, 2.0)]
    outer = KFold(5, shuffle=True, random_state=3)
    for iso in ISO:
        P = blend_oof(iso, W, smi)
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        m = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi) & ~np.isnan(P)
        idx = np.where(m)[0]
        yv, Pv, lov, hiv, Xv, fv = y[idx], P[idx], lo[idx], hi[idx], Xa[idx], folds[idx]

        def hz_fit_apply(tr_idx, te_idx, thr, d, seed=0):
            ytr = yv[tr_idx]
            lo_q = np.percentile(ytr, 1.5)
            hb = (ytr <= lo_q + 1e-9).astype(int)
            g = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05,
                                   num_leaves=15, verbose=-1,
                                   class_weight="balanced")
            g.fit(Xv[tr_idx], hb)
            hz = g.predict_proba(Xv[te_idx])[:, 1]
            # test-applicable: flag the top (1-thr) fraction by hazard rank
            thrq = np.quantile(hz, thr)
            Padj = Pv[te_idx].copy()
            hm = hz >= thrq
            Padj[hm] = np.maximum(Padj[hm] - d, lo_q)
            return soft_rae(yv[te_idx], Padj, lov[te_idx], hiv[te_idx]), hm.sum()

        gains = []
        for tr_e, te_e in outer.split(idx):
            # pick hyperparameter on inner CV of outer-train rows
            best, best_s = None, None
            inner = KFold(4, shuffle=True, random_state=5)
            for thr, d in GRID:
                gg = []
                for tr_i, te_i in inner.split(idx[tr_e]):
                    a = tr_e[tr_i]
                    b = tr_e[te_i].astype(int)
                    c0 = soft_rae(yv[b], Pv[b], lov[b], hiv[b])
                    c1, _ = hz_fit_apply(a, b, thr, d)
                    gg.append(c0 - c1)
                if best is None or np.mean(gg) > best:
                    best, best_s = np.mean(gg), (thr, d)
            thr, d = best_s
            c0 = soft_rae(yv[te_e], Pv[te_e], lov[te_e], hiv[te_e])
            c1, nflag = hz_fit_apply(tr_e, te_e, thr, d)
            gains.append(c0 - c1)
        g = np.array(gains)
        print(f" R5 {iso}: gain={g.mean():+.4f} pos={int((g>0).sum())}/5 "
              f"per-fold={np.round(g,3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
