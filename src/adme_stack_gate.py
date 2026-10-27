"""R4: ADME predictions as STACKED INPUTS to a blend meta-model.

User idea: use ADME model predictions as inputs to other models.
Setup per isoform, honest CV (no leakage):
  meta-features = zoo6 blend P, per-family spread (UQ), 8 predicted-ADME
  features, hazard-free (floor label NOT a feature).
  meta-model   = GBM regressor trained to predict y from meta-features.
Control = zoo6 blend alone (identity-ish: P is the dominant feature,
so control MAE/softRAE = P's own CV score through a GBM fit).
Ship-relevant: meta soft_rae (ST-RAE proxy) beats control on 4/5 folds.

NOTE: P here is the OOF blend; meta CV on top of OOF P is honest w.r.t.
y. The stacked model consumes P + ADME, so gains are attributable to
ADME conditioning the blend.
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


def fam_spread(iso, W, smi):
    cols = []
    for fam, w in W[iso]["w"].items():
        if not isinstance(w, (int, float)) or abs(w) < 1e-9:
            continue
        v = H.raw_oof_fam(fam)[iso].reindex(smi).values.astype(float)
        cols.append(v)
    M = np.stack(cols, axis=1)
    return np.nan_to_num(np.nanstd(M, axis=1), nan=0.0)


def main():
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
    Afe = np.load(os.path.join(CACHE, "adme_aux", "adme_feats_tr.npy"))
    smi = H.Xtr["SMILES"].values
    kf = KFold(5, shuffle=True, random_state=11)
    results = {}
    for iso in ISO:
        P = blend_oof(iso, W, smi)
        U = fam_spread(iso, W, smi)
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        m = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi) & ~np.isnan(P)
        X = np.column_stack([P[m], U[m], np.nan_to_num(Afe[m], nan=0.0)])
        yv, lov, hiv, Pv = y[m], lo[m], hi[m], P[m]
        base = soft_rae(yv, Pv, lov, hiv)
        preds = np.zeros(len(yv))
        for tr, te in kf.split(X):
            g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=15, verbose=-1,
                                  min_child_samples=40)
            g.fit(X[tr], yv[tr])
            preds[te] = g.predict(X[te])
        meta = soft_rae(yv, preds, lov, hiv)
        # fold-level win rate
        wins = []
        for tr, te in kf.split(X):
            g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=15, verbose=-1,
                                  min_child_samples=40)
            g.fit(X[tr], yv[tr])
            c0 = soft_rae(yv[te], Pv[te], lov[te], hiv[te])
            c1 = soft_rae(yv[te], g.predict(X[te]), lov[te], hiv[te])
            wins.append(c0 - c1)
        gain = base - meta
        results[iso] = (gain, float(np.mean(np.array(wins) > 0)))
        print(f" R4 {iso}: control softRAE={base:.4f} meta={meta:.4f} "
              f"gain={gain:+.4f} fold_wins={np.mean(np.array(wins)>0):.2f}",
              flush=True)
    pos = [i for i, (g, w) in results.items() if g > 0]
    print("R4 positive isoforms:", pos if pos else "NONE")


if __name__ == "__main__":
    main()
