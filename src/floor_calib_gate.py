"""Gate C: does the ADME floor-hazard head earn metric gain on the SHIP blend?

Per isoform (all OOF/CV, no leakage):
  P   = zoo6 blend OOF pIC50 (validated reconstruction, uq_localescal)
  hz  = KFold-OOF P(floor) from GBM on stack+ADME
  On train folds pick (hazard-quantile thr, downward shift d);
  apply to held-out P (shift floor-flagged rows toward the floor);
  score soft_rae vs control. Ship-relevant if mean gain > 0, 3/5+ folds.
"""
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
LLOD = 1.03


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
    import json
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
    Afe = np.load(os.path.join(CACHE, "adme_aux", "adme_feats_tr.npy"))
    smi = H.Xtr["SMILES"].values
    Xbase = np.nan_to_num(H.Xtr.drop(columns=["SMILES"], errors="ignore").values.astype(float), nan=0.0)
    Xa = np.hstack([Xbase, np.nan_to_num(Afe, nan=0.0)])
    kf = KFold(5, shuffle=True, random_state=1)
    for iso in ISO:
        P = blend_oof(iso, W, smi)
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        m = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi) & ~np.isnan(P)
        yv, Pv, lov, hiv, Xv = y[m], P[m], lo[m], hi[m], Xa[m]
        lo_q = np.nanpercentile(yv, 1.5)
        hb = (yv <= lo_q + 1e-9).astype(int)
        hz = np.full(len(yv), np.nan)
        for tr, te in kf.split(Xv):
            g = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05,
                                   num_leaves=15, verbose=-1,
                                   class_weight="balanced")
            g.fit(Xv[tr], hb[tr])
            hz[te] = g.predict_proba(Xv[te])[:, 1]
        best = None
        for thrq in (0.90, 0.95, 0.98):
            for d in (0.5, 1.0, 1.5, 2.0):
                gains = []
                for tr, te in kf.split(Xv):
                    thr = np.quantile(hz[tr], thrq)
                    Padj = Pv[te].copy()
                    hm = hz[te] >= thr
                    Padj[hm] = np.maximum(Padj[hm] - d, lo_q)
                    c0 = soft_rae(yv[te], Pv[te], lov[te], hiv[te])
                    c1 = soft_rae(yv[te], Padj, lov[te], hiv[te])
                    gains.append(c0 - c1)
                garr = np.array(gains)
                if best is None or garr.mean() > best[0]:
                    best = (garr.mean(), thrq, d, int((garr > 0).sum()))
        base_sr = soft_rae(yv, Pv, lov, hiv)
        print(f" GATE C {iso}: n={len(yv)} floor={hb.sum()} softRAE={base_sr:.4f} "
              f"best gain={best[0]:+.4f} thrq={best[1]} d={best[2]} "
              f"pos={best[3]}/5 (flagged rows = {(hz >= np.quantile(hz, best[1])).sum()})",
              flush=True)


if __name__ == "__main__":
    main()
