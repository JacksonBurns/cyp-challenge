"""R6: physics-gated interval-center placement (post-processing).

Novel vs gate C/R5: those shifted flagged rows toward the FLOOR. R6 shifts
toward the interval CENTER: for high floor-hazard rows, predictions sit
too high (signed residual -1.2..-2.4 on floor rows) and metric cost is
below lo. Optimal single-point prediction minimizing soft-RAE for an
interval [lo,hi] is the CENTER (cost (hi-lo)/2). High-hazard rows have
WIDE assay intervals (lo far below P), so shrinking P toward (lo+hi)/2
strictly reduces expected cost. Gate per isoform: honest 5-fold, w* grid
[0,1] by inner CV on train folds; measure on held-out fold with
zoo6-blend P, bounds lo/hi, hazard head on stack+ADME2.
"""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402
import uq_localescal as UL  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CACHE = H.CACHE
import json
W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))


def soft_rae(y, p, lo, hi):
    return np.nanmean(np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None))


def zoo6_oof(iso):
    fams = H.POOLS["zoo6"]
    P = np.zeros(len(H.Xtr))
    for f in fams:
        p = H.raw_oof_fam(f)[iso]
        P += np.asarray(p.reindex(H.Xtr.index).values, dtype=float) * W[iso]["w"][f]
    wsum = sum(abs(W[iso]["w"][f]) for f in fams)
    return P / wsum


def main():
    A2 = np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy"))
    Xbase = H.Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    Xa = np.hstack([Xbase, np.nan_to_num(A2, nan=0.0)])
    smi = H.Xtr["SMILES"].values
    for iso in ISO:
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        P = zoo6_oof(iso)
        m = np.isfinite(y) & np.isfinite(P) & np.isfinite(lo) & np.isfinite(hi)
        floor = (y <= np.nanmin(y) + 1e-9)
        hb = np.where(m, floor.astype(int), -1)
        # OOF hazard prob over all scored rows
        hz = np.full(len(smi), np.nan)
        idx = np.where(m)[0]
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, te in skf.split(idx, hb[idx]):
            trc, tec = idx[tr], idx[te]
            g = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               verbose=-1, random_state=0)
            g.fit(Xa[trc], hb[trc])
            hz[tec] = g.predict_proba(Xa[tec])[:, 1]
        center = (lo + hi) / 2.0
        # grid w in [0, .5] applied on hz-weighted blend of P and center
        best_w, best_gain = 0.0, 0.0
        c0 = soft_rae(y[m], P[m], lo[m], hi[m])
        for w in np.linspace(0, 0.5, 26):
            Pw = P + w * hz * (center - P)
            g = c0 - soft_rae(y[m], Pw[m], lo[m], hi[m])
            if g > best_gain:
                best_w, best_gain = w, g
        # honest split-CV of the rule: fit w on 4/5 of scored rows, eval 1/5
        kf5 = KFold = __import__("sklearn.model_selection", fromlist=["KFold"]).KFold
        gains, wins = [], 0
        for tr, te in kf5(5, shuffle=True, random_state=1).split(idx):
            trc, tec = idx[tr], idx[te]
            bw, bg = 0.0, 0.0
            c0t = soft_rae(y[trc], P[trc], lo[trc], hi[trc])
            for w in np.linspace(0, 0.5, 26):
                Pw = P + w * hz * (center - P)
                gg = c0t - soft_rae(y[trc], Pw[trc], lo[trc], hi[trc])
                if gg > bg:
                    bw, bg = w, gg
            Pw = P + bw * hz * (center - P)
            g = soft_rae(y[tec], P[tec], lo[tec], hi[tec]) - \
                soft_rae(y[tec], Pw[tec], lo[tec], hi[tec])
            gains.append(g)
            wins += g > 0
        print(f"R6 {iso}: best_w {best_w:.2f} insample gain {best_gain:+.4f} | "
              f"CV gain {np.mean(gains):+.4f} wins {wins}/5", flush=True)


if __name__ == "__main__":
    main()
