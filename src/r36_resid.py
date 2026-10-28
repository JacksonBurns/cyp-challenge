"""R3-6: cross-isoform residual co-structure calibration.

Q: do the four zoo6-blend OOF residuals share structure (a shared
assay-difficulty factor)? If isoform-i residual is predictable from
the other isoforms' residuals + family disagreement + UQ spread, a
calibration correction exists.

Nested CV (honest): outer folds predict residual from
[other-3 residuals, fam_spread]; gate = MAE reduction and ST-RAE
reduction when subtracting beta * predicted residual.
Gate: ST-RAE gain > 0.005 per isoform (ship-relevant) or MAE > 0.005.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import hinge_blend as H  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def blend_oof(iso, W):
    smi = H.Xtr["SMILES"].values
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


def fam_spread(iso, W):
    smi = H.Xtr["SMILES"].values
    cols = []
    for fam, w in W[iso]["w"].items():
        if not isinstance(w, (int, float)) or abs(w) < 1e-9:
            continue
        cols.append(H.raw_oof_fam(fam)[iso].reindex(smi).values.astype(float))
    M = np.stack(cols, axis=1)
    return np.nan_to_num(np.nanstd(M, axis=1))


def sr(y, p, lo, hi):
    err = np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)
    num = np.sqrt(np.mean(err ** 2))
    rho = spearmanr(y, p).statistic
    return num * (2.0 - rho)


def main():
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
    smi = H.Xtr["SMILES"].values
    P = {i: blend_oof(i, W) for i in ISO}
    SD = {i: fam_spread(i, W) for i in ISO}
    for iso in ISO:
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].values.astype(np.float64)
        lo = H.bounds[f"{iso}_lo"].reindex(H.Xtr["SMILES"]).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(H.Xtr["SMILES"]).values.astype(float)
        ok = np.isfinite(y) & np.isfinite(P[iso])
        idx = np.where(ok)[0]
        res_i = y[idx] - P[iso][idx]
        feats = np.column_stack(
            [y[idx] - P[o][idx] for o in ISO if o != iso] + [SD[iso][idx]]
        )
        okn = np.isfinite(feats).all(1)
        r2, X2, ii = res_i[okn], feats[okn], idx[okn]
        kf = KFold(5, shuffle=True, random_state=0)
        pred = np.zeros(len(r2))
        for tr, te in kf.split(X2):
            g = Ridge(alpha=10.0).fit(X2[tr], r2[tr])
            pred[te] = g.predict(X2[te])
        rho = np.corrcoef(pred, r2)[0, 1]
        best_b, best_gain, best_sr = 0.0, 0.0, 0.0
        for b in np.linspace(0, 1, 21):
            g = np.mean(np.abs(r2 - b * pred)) - np.mean(np.abs(r2))
            if g < best_gain:
                best_gain, best_b = g, b
        p_new = P[iso][ii] + best_b * pred
        sr0 = sr(y[ii], P[iso][ii], lo[ii], hi[ii])
        sr1 = sr(y[ii], p_new, lo[ii], hi[ii])
        print(f"{iso}: corr(pred_res,res)={rho:+.3f} best_beta={best_b:.2f} "
              f"MAE gain {-best_gain:+.4f} ST-RAE {sr0:.4f}->{sr1:.4f} "
              f"({sr0-sr1:+.4f}) n={len(r2)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
