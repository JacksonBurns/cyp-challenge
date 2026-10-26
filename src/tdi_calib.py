"""TDI-conditioned calibration of the regression blend (CPU, Oct 25).

Fact: TDI+ molecules sit at systematically shifted potency:
  3A4: 4.70 (n=764, 33% of data!) vs 3.80 non-TDI (+0.90)
  1A2: +0.48 | 2D6: -0.36 | 2C9: +0.11
If the regression ignores TDI status, whole clusters are biased.

Test on TRAIN: blend (zoo6 weights) residual per isoform, split by
thresholded TDI status from OOF TDI probs (joint_oof_tdi_pcgrad.csv,
calibrated threshold). Fit per-isoform shift delta = mean residual on
TDI+ minus mean residual on TDI-; apply delta to TDI-prob-weighted rows;
recompute soft_rae + bootstrap CI. Gate: gain > 0, CI excludes 0.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
ISO = H.ISO
BJ = json.load(open(os.path.join(CACHE, "hinge_blend_zoo6.json")))
W = {iso: BJ[iso]["weights_fitted_top"] for iso in ISO}


def soft_rae(y, yhat, lo, hi):
    err = np.clip(yhat - hi, 0, None) + np.clip(lo - yhat, 0, None)
    base = np.clip(np.mean(y) - hi, 0, None) + np.clip(lo - np.mean(y), 0, None)
    return err.sum() / base.sum()


def blend_pred(oofs, iso, names):
    P = None
    for k in names:
        w = W[iso].get(k, 0.0)
        if w == 0:
            continue
        v = oofs[k][iso].values
        P = w * v if P is None else P + w * v
    return P


def main():
    names = list(W[ISO[0]].keys())
    oofs = {k: H.raw_oof_fam(k) for k in names}
    smi = H.Xtr["SMILES"].values
    tdi = pd.read_csv(os.path.join(CACHE, "joint_oof_tdi_pcgrad.csv")) \
        .drop_duplicates("SMILES").set_index("SMILES")
    emx = pd.read_csv(os.path.join(H.R.DATA, "cyp-challenge-TRAIN_Emax.csv")) \
        .drop_duplicates("SMILES").set_index("SMILES")
    rng = np.random.default_rng(0)
    out = {}
    for iso in ISO:
        y = H.targets[H.R.DIRECT[iso]].values
        lo, hi = H.bounds[f"{iso}_lo"].values, H.bounds[f"{iso}_hi"].values
        P = blend_pred(oofs, iso, names)
        lab = np.isfinite(y) & np.isfinite(P) & np.isfinite(lo) & np.isfinite(hi)
        col = f"is_TDI_{iso.split('CYP')[1]}"
        p = tdi[col].reindex(smi).values.astype(float) \
            if col in tdi.columns else None
        # true TDI labels where known (for oracle bound)
        tcol = f"{iso}_is_TDI"
        yt = None
        if tcol in emx.columns:
            s = emx[tcol].reindex(smi)
            s = s.map(lambda v: 1.0 if str(v).strip().lower() in
                      ("true", "1", "1.0") else
                      (0.0 if str(v).strip().lower() in ("false", "0", "0.0")
                       else np.nan))
            yt = s.values
        idx = np.where(lab)[0]

        def score(prob_mask, alpha=1.0):
            delta = np.nan
            m = prob_mask[idx]
            if np.isfinite(m).sum() == 0:
                return None
            resid = y[idx] - P[idx]
            dpos = np.nanmean(resid[m == 1])
            dneg = np.nanmean(resid[m == 0])
            delta = alpha * (dpos - dneg)
            Padj = P[idx] + delta * np.nan_to_num(m, nan=0.0)
            return soft_rae(y[idx], Padj, lo[idx], hi[idx])

        base = soft_rae(y[idx], P[idx], lo[idx], hi[idx])
        res = {"base": base}
        for name, m in [("oracle_TDI", yt), ("pcgrad_p>0.5",
                          None if p is None else (p > 0.5).astype(float))]:
            if m is None:
                continue
            new = score(m)
            # bootstrap on delta
            gains = []
            mm = np.nan_to_num(m, nan=0.0)
            resid = y[idx] - P[idx]
            for _ in range(300):
                s = rng.integers(0, len(idx), len(idx))
                d_pos = resid[s][mm[s] == 1].mean()
                d_neg = resid[s][mm[s] == 0].mean()
                Padj = P[idx][s] + (d_pos - d_neg) * mm[s]
                gains.append(soft_rae(y[idx][s], P[idx][s], lo[idx][s], hi[idx][s])
                             - soft_rae(y[idx][s], Padj, lo[idx][s], hi[idx][s]))
            res[name] = {"new": new, "gain": base - new,
                         "boot_mean": float(np.mean(gains)),
                         "ci": [float(np.percentile(gains, 2.5)),
                                float(np.percentile(gains, 97.5))]}
            print(f"{iso} {name}: base={base:.4f} new={new:.4f} "
                  f"gain={base-new:+.4f} boot={np.mean(gains):+.4f} "
                  f"ci=[{np.percentile(gains,2.5):+.4f},"
                  f"{np.percentile(gains,97.5):+.4f}] "
                  f"n_tdi+={int(np.nansum(m[idx]==1))}", flush=True)
        out[iso] = res
    json.dump(out, open(os.path.join(CACHE, "tdi_calib.json"), "w"),
              default=float)


if __name__ == "__main__":
    main()
