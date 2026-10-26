"""UQ-direction + bound-clip calibration test (Oct 25 evening).

Metric fact: soft_rae charges err = clip(yhat-hi,0,.)+clip(lo-yhat,0,.)
- predictions INSIDE the assay interval [lo,hi] cost ZERO.

Q1 (direction): on high-UQ rows (cross-family disagreement top decile),
    is the blend signed residual (y - pred) systematically positive
    (we predict LOW, near LLOD) or negative?
Q2 (fix): does clipping predictions into [lo,hi] - globally, or only on
    high-UQ rows - improve pooled soft_rae? Bootstrap 95% CI on delta.

Pre-registered gate: adopt only if mean delta > 0 with CI excluding 0.
CPU only. Blend = zoo6 fitted weights on family OOFs.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402


def soft_rae(y, yhat, lo, hi):
    err = np.clip(yhat - hi, 0, None) + np.clip(lo - yhat, 0, None)
    base = np.clip(np.mean(y) - hi, 0, None) + np.clip(lo - np.mean(y), 0, None)
    return err.sum() / base.sum()

CACHE = H.CACHE
ISO = H.ISO
BJ = json.load(open(os.path.join(CACHE, "hinge_blend_zoo6.json")))
W = {iso: BJ[iso]["weights_fitted_top"] for iso in ISO}
NAMES = {iso: list(W[iso].keys()) for iso in ISO}


def main():
    oofs = {k: H.raw_oof_fam(k) for k in set(sum(NAMES.values(), []))}
    uq = pd.read_csv(os.path.join(CACHE, "uq_cross_spread.csv"), index_col=0)
    rows = np.arange(len(H.Xtr["SMILES"]))
    smi = H.Xtr["SMILES"].values
    print("Q1: signed residual on high-UQ decile (resid = y - pred)",
          flush=True)
    summary = {}
    clips = {}
    rng = np.random.default_rng(0)
    for iso in ISO:
        y = H.targets[H.R.DIRECT[iso]].values
        lo, hi = H.bounds[f"{iso}_lo"].values, H.bounds[f"{iso}_hi"].values
        lab = np.isfinite(y)
        P = np.full(len(smi), np.nan)
        wsum = 0.0
        for k in NAMES[iso]:
            w = W[iso].get(k, 0.0)
            P = np.where(np.isfinite(P), P + w * oofs[k][iso].values,
                         w * oofs[k][iso].values)
        u = uq[f"{iso}_uq"].reindex(smi).values
        m = lab & np.isfinite(P) & np.isfinite(u) & np.isfinite(lo) & np.isfinite(hi)
        idx = rows[m]
        resid = y[idx] - P[idx]
        u_hi = u[idx] >= np.quantile(u[idx], 0.9)
        out_lo = P[idx] < lo[idx]
        out_hi = P[idx] > hi[idx]
        print(f" {iso}: n={len(idx)} | high-UQ rows: resid mean="
              f"{resid[u_hi].mean():+.3f} med={np.median(resid[u_hi]):+.3f} "
              f"frac_below_lo={out_lo[u_hi].mean():.2f} "
              f"frac_above_hi={out_hi[u_hi].mean():.2f} | low-UQ resid mean="
              f"{resid[~u_hi].mean():+.3f} frac_out={np.mean(out_lo|out_hi)}",
              flush=True)
        # Q2: clip variants (blend preds, labeled rows)
        base = soft_rae(y[idx], P[idx], lo[idx], hi[idx])
        v_all = np.clip(P[idx], lo[idx], hi[idx])
        u90 = np.quantile(u[idx], 0.9)
        v_uq = np.where(u[idx] >= u90, np.clip(P[idx], lo[idx], hi[idx]),
                        P[idx])
        v_uq5 = np.where(u[idx] >= np.quantile(u[idx], 0.5),
                         np.clip(P[idx], lo[idx], hi[idx]), P[idx])
        def boot(a, b, n=400):
            d = []
            for _ in range(n):
                s = rng.integers(0, len(idx), len(idx))
                d.append(soft_rae(y[idx][s], a[s], lo[idx][s], hi[idx][s])
                         - soft_rae(y[idx][s], b[s], lo[idx][s], hi[idx][s]))
            return float(np.mean(d)), np.percentile(d, [2.5, 97.5])
        for name, v in [("clip_all", v_all), ("clip_uq90", v_uq),
                        ("clip_uq50", v_uq5)]:
            new = soft_rae(y[idx], v, lo[idx], hi[idx])
            bm, ci = boot(v, P[idx])
            summary[f"{iso}_{name}"] = {"base": base, "new": new,
                                        "delta": base - new,
                                        "boot_mean": bm, "ci": ci.tolist()}
            print(f"  {iso} {name}: base={base:.4f} new={new:.4f} "
                  f"gain={base-new:+.4f} boot={bm:+.4f} ci=[{ci[0]:.4f},"
                  f"{ci[1]:.4f}]", flush=True)
        clips[iso] = {"u90": float(u90)}
    json.dump(summary, open(os.path.join(CACHE, "uq_clipcal.json"), "w"),
              default=float)


if __name__ == "__main__":
    main()
