"""Evaluate CheMeLeon fine-tune OOF (per-isoform + multitask) vs the gate.

Reads cache/ft_oof_<tag>.csv (rows = ft_data order) and reports per-isoform
OOF Pearson + ST-RAE (raw) and z-mixture vs the shipped blend (cache/oof_blend.csv).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()

tag = sys.argv[1] if len(sys.argv) > 1 else "frozen"
ft = pd.read_csv(os.path.join(CACHE, f"ft_oof_{tag}.csv"))
gate = json.load(open(os.path.join(CACHE, "oof_pearson.json")))

print(f"{'iso':7s} {'pearson':>8s} {'st_rae_raw':>10s} {'mix_pearson':>11s} {'w*':>4s} gate")
report = {}
for iso in R.ISOFORMS:
    y = targets[R.DIRECT[iso]].values
    m = ~np.isnan(y)
    p = ft[iso].values
    valid = m & ~np.isnan(p)
    r = float(pearsonr(y[valid], p[valid]).statistic)
    lo, hi = bounds[f"{iso}_lo"].values, bounds[f"{iso}_hi"].values
    ra = float(R.soft_rae(y[valid], p[valid], lo[valid], hi[valid]))
    # z-mixture vs shipped blend OOF
    b = pd.read_csv(os.path.join(CACHE, "oof_blend.csv"))[iso].values
    zb = np.full(len(p), np.nan)
    zp = np.full(len(p), np.nan)
    zb[valid] = (b[valid] - b[valid].mean()) / b[valid].std()
    zp[valid] = (p[valid] - p[valid].mean()) / p[valid].std()
    best = (0, -2)
    for w in np.arange(0, 1.01, 0.1):
        z = w * zb + (1 - w) * zp
        rr = float(pearsonr(y[valid], z[valid]).statistic)
        if rr > best[0]:
            best = (rr, round(float(w), 1))
    report[iso] = {"pearson": r, "st_rae_raw": ra, "mix_pearson": best[0], "w_blend": best[1]}
    print(f"{iso:7s} {r:8.3f} {ra:10.3f} {best[0]:11.3f} {best[1]:4.1f} {gate[iso]:.3f}")

with open(os.path.join(CACHE, f"ft_eval_{tag}.json"), "w") as fh:
    json.dump(report, fh, indent=2)
