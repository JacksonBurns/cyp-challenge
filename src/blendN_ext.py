"""N-way z-space mixture including the external-aux multitask (ft_ext).

Candidates: gbm, emb, ft_frozen, ft_fullft_avg (seed-averaged). When
cache/ft_oof_ext.csv exists it joins the pool. Greedy forward selection
(with replacement, 20 rounds) per isoform. Writes cache/blendN_ext.json.
"""
import glob
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

CANDS = {"gbm": ["oof_blend.csv"], "emb": ["oof_emb_concat.csv"],
         "ft_frozen": ["ft_oof_frozen.csv"],
         "ft_fullft_avg": None}  # resolved by glob, z-averaged


def zavg(files):
    dfs = [pd.read_csv(f) for f in files]
    df = dfs[0].copy()
    for iso in R.ISOFORMS:
        if len(dfs) > 1:
            z = np.zeros(len(df))
            for d in dfs:
                p = d[iso].values.astype(float)
                z += (p - p.mean()) / p.std()
            df[iso] = z / len(dfs)
    return df


CANDS["ft_fullft_avg"] = sorted(glob.glob(os.path.join(CACHE, "ft_oof_fullft*.csv")))
import glob as _g
ext_files = sorted(_g.glob(os.path.join(CACHE, "ft_oof_ext*.csv")))
if ext_files:
    CANDS["ft_ext"] = ext_files

gate = json.load(open(os.path.join(CACHE, "oof_pearson.json")))
oofs = {k: zavg([f if os.path.isabs(f) else os.path.join(CACHE, f) for f in fs])
        for k, fs in CANDS.items()}

out = {}
for iso in R.ISOFORMS:
    y = targets[R.DIRECT[iso]].values
    m = ~np.isnan(y)
    zs = {}
    for k, df in oofs.items():
        p = df[iso].values.astype(float)
        z = np.full(len(p), np.nan)
        z[m] = (p[m] - p[m].mean()) / p[m].std()
        zs[k] = z
    keys = list(zs)
    w = {k: 0.0 for k in keys}
    cur = None
    for _ in range(20):
        bestk, bestr = None, -2
        for k in keys:
            cand_z = sum((w[j] + (1 if j == k else 0)) * zs[j] for j in keys)
            tot = sum(w.values()) + 1
            r = float(pearsonr(y[m], (cand_z / tot)[m]).statistic)
            if r > bestr:
                bestr, bestk = r, k
        w[bestk] += 1
        cur = bestr
    w_norm = {k: round(v / sum(w.values()), 2) for k, v in w.items() if v > 0}
    singles = {k: round(float(pearsonr(y[m], zs[k][m]).statistic), 3) for k in keys}
    out[iso] = {"pearson": cur, "weights": w_norm, "singles": singles, "gate": gate[iso]}
    print(iso, f"{cur:.3f}", w_norm, "singles", singles, flush=True)

mac = float(np.mean([out[i]["pearson"] ** 2 for i in R.ISOFORMS]))
print("macro R2:", round(mac, 4))
with open(os.path.join(CACHE, "blendN_ext.json"), "w") as fh:
    json.dump(out, fh, indent=2)
