"""Nested (honest) evaluation of the greedy blend weights.

Weights are chosen greedily on 4 scaffold folds and scored on the held-out
fold; compares 4-way seed-avg pool vs 5-way pool that adds ft_ext. This is
the OOF-selection-overfitting check for NOTES sec 10 blend claims.
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
folds = np.load(os.path.join(CACHE, "folds.npy"))


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


POOLS = {
    "4way_seedavg": {
        "gbm": ["oof_blend.csv"], "emb": ["oof_emb_concat.csv"],
        "ft_frozen": ["ft_oof_frozen.csv"],
        "ft_fullft_avg": sorted(glob.glob(os.path.join(CACHE, "ft_oof_fullft*.csv"))),
    },
    "5way_ext": {
        "gbm": ["oof_blend.csv"], "emb": ["oof_emb_concat.csv"],
        "ft_frozen": ["ft_oof_frozen.csv"],
        "ft_fullft_avg": sorted(glob.glob(os.path.join(CACHE, "ft_oof_fullft*.csv"))),
        "ft_ext": ["ft_oof_ext.csv"],
    },
}


def greedy(zs, y, m, rounds=20):
    keys = list(zs)
    w = {k: 0.0 for k in keys}
    cur = -2
    for _ in range(rounds):
        bestk, bestr = None, -2
        for k in keys:
            cand = sum((w[j] + (1 if j == k else 0)) * zs[j] for j in keys)
            tot = sum(w.values()) + 1
            r = float(pearsonr(y[m], (cand / tot)[m]).statistic)
            if r > bestr:
                bestr, bestk = r, k
        w[bestk] += 1
        cur = bestr
    return {k: v / sum(w.values()) for k, v in w.items()}, cur


for pname, pool in POOLS.items():
    oofs = {k: zavg([f if os.path.isabs(f) else os.path.join(CACHE, f) for f in fs])
            for k, fs in pool.items()}
    print(f"== {pname}")
    nested = {}
    for iso in R.ISOFORMS:
        y = targets[R.DIRECT[iso]].values
        m = ~np.isnan(y)
        p = oofs["gbm"][iso].values  # length only
        zs_all = {}
        for k, df in oofs.items():
            v = df[iso].values.astype(float)
            z = np.full(len(v), np.nan)
            z[m] = (v[m] - v[m].mean()) / v[m].std()
            zs_all[k] = z
        # pooled honest score via Fisher-averaged per-fold Pearson
        rs = []
        wsel = []
        for f in range(5):
            trm = m & (folds != f)
            vam = m & (folds == f)
            zt = {}
            for k in zs_all:
                v = zs_all[k].copy()
                mu = v[trm].mean() if not np.isnan(v[trm]).all() else 0
                sd = v[trm].std()
                zt[k] = (v - mu) / (sd if sd > 0 else 1)
            w, _ = greedy(zt, y, trm, rounds=12)
            wsel.append(w)
            zf = sum(w[k] * zt[k] for k in w)
            rs.append(float(pearsonr(y[vam], zf[vam]).statistic))
        r_honest = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
        nested[iso] = r_honest
        print(f"  {iso}: honest nested Pearson {r_honest:.3f}")
    mac = float(np.mean([nested[i] ** 2 for i in R.ISOFORMS]))
    print(f"  macro R2 (honest): {mac:.4f}")
