"""Nested (honest) blend check over the generalized families in blendN_all.

Pools are defined like blendN_all.py (seed-averaged families); per-fold greedy
weight selection scored on the held-out fold, Fisher-averaged. Usage:
  python src/blend_nested_all.py <pool> [pool2 ...]   (default all pools)
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


def g(tag):
    return sorted(glob.glob(os.path.join(CACHE, f"ft_oof_{tag}*.csv")))


FAMS = {
    "gbm": [os.path.join(CACHE, "oof_blend.csv")],
    "emb": [os.path.join(CACHE, "oof_emb_concat.csv")],
    "ft_frozen": [os.path.join(CACHE, "ft_oof_frozen.csv")],
    "ft_fullft": g("fullft"),
    "ft_ext": g("ext"),
    "ft_pre": g("pre"),
    "ft_dmpnn": g("dmpnn"),
    "ft_cpmed": g("cpmed"),
    "ft_cpchm": g("cpchm"),
}
POOLS = {"ext2": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext"],
         "all7": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre", "ft_dmpnn"],
         "all6": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_dmpnn"],
         "pre6": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre"],
         "cp7": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre", "ft_dmpnn", "ft_cpmed", "ft_cpchm"]}


def greedy(zs, y, m, rounds=12):
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


for pname in (sys.argv[1:] or list(POOLS)):
    names = [n for n in POOLS[pname] if FAMS.get(n)]
    if not names:
        continue
    oofs = {k: zavg(FAMS[k]) for k in names}
    print(f"== {pname} ({', '.join(names)})")
    nested = {}
    for iso in R.ISOFORMS:
        y = targets[R.DIRECT[iso]].values
        m = ~np.isnan(y)
        zs_all = {}
        for k, df in oofs.items():
            v = df[iso].values.astype(float)
            z = np.full(len(v), np.nan)
            z[m] = (v[m] - v[m].mean()) / v[m].std()
            zs_all[k] = z
        rs = []
        for f in range(5):
            trm = m & (folds != f)
            vam = m & (folds == f)
            zt = {}
            for k in zs_all:
                v = zs_all[k].copy()
                mu = v[trm].mean() if not np.isnan(v[trm]).all() else 0
                sd = v[trm].std()
                zt[k] = (v - mu) / (sd if sd > 0 else 1)
            w, _ = greedy(zt, y, trm)
            zf = sum(w[k] * zt[k] for k in w)
            rs.append(float(pearsonr(y[vam], zf[vam]).statistic))
        r_honest = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
        nested[iso] = r_honest
        print(f"  {iso}: honest nested Pearson {r_honest:.3f}")
    mac = float(np.mean([nested[i] ** 2 for i in R.ISOFORMS]))
    print(f"  macro R2 (honest): {mac:.4f}")
    with open(os.path.join(CACHE, f"blend_nested_{pname}.json"), "w") as fh:
        json.dump(nested, fh, indent=2)
