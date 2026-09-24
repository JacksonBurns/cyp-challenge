"""Generalized N-way z-space greedy blend with seed-averaged families.

Families (z-mean across seeds within each):
  gbm           cache/oof_blend.csv
  emb           cache/oof_emb_concat.csv
  ft_frozen     cache/ft_oof_frozen.csv
  ft_fullft     cache/ft_oof_fullft*.csv (avg)
  ft_ext        cache/ft_oof_ext*.csv    (avg)
  ft_pre        cache/ft_oof_pre*.csv    (avg; --pretrained ft_ext runs)
  ft_dmpnn      cache/ft_oof_dmpnn*.csv  (avg)

Pools:
  ext2   = gbm, emb, ft_frozen, ft_fullft, ft_ext        (== blendN_ext)
  all7   = ext2 + ft_pre + ft_dmpnn
  all6   = ext2 + ft_dmpnn
  pre6   = ext2 + ft_pre

Usage: python src/blendN_all.py <pool>  -> cache/blendN_<pool>.json
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
}
FAMS["ft_pre"] = g("pre")
FAMS["ft_dmpnn"] = g("dmpnn")
FAMS["ft_cpmed"] = g("cpmed")
FAMS["ft_cpchm"] = g("cpchm")
POOLS = {"ext2": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext"],
         "all7": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre", "ft_dmpnn"],
         "all6": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_dmpnn"],
         "pre6": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre"],
         "cp7": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre", "ft_dmpnn", "ft_cpmed", "ft_cpchm"]}

pool = sys.argv[1] if len(sys.argv) > 1 else "ext2"
names = [n for n in POOLS[pool] if FAMS.get(n)]
oofs = {k: zavg([f if os.path.isabs(f) else os.path.join(CACHE, f) for f in FAMS[k]]) for k in names}
gate = json.load(open(os.path.join(CACHE, "oof_pearson.json")))

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
with open(os.path.join(CACHE, f"blendN_{pool}.json"), "w") as fh:
    json.dump(out, fh, indent=2)
