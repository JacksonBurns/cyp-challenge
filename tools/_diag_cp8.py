"""Interim diagnostic: cp8 blend numbers + chmridge singles (will fold into NOTES sec 17)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, "src")
CACHE = "cache"
import run_regression as R

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
print("loaded")


def g(tag):
    return sorted(glob.glob(os.path.join(CACHE, f"ft_oof_{tag}*.csv")))


print("chmridge files:", g("chmridge"))
print("cpmed files:", g("cpmed"))

names = ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre",
         "ft_dmpnn", "ft_cpmed", "ft_cpchm", "ft_chmridge"]
oofs = {}
for k in names:
    if k == "gbm":
        dfs = [pd.read_csv(os.path.join(CACHE, "oof_blend.csv"))]
    elif k == "emb":
        dfs = [pd.read_csv(os.path.join(CACHE, "oof_emb_concat.csv"))]
    elif k == "ft_frozen":
        dfs = [pd.read_csv(os.path.join(CACHE, "ft_oof_frozen.csv"))]
    else:
        tag = k[3:]
        files = g(tag)
        assert files, f"missing {tag}"
        dfs = [pd.read_csv(f) for f in files]
    df = dfs[0].copy()
    for iso in R.ISOFORMS:
        if len(dfs) > 1:
            z = np.zeros(len(df))
            for d in dfs:
                p = d[iso].values.astype(float)
                z += (p - p.mean()) / p.std()
            df[iso] = z / len(dfs)
    oofs[k] = df

for iso in R.ISOFORMS:
    y = targets[R.DIRECT[iso]].values
    m = ~np.isnan(y)
    singles = {k: round(float(pearsonr(y[m], oofs[k][iso].values[m]).statistic), 3) for k in names}
    print(iso, "singles:", singles)
