"""Average extw005 OOF seeds -> avg; shuffled control -> avg_shuf."""
import numpy as np
import pandas as pd
import os

CACHE = "/home/jackson/cyp-challenge/cache"
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
SEEDFILES = {0: "ft_oof_extw005.csv", 1: "ft_oof_extw005_s1.csv", 2: "ft_oof_extw005_s2.csv"}
dfs = []
for s, fn in SEEDFILES.items():
    p_ = os.path.join(CACHE, fn)
    if os.path.exists(p_):
        dfs.append(pd.read_csv(p_))
print("seeds:", len(dfs))
base = dfs[0]
for iso in ISO:
    cols = [d[iso].values for d in dfs]
    base[iso] = np.nanmean(np.vstack(cols), axis=0)
base.to_csv(os.path.join(CACHE, "ft_oof_extw005_avg.csv"), index=False)
rng = np.random.default_rng(7)
sh = base.copy()
for iso in ISO:
    v = np.array(sh[iso].values)
    ok = ~np.isnan(v)
    v[ok] = rng.permutation(v[ok])
    sh[iso] = v
sh.to_csv(os.path.join(CACHE, "ft_oof_extw005_avg_shuf.csv"), index=False)
print("avg + shuf written")
