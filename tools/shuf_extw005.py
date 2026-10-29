"""Average extw005 OOF seeds -> avg; shuffled control -> avg_shuf."""
import numpy as np
import pandas as pd
import os

CACHE = "/home/jackson/cyp-challenge/cache"
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
dfs = []
for s in [0, 1, 2]:
    p = os.path.join(CACHE, f"ft_oof_extw005_{s}.csv")
    if os.path.exists(p):
        dfs.append(pd.read_csv(p))
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
