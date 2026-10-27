"""Shuffle ft_oof_adme2.csv per isoform -> ft_oof_adme2_shuf.csv (control)."""
import numpy as np
import pandas as pd
import os

CACHE = "/home/jackson/cyp-challenge/cache"
d = pd.read_csv(os.path.join(CACHE, "ft_oof_adme2.csv"))
rng = np.random.default_rng(2)
for iso in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
    v = np.array(d[iso].values)
    ok = ~np.isnan(v)
    v[ok] = rng.permutation(v[ok])
    d[iso] = v
d.to_csv(os.path.join(CACHE, "ft_oof_adme2_shuf.csv"), index=False)
print("shuffled adme2 written")
