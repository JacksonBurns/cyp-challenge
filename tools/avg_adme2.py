"""Average ft_adme2 seed OOF + test preds into one 4-seed family member."""
import os

import numpy as np
import pandas as pd

CACHE = "/home/jackson/cyp-challenge/cache"
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

files = ["ft_oof_adme2.csv", "ft_oof_adme2_s1.csv",
         "ft_oof_adme2_s2.csv", "ft_oof_adme2_s3.csv"]
dfs = [pd.read_csv(os.path.join(CACHE, f)) for f in files]
avg = dfs[0].copy()
for iso in ISO:
    avg[iso] = np.mean([d[iso].values.astype(float) for d in dfs], axis=0)
avg.to_csv(os.path.join(CACHE, "ft_oof_adme2_avg.csv"), index=False)
print("oof avg written", avg.shape)

tf = ["ft_test_preds_adme2.csv"] + [f"ft_test_preds_adme2_s{s}.csv" for s in (1, 2, 3)]
tdfs = [pd.read_csv(os.path.join(CACHE, f)) for f in tf]
tavg = tdfs[0].copy()
for iso in ISO:
    tavg[iso] = np.mean([d[iso].values.astype(float) for d in tdfs], axis=0)
tavg.to_csv(os.path.join(CACHE, "ft_test_preds_adme2_avg.csv"), index=False)
print("test avg written", tavg.shape)
