"""Build averaged seed-family members for pool gates:
  ft_oof_adme2w002_avg.csv  (3 seeds of AUX_W=0.02)  + shuf control
  ft_oof_adme2w01_avg.csv   (4 seeds of AUX_W=0.1)   + shuf control
  ft_oof_adme2band_avg.csv  (7 seeds, w0.02+w0.1)    + shuf control
Test preds: ft_test_preds_<same tags>."""
import os

import numpy as np
import pandas as pd

CACHE = "/home/jackson/cyp-challenge/cache"
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

FAM = {
    "adme2w002_avg": ["adme2w002", "adme2w002_1", "adme2w002_2", "adme2w002_3"],
    "adme2w01_avg": ["adme2w01", "adme2w1_1", "adme2w1_2", "adme2w1_3", "adme2w1_4"],
    "adme2band_avg": ["adme2w002", "adme2w002_1", "adme2w002_2", "adme2w002_3",
                      "adme2w002_4", "adme2w002_5", "adme2w002_6", "adme2w002_7",
                      "adme2w002_8",
                      "adme2w01", "adme2w1_1", "adme2w1_2", "adme2w1_3", "adme2w1_4"],
}

rng = np.random.default_rng(7)
for tag, seeds in FAM.items():
    dfs = [pd.read_csv(os.path.join(CACHE, f"ft_oof_{s}.csv")) for s in seeds]
    avg = dfs[0].copy()
    for iso in ISO:
        avg[iso] = np.mean([d[iso].values.astype(float) for d in dfs], axis=0)
    avg.to_csv(os.path.join(CACHE, f"ft_oof_{tag}.csv"), index=False)
    sh = avg.copy()
    for iso in ISO:
        v = np.array(sh[iso].values)
        ok = ~np.isnan(v)
        v[ok] = rng.permutation(v[ok])
        sh[iso] = v
    sh.to_csv(os.path.join(CACHE, f"ft_oof_{tag}_shuf.csv"), index=False)
    # test preds
    tdfs = [pd.read_csv(os.path.join(CACHE, f"ft_test_preds_{s}.csv")) for s in seeds]
    tavg = tdfs[0].copy()
    for iso in ISO:
        tavg[iso] = np.mean([d[iso].values.astype(float) for d in tdfs], axis=0)
    tavg.to_csv(os.path.join(CACHE, f"ft_test_preds_{tag}.csv"), index=False)
    print(tag, len(seeds), "seeds; oof+shuf+test written")
