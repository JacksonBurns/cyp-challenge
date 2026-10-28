"""Standalone macro softRAE for any ft_oof_*.csv in cache (matching ft_adme2 eval)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402

smi = H.Xtr["SMILES"].values
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def softrae(iso, P):
    y = H.targets[f"{iso}_pIC50_direct_inhibition"].values
    lo = H.bounds[f"{iso}_lo"].values
    hi = H.bounds[f"{iso}_hi"].values
    m = np.isfinite(y)
    err = np.clip(P[m] - hi[m], 0, None) + np.clip(lo[m] - P[m], 0, None)
    return float(np.mean(err))


for tag in sys.argv[1:]:
    d = pd.read_csv(f"/home/jackson/cyp-challenge/cache/ft_oof_{tag}.csv")
    ms = []
    for iso in ISO:
        P = d[iso].values
        ms.append(softrae(iso, P))
    print(tag, "macro softRAE %.4f" % np.mean(ms), [round(x, 4) for x in ms])
