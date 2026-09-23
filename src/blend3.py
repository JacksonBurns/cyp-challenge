"""Per-isoform z-space mixture: GBM blend + emb-concat GBM + CheMeLeon FT OOF.

Convex grid over (w_gbm, w_emb, w_ft); reports best mixture Pearson + which
candidates earn weight. Also ST-RAE of the mixture under reveal placement
(sims use the same shape logic as emb_blend.py). Writes cache/blend3.json.
"""
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

parts = {"gbm": pd.read_csv(os.path.join(CACHE, "oof_blend.csv"))}
for tag, path in [("emb", "oof_emb_concat.csv"), ("ft", "ft_oof_frozen.csv")]:
    if os.path.exists(os.path.join(CACHE, path)):
        parts[tag] = pd.read_csv(os.path.join(CACHE, path))

gate = json.load(open(os.path.join(CACHE, "oof_pearson.json")))
STEP = 0.1
grid = [(round(a, 2), round(b, 2), round(1 - a - b, 2))
        for a in np.arange(0, 1.001, STEP) for b in np.arange(0, 1.001 - a + 1e-9, STEP)]

out = {}
for iso in R.ISOFORMS:
    y = targets[R.DIRECT[iso]].values
    m = ~np.isnan(y)
    zs = {}
    for tag, df in parts.items():
        p = df[iso].values
        if tag == "ft":  # ft csv has SMILES+fold cols; iso columns aligned to ft_data order == Xtr order
            pass
        z = np.full(len(p), np.nan)
        z[m] = (p[m] - p[m].mean()) / p[m].std()
        zs[tag] = z
    keys = list(parts)
    best = (-2, None)
    for combo in grid:
        if len(keys) == 3:
            w = dict(zip(keys, combo))
        else:
            w = {"gbm": combo[0], keys[1]: round(1 - combo[0], 2)}
        z = sum(w[k] * zs[k] for k in keys)
        r = float(pearsonr(y[m], z[m]).statistic)
        if r > best[0]:
            best = (r, w)
    singles = {k: round(float(pearsonr(y[m], zs[k][m]).statistic), 3) for k in keys}
    out[iso] = {"best_pearson": best[0], "weights": best[1], "singles": singles,
                "gate": gate[iso]}
    print(iso, f"best {best[0]:.3f}", best[1], "singles", singles, "gate", round(gate[iso], 3),
          flush=True)

macro_best = float(np.mean([out[i]["best_pearson"] ** 2 for i in R.ISOFORMS]))
print("macro R2 of mixtures:", round(macro_best, 4), "(reveal MA-R2 was 0.387 at k~1.05)")
with open(os.path.join(CACHE, "blend3.json"), "w") as fh:
    json.dump(out, fh, indent=2)
