"""N-way z-space mixture with seed-averaged CheMeLeon full-FT candidate.

Same greedy Caruana search as blendN.py, but ft_fullft is replaced by the
mean (z-space) of the two seed runs (fullft + fullft_s1). Hypothesis: seed
averaging reduces NN variance and lifts both standalone and blend Pearson.
Writes cache/blendN_seedavg.json.
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

CANDS = {"gbm": ["oof_blend.csv"], "emb": ["oof_emb_concat.csv"],
         "ft_frozen": ["ft_oof_frozen.csv"],
         "ft_fullft_avg": ["ft_oof_fullft.csv", "ft_oof_fullft_s1.csv"]}
gate = json.load(open(os.path.join(CACHE, "oof_pearson.json")))

oofs = {}
for k, files in CANDS.items():
    dfs = [pd.read_csv(os.path.join(CACHE, f)) for f in files]
    df = dfs[0].copy()
    for iso in R.ISOFORMS:
        if len(dfs) > 1:
            # z-average each seed column (same row order guaranteed: all derived
            # from ft_data.csv fold export), then it gets re-z'd downstream.
            z = np.zeros(len(df))
            for d in dfs:
                p = d[iso].values
                z += (p - p.mean()) / p.std()
            df[iso] = z / len(dfs)
    oofs[k] = df

# sanity: seed runs row-aligned?
d0 = pd.read_csv(os.path.join(CACHE, "ft_oof_fullft.csv"))
d1 = pd.read_csv(os.path.join(CACHE, "ft_oof_fullft_s1.csv"))
assert (d0["SMILES"].values == d1["SMILES"].values).all(), "seed OOF row order mismatch"

out = {}
for iso in R.ISOFORMS:
    y = targets[R.DIRECT[iso]].values
    m = ~np.isnan(y)
    zs = {}
    for k, df in oofs.items():
        p = df[iso].values
        z = np.full(len(p), np.nan)
        z[m] = (p[m] - p[m].mean()) / p[m].std()
        zs[k] = z
    keys = list(zs)
    w = {k: 0.0 for k in keys}
    chosen = []
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
        chosen.append(bestk)
        cur = bestr
    w_norm = {k: round(v / sum(w.values()), 2) for k, v in w.items() if v > 0}
    singles = {k: round(float(pearsonr(y[m], zs[k][m]).statistic), 3) for k in keys}
    out[iso] = {"pearson": cur, "weights": w_norm, "singles": singles, "gate": gate[iso]}
    print(iso, f"{cur:.3f}", w_norm, "singles", singles, "gate", round(gate[iso], 3), flush=True)

mac = float(np.mean([out[i]["pearson"] ** 2 for i in R.ISOFORMS]))
print("macro R2:", round(mac, 4))
with open(os.path.join(CACHE, "blendN_seedavg.json"), "w") as fh:
    json.dump(out, fh, indent=2)
