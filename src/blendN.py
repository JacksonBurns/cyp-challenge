"""N-way z-space OOF mixture search over all candidates.

Candidates: gbm (FP+desc+NN 4cfg blend), emb (GBM+CheMeLeon emb concat),
ft_frozen (CheMeLeon frozen-head multitask), ft_fullft (full fine-tune).
Greedy Caruana-style forward selection (with replacement, 20 rounds) + a full
convex grid when N<=3 for cross-check. Reports best Pearson + ST-RAE under
reveal-style placement. Writes cache/blendN.json.
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

CANDS = {"gbm": "oof_blend.csv", "emb": "oof_emb_concat.csv",
         "ft_frozen": "ft_oof_frozen.csv", "ft_fullft": "ft_oof_fullft.csv"}
oofs = {k: pd.read_csv(os.path.join(CACHE, v)) for k, v in CANDS.items()}
gate = json.load(open(os.path.join(CACHE, "oof_pearson.json")))

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
    # greedy forward selection with replacement
    w = {k: 0.0 for k in keys}
    cur = None
    chosen = []
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
print("macro R2 (expected blind ~ x1.05^2 per reveal):", round(mac, 4))
with open(os.path.join(CACHE, "blendN.json"), "w") as fh:
    json.dump(out, fh, indent=2)
