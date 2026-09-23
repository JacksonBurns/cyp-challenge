"""OOF blend analysis: GBM blend vs CheMeLeon-embedding GBM (and mixtures).

Compares rank/z-weighted mixtures on scaffold OOF, per isoform:
  pearson + spearman + ST-RAE at placement (z-onto-blind-moments with
  rho=pearson, the same transform final_submit does, so ST-RAE here is a
  calibrated-blind expectation, not raw).

Gate: per-isoform OOF Pearson vs cache/oof_pearson.json (0.527/0.604/0.399/0.771).
Writes cache/emb_blend.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()

oof_blend = pd.read_csv(os.path.join(CACHE, "oof_blend.csv"))
variants = {
    "emb2048": pd.read_csv(os.path.join(CACHE, "oof_emb_emb.csv")),
    "concat": pd.read_csv(os.path.join(CACHE, "oof_emb_concat.csv")),
}

BM = {"CYP1A2": (4.412, 1.553), "CYP2C9": (4.830, 1.101),
      "CYP2D6": (3.107, 1.599), "CYP3A4": (4.880, 1.272)}

# for ST-RAE eval of calibrated placement we need truth-vs-bounds as always;
# placement of OOF preds (z->moments with rho=pearson) is monotone, so raw OOF
# ST-RAE vs placed ST-RAE differ; compute BOTH (raw = conservative, placed = pipeline view)
weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

out = {}
for vname, oofv in variants.items():
    for iso in R.ISOFORMS:
        y = targets[R.DIRECT[iso]].values
        m = ~np.isnan(y)
        lo, hi = bounds[f"{iso}_lo"].values, bounds[f"{iso}_hi"].values
        b = oof_blend[iso].values
        e = oofv[iso].values
        zb = np.full(len(b), np.nan)
        ze = np.full(len(e), np.nan)
        zb[m] = (b[m] - b[m].mean()) / b[m].std()
        ze[m] = (e[m] - e[m].mean()) / e[m].std()
        rows = []
        for w in weights:
            z = w * zb + (1 - w) * ze
            mm = m & ~np.isnan(z)
            yy = y[mm]
            z2 = z[mm]
            placed = BM[iso][0] + pearsonr(yy, z2).statistic * BM[iso][1] * z2
            rows.append({
                "w_gbm": w,
                "pearson": float(pearsonr(yy, z2).statistic),
                "spearman": float(spearmanr(yy, z2).statistic),
                "st_rae_raw": float(R.soft_rae(yy, z2, lo[mm], hi[mm])),
                "st_rae_placed": float(R.soft_rae(yy, np.clip(placed, 1.0, None), lo[mm], hi[mm])),
            })
        out.setdefault(vname, {})[iso] = rows

print(f"{'variant':10s} {'iso':7s} {'w':>4s} {'pearson':>8s} {'spearman':>8s} {'rae_raw':>8s} {'rae_plc':>8s}")
best = {}
for vname in out:
    for iso in R.ISOFORMS:
        rows = out[vname][iso]
        br = max(rows, key=lambda r: r["pearson"])
        best[f"{vname}|{iso}"] = br
        for r in rows[::2]:
            mark = " <<<" if r is br else ""
            print(f"{vname:10s} {iso:7s} {r['w_gbm']:4.1f} {r['pearson']:8.3f} {r['spearman']:8.3f} "
                  f"{r['st_rae_raw']:8.3f} {r['st_rae_placed']:8.3f}{mark}")

gate = json.load(open(os.path.join(CACHE, "oof_pearson.json")))
print("\nGATE per-isoform:", {k: round(v, 3) for k, v in gate.items()})
for iso in R.ISOFORMS:
    cands = {k: v["pearson"] for k, v in best.items() if k.endswith(iso)}
    print(iso, {k.split('|')[0]: round(v, 3) for k, v in cands.items()}, "gate", round(gate[iso], 3))

with open(os.path.join(CACHE, "emb_blend.json"), "w") as fh:
    json.dump({"grid": out, "best": best}, fh, indent=2)
