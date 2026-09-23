"""Small param sweep on the periso config, then write final test predictions."""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
R.S_ALL, R.STE_ALL = S, Ste
Xbase = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
groups = R.scaffold_groups(Xtr["SMILES"].tolist())
folds = R.make_folds(groups)

SWEEP = {
    "A_current": {},
    "B_shallow": dict(num_leaves=31, learning_rate=0.02, n_estimators=2500, colsample_bytree=0.35, min_child_samples=25),
    "C_wide": dict(num_leaves=127, learning_rate=0.05, n_estimators=800, colsample_bytree=0.5, min_child_samples=10),
    "D_slow_deep": dict(num_leaves=63, learning_rate=0.015, n_estimators=4000, colsample_bytree=0.45, min_child_samples=15),
}
results = {}
oofs = {}
for name, params in SWEEP.items():
    n, s, oof = R.cv_config(name, Xbase, None, targets, bounds, folds, R.ISOFORMS, None, params)
    results[name] = s
    oofs[name] = oof

with open(os.path.join(CACHE, "cv_sweep.json"), "w") as fh:
    json.dump(results, fh, indent=2)

best = max(results, key=lambda k: -results[k]["MA_st_rae"])
print("BEST:", best)
# also check average-of-all blend
blendp = {
    iso: np.nanmean(np.vstack([oofs[n][iso] for n in SWEEP]), axis=0) for iso in R.ISOFORMS
}
ys = {iso: targets[R.DIRECT[iso]].values for iso in R.ISOFORMS}
blend = {}
for iso in R.ISOFORMS:
    preds = blendp[iso]
    blend[iso] = float(R.soft_rae(ys[iso][~np.isnan(ys[iso])], preds[~np.isnan(ys[iso])],
                                   bounds[f"{iso}_lo"].values[~np.isnan(ys[iso])],
                                   bounds[f"{iso}_hi"].values[~np.isnan(ys[iso])]))
blend["MA_st_rae"] = float(np.mean([blend[i] for i in R.ISOFORMS]))
print("BLEND:", json.dumps(blend))
with open(os.path.join(CACHE, "cv_sweep.json"), "w") as fh:
    json.dump({"results": results, "blend": blend, "params": SWEEP}, fh, indent=2)
pd.DataFrame({f"{iso}": np.nanmean(np.vstack([oofs[n][iso] for n in SWEEP]), axis=0) for iso in R.ISOFORMS}).to_csv(os.path.join(CACHE, "oof_blend.csv"), index=False)

# OOF Pearson of the blend, per isoform -> calibration input
poof = {}
for iso in R.ISOFORMS:
    y = targets[R.DIRECT[iso]].values
    m = ~np.isnan(y)
    from scipy.stats import pearsonr
    poof[iso] = float(pearsonr(y[m], blendp[iso][m]).statistic)
with open(os.path.join(CACHE, "oof_pearson.json"), "w") as fh:
    json.dump(poof, fh, indent=2)
print("oof pearson:", poof)
