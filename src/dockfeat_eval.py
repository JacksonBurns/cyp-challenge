"""Honest nested eval of the CYP2D6 docking feature member (Avenue 3).

Docking features (Vina into 2F9Q CYP2D6 heme pocket) are a LABEL-FREE
prior-knowledge representation - same class as the LLM pharmacophore member.
The hypothesis: 2D6 selectivity is driven by the specific heme-binding pose,
so docking-derived interaction fingerprints should carry 2D6-specific signal
that vacuum 3D (SchNet, dead) and 2D descriptors (saturated) both miss.

Anti-leak protocol (same as llm_pharmaco_eval / mordred_probe):
  - features are docking-derived (no target in X), so no direct leakage
  - per fold: impute + z-score with TRAIN-fold stats; fit Ridge on train;
    predict val; score Pearson on val
  - nested = Fisher-z avg per isoform; macro = mean(nested^2)

Learner: per-isoform Ridge (alpha=1.0). 14 docking features, ~1200-1500
labeled points per isoform -> well-conditioned.

Output: cache/ft_oof_dockfeat.csv (blend-ready), cache/dockfeat_eval.json
Then:  new_member_pool_test.py dockfeat  for the pool gate.

Run:  ~/miniforge3/envs/cyp/bin/python src/dockfeat_eval.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
folds = ft["fold"].values
print(f"ft_data rows: {len(ft)}")

dock = pd.read_csv(os.path.join(CACHE, "dockfeat_cyp2d6_full.csv"))
print(f"dockfeat rows: {len(dock)}  (dock_fail: {int(dock['dock_fail'].sum())})")

# Merge on SMILES (dock was run on unique SMILES from X_train)
ft2 = ft.merge(dock, on="SMILES", how="left")
n_nan = ft2["vina_score"].isna().sum()
print(f"merged rows: {len(ft2)}  (unmatched SMILES: {n_nan})")

# Feature columns (exclude SMILES, dock_fail is a flag not a feature)
FEAT_COLS = [c for c in dock.columns if c not in ("SMILES", "dock_fail")]
print(f"features ({len(FEAT_COLS)}): {FEAT_COLS}")

X_all = ft2[FEAT_COLS].values.astype(float)
# dock_fail rows have all features 0 - mark them NaN so impute handles them
X_all[ft2["dock_fail"].values.astype(bool), :] = np.nan
# also NaN out unmatched rows
X_all[ft2["vina_score"].isna().values, :] = np.nan

# ---------------------------------------------------------------------------
# Anti-leak impute + scale (same recipe as llm_pharmaco_eval)
# ---------------------------------------------------------------------------
def impute_scale(Xtr, Xva):
    """Per-column median impute + z-score using TRAIN-fold stats only."""
    Xtr_i = Xtr.copy()
    Xva_i = Xva.copy()
    for j in range(Xtr.shape[1]):
        c = Xtr_i[:, j]
        med = np.nanmedian(c)
        med = 0.0 if np.isnan(med) else med
        Xtr_i[:, j] = np.where(np.isnan(c), med, c)
        Xva_i[:, j] = np.where(np.isnan(Xva_i[:, j]), med, Xva_i[:, j])
    mu = Xtr_i.mean(axis=0)
    sd = Xtr_i.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Xtr_i - mu) / sd, (Xva_i - mu) / sd


def fit_predict(Xtr, ytr, Xva, alpha=1.0):
    m = ~np.isnan(ytr)
    rg = Ridge(alpha=alpha)
    rg.fit(Xtr[m], ytr[m])
    return rg.predict(Xva), rg


# ---------------------------------------------------------------------------
# Honest nested eval
# ---------------------------------------------------------------------------
print("\n=== Docking feature member: standalone honest nested ===")
oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
full_models = {}

for iso in ISO:
    y = ft[iso].values.astype(float)
    m = ~np.isnan(y)
    rs = []
    for fd in range(5):
        trm = m & (folds != fd)
        vam = m & (folds == fd)
        a, b = impute_scale(X_all[trm], X_all[vam])
        pred, _ = fit_predict(a, y[trm], b)
        oof.loc[vam, iso] = pred
        r = pearsonr(y[vam], pred).statistic
        if not np.isnan(r):
            rs.append(r)
    nested = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
    # full-train model for test preds
    mfull = m
    a, b = impute_scale(X_all[mfull], X_all)
    pred_full, model = fit_predict(a, y[mfull], b)
    full_models[iso] = model
    oof[iso] = oof[iso].fillna(0)
    print(f"  {iso}: nested={nested:.4f} R2={nested**2:.4f}  "
          f"(n_labeled={m.sum()})")

# recompute nested per isoform from oof (clean)
nested_per = {}
for iso in ISO:
    y = ft[iso].values.astype(float)
    m = ~np.isnan(y)
    rs = []
    for fd in range(5):
        trm = m & (folds != fd)
        vam = m & (folds == fd)
        r = pearsonr(y[vam], oof[iso].values[vam]).statistic
        if not np.isnan(r):
            rs.append(r)
    nested_per[iso] = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
macro = float(np.mean([nested_per[i] ** 2 for i in ISO]))
print(f"  MACRO nested R2 (dockfeat alone): {macro:.4f}")

# Save OOF (blend-ready format for new_member_pool_test.py)
oof_out = oof.copy()
oof_out.insert(0, "SMILES", ft["SMILES"])
oof_out["fold"] = folds
oof_out.to_csv(os.path.join(CACHE, "ft_oof_dockfeat.csv"), index=False)
print(f"saved cache/ft_oof_dockfeat.csv")

# Test predictions
test = pd.read_csv(os.path.join(CACHE, "..", "data",
                                "cyp-challenge-TEST-BLINDED.csv"))
# Dock test SMILES (reuse the same pipeline - but we don't have test docking
# features yet. For now, save the model and note that test preds need a
# separate docking run on the 750 test molecules.)
# For the pool test, only the OOF matters.
print(f"note: test preds need a separate docking run on 750 test SMILES")

# ---------------------------------------------------------------------------
# Save eval summary
# ---------------------------------------------------------------------------
res = {
    "dockfeat_standalone": {"nested": nested_per, "macro": macro},
    "n_features": len(FEAT_COLS),
    "feature_names": FEAT_COLS,
    "dock_fail_count": int(dock["dock_fail"].sum()),
    "n_docked": len(dock) - int(dock["dock_fail"].sum()),
}
with open(os.path.join(CACHE, "dockfeat_eval.json"), "w") as fh:
    json.dump(res, fh, indent=2)
print(f"saved cache/dockfeat_eval.json")
print("DONE")
