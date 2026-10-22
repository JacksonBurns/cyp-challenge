"""Track B: does adding CYP2D6 docking features lift the TDI classifier?

Same machinery as tdi_phase1_v2_classifier.py, but adds the 14 docking
features (cache/dockfeat_cyp2d6_full.csv) to each base feature set.
Compares with/without docking to isolate the docking contribution.

Output: cache/tdi_dockfeat_eval.json
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
from run_regression import make_folds, scaffold_groups  # noqa: E402
from tdi_phase1_v2_classifier import (  # noqa: E402
    TDI_ISO, GATE_THR, SHIFT_THR, K, PROD_GRID,
    oof_classifier, nested_mcc, tuned_mcc,
)

warnings.filterwarnings("ignore")

# --- Load base features (same as v2 classifier) ---
print("Loading features...", flush=True)
Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"))
Xtr = Xtr.drop_duplicates("SMILES").reset_index(drop=True)
smi = Xtr["SMILES"].values
base = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)

cpmed = pd.read_parquet(os.path.join(CACHE, "emb_cpmed_all.parquet")).set_index("SMILES")
cpmed_mat = cpmed.loc[smi].drop(columns=["SMILES"], errors="ignore").values.astype(np.float32)
if cpmed_mat.ndim == 1:
    cpmed_mat = cpmed_mat.reshape(1, -1)

umtm_files = sorted(
    [os.path.join(CACHE, f) for f in os.listdir(CACHE)
     if f.startswith("umtm_oof_") and f.endswith(".csv")
     and "_ext" not in f and "_primary" not in f]
)
umtm = pd.concat([pd.read_csv(f) for f in umtm_files], ignore_index=True)
umtm_cols = [c for c in umtm.columns if c not in ("SMILES", "fold", "fold_tdi")]
umtm_avg = umtm.groupby("SMILES")[umtm_cols].mean().reindex(smi).values.astype(np.float32)
umtm_avg = np.nan_to_num(umtm_avg)

# --- Load docking features ---
dock = pd.read_csv(os.path.join(CACHE, "dockfeat_cyp2d6_full.csv")).set_index("SMILES")
DOCK_FEATS = [c for c in dock.columns if c not in ("dock_fail",)]
dock_mat = dock.reindex(smi)[DOCK_FEATS].values.astype(np.float32)
# dock_fail rows: features are 0 -> NaN so the tree can split on "no dock"
dock_mat[dock.reindex(smi)["dock_fail"].values.astype(bool), :] = np.nan
n_docked = int((~dock.reindex(smi)["dock_fail"].values.astype(bool)).sum())
print(f"  docking: {n_docked}/{len(smi)} train SMILES docked, {len(DOCK_FEATS)} features", flush=True)

# --- Feature sets: with and without docking ---
feat_sets = {
    # baselines (reproduced for comparison)
    "umtm": umtm_avg,
    "base_umtm": np.hstack([base, umtm_avg]),
    "cpmed_umtm": np.hstack([cpmed_mat, umtm_avg]),
    # + docking
    "umtm+dock": np.hstack([umtm_avg, dock_mat]),
    "base_umtm+dock": np.hstack([base, umtm_avg, dock_mat]),
    "cpmed_umtm+dock": np.hstack([cpmed_mat, umtm_avg, dock_mat]),
}
for name, mat in feat_sets.items():
    print(f"  '{name}': {mat.shape}", flush=True)

# --- Folds + TDI data (same as v2) ---
groups = scaffold_groups(smi.tolist())
folds = make_folds(groups, seed=7)
tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES").set_index("SMILES")

clf_params = dict(
    n_estimators=800, learning_rate=0.02, num_leaves=31,
    min_child_samples=20, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.15, reg_lambda=10.0, reg_alpha=5.0,
    n_jobs=-1, random_state=42, verbosity=-1
)

report = {}
for feat_name, X in feat_sets.items():
    report[feat_name] = {}
    for iso in TDI_ISO:
        short = iso[3:]
        ycol = f"{iso}_is_TDI"
        tdic_col = f"CYP{short}_pIC50_TDI_condition"
        dir_col = f"CYP{short}_pIC50_direct_inhibition"

        tdic = pd.to_numeric(tdi[tdic_col].reindex(smi), errors="coerce").values
        gate_mask = ~np.isnan(tdic)
        gate_y = (tdic > GATE_THR).astype(float)
        gate_idx = np.where(gate_mask)[0]

        dirv = pd.to_numeric(tdi[dir_col].reindex(smi), errors="coerce").values
        both_mask = ~np.isnan(tdic) & ~np.isnan(dirv)
        delta = (tdic - dirv)
        shift_y = (delta > SHIFT_THR).astype(float)
        shift_idx = np.where(both_mask)[0]

        y_lab = pd.to_numeric(tdi[ycol].reindex(smi), errors="coerce").values
        lab_mask = ~np.isnan(y_lab)
        y = np.zeros(len(smi))
        y[lab_mask] = y_lab[lab_mask]

        gate_oof = oof_classifier(X, gate_y, folds, gate_idx,
                                  lambda: LGBMClassifier(**clf_params))
        shift_oof = oof_classifier(X, shift_y, folds, shift_idx,
                                   lambda: LGBMClassifier(**clf_params))

        eval_mask = lab_mask & both_mask
        eval_idx = np.where(eval_mask)[0]
        y_ev = y[eval_idx]
        f_ev = folds[eval_idx]
        g_ev = np.nan_to_num(gate_oof[eval_idx], nan=0.5)
        s_ev = np.nan_to_num(shift_oof[eval_idx], nan=0.5)

        prod = g_ev * s_ev
        p_nested = nested_mcc(prod, y_ev, f_ev, PROD_GRID)
        p_tuned = tuned_mcc(prod, y_ev, PROD_GRID)

        report[feat_name][iso] = {
            "n": int(len(y_ev)),
            "product_nested": float(p_nested),
            "product_tuned": float(p_tuned),
        }
        print(f"  [{feat_name}] {iso}: product nested={p_nested:.4f} tuned={p_tuned:.4f}", flush=True)

# --- Summary: docking delta per feature set ---
print("\n" + "=" * 70)
print("TDI DOCKFEAT EVAL - product nested MCC (macro = mean of 2D6, 3A4)")
print("=" * 70)
print(f"  {'feature set':<20} {'2D6':>8} {'3A4':>8} {'macro':>8}  {'vs no-dock':>12}")
summary = {}
for feat_name in feat_sets:
    v2d6 = report[feat_name]["CYP2D6"]["product_nested"]
    v3a4 = report[feat_name]["CYP3A4"]["product_nested"]
    macro = (v2d6 + v3a4) / 2
    summary[feat_name] = {"CYP2D6": v2d6, "CYP3A4": v3a4, "macro": macro}
    base_name = feat_name.replace("+dock", "")
    delta = ""
    if "+dock" in feat_name and base_name in summary:
        delta = f"{macro - summary[base_name]['macro']:+.4f}"
    print(f"  {feat_name:<20} {v2d6:>8.4f} {v3a4:>8.4f} {macro:>8.4f}  {delta:>12}")

best = max(summary.items(), key=lambda kv: kv[1]["macro"])
print(f"\n  BEST: {best[0]} (macro {best[1]['macro']:.4f})")
print(f"  bar: macro 0.3448 (2D6 0.219, 3A4 0.470)")
print(f"  leaderboard best: 0.5346")

with open(os.path.join(CACHE, "tdi_dockfeat_eval.json"), "w") as f:
    json.dump({"per_set": summary, "best": best[0]}, f, indent=2)
print(f"\nSaved {os.path.join(CACHE, 'tdi_dockfeat_eval.json')}")
print("DONE")
