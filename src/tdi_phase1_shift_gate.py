"""Phase 1 T1: FRESH shift/gate regressors for the conjunction-decomposed TDI.

Builds per isoform:
  - shift_hat: LightGBM regressor predicting delta = pi_TDI - pi_dir directly
  - gate_hat:  LightGBM regressor predicting pi_TDI (TDI-condition pIC50)
Then forms the product score and evaluates nested-honest MCC.

Features: base X_train (FP+desc) + cpmed embeddings + UMTM OOF heads.
Folds: scaffold seed 7 (same as all incumbents).
CPU-only (LightGBM), no GPU.

Gate: nested macro > 0.3448, 2D6 > 0.219, 3A4 > 0.470.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
from run_regression import make_folds, scaffold_groups  # noqa: E402

warnings.filterwarnings("ignore")

TDI_ISO = ["CYP2D6", "CYP3A4"]
GATE_THR = 4.301
SHIFT_THR = 0.301
K = 5

# Scale-appropriate threshold grids (same as Phase 0)
GATE_GRID = np.linspace(3.0, 5.5, 101)
SHIFT_GRID = np.linspace(-1.0, 1.5, 101)
PROD_GRID = np.linspace(0.05, 0.85, 81)


def mcc_at(p, y, t):
    pr = (p >= t).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum())
    fp = int(((pr == 1) & (y == 0)).sum())
    tn = int(((pr == 0) & (y == 0)).sum())
    fn = int(((pr == 0) & (y == 1)).sum())
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def nested_mcc(p, y, folds, grid):
    nested_preds = np.zeros(len(y), dtype=int)
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        if tr.sum() == 0 or va.sum() == 0:
            continue
        t = max(grid, key=lambda t: mcc_at(p[tr], y[tr], t))
        nested_preds[va] = (p[va] >= t).astype(int)
    tp = int(((nested_preds == 1) & (y == 1)).sum())
    fp = int(((nested_preds == 1) & (y == 0)).sum())
    tn = int(((nested_preds == 0) & (y == 0)).sum())
    fn = int(((nested_preds == 0) & (y == 1)).sum())
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def tuned_mcc(p, y, grid):
    return max(mcc_at(p, y, t) for t in grid)


def sigprod(gate_pred, shift_pred, k):
    pg = 1 / (1 + np.exp(-k * (gate_pred - GATE_THR)))
    ps = 1 / (1 + np.exp(-k * (shift_pred - SHIFT_THR)))
    return pg * ps


def oof_regressor(X, y, folds, cand_idx, model_fn):
    """Out-of-fold regression: train on folds != f, predict fold f.
    Returns OOF predictions for all cand_idx rows (NaN elsewhere)."""
    n = len(X)
    oof = np.full(n, np.nan)
    for f in range(K):
        tr = cand_idx[folds[cand_idx] != f]
        va = cand_idx[folds[cand_idx] == f]
        if len(tr) == 0 or len(va) == 0:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    return oof


def main():
    # --- Load features ---
    print("Loading features...", flush=True)
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"))
    Xtr = Xtr.drop_duplicates("SMILES").reset_index(drop=True)
    smi = Xtr["SMILES"].values
    base = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    print(f"  base: {base.shape}")

    # cpmed embeddings (600-d, frozen chemprop_medium)
    cpmed = pd.read_parquet(os.path.join(CACHE, "emb_cpmed_all.parquet")).set_index("SMILES")
    cpmed_cols = [c for c in cpmed.columns if c != "SMILES"]
    cpmed_mat = cpmed.loc[smi, cpmed_cols].values.astype(np.float32)
    print(f"  cpmed: {cpmed_mat.shape}")

    # UMTM OOF heads (averaged across all seeds/lineages)
    umtm_files = sorted(
        [os.path.join(CACHE, f) for f in os.listdir(CACHE)
         if f.startswith("umtm_oof_") and f.endswith(".csv")
         and "_ext" not in f and "_primary" not in f]
    )
    umtm = pd.concat([pd.read_csv(f) for f in umtm_files], ignore_index=True)
    umtm_cols = [c for c in umtm.columns if c not in ("SMILES", "fold", "fold_tdi")]
    umtm_avg = umtm.groupby("SMILES")[umtm_cols].mean().reindex(smi).values.astype(np.float32)
    # Fill any NaN (shouldn't be any, all 6145 SMILES are in UMTM)
    umtm_avg = np.nan_to_num(umtm_avg)
    print(f"  umtm heads: {umtm_avg.shape}")

    # Combined feature matrix
    X = np.hstack([base, cpmed_mat, umtm_avg])
    print(f"  combined X: {X.shape}")

    # --- Folds ---
    groups = scaffold_groups(smi.tolist())
    folds = make_folds(groups, seed=7)
    print(f"  folds: {np.bincount(folds)}")

    # --- TDI data ---
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES").set_index("SMILES")

    report = {}
    for iso in TDI_ISO:
        short = iso[3:]
        ycol = f"{iso}_is_TDI"
        tdic_col = f"CYP{short}_pIC50_TDI_condition"
        dir_col = f"CYP{short}_pIC50_direct_inhibition"

        print(f"\n{'='*60}")
        print(f"  CYP{short}")
        print(f"{'='*60}")

        # --- Targets ---
        # Gate target: pi_TDI (where TDI arm exists)
        tdic = tdi[tdic_col].reindex(smi).values.astype(float)
        gate_mask = ~np.isnan(tdic)
        gate_y = tdic[gate_mask]
        gate_idx = np.where(gate_mask)[0]
        print(f"  gate target: n={len(gate_y)}, mean={gate_y.mean():.3f}, std={gate_y.std():.3f}")

        # Shift target: delta = pi_TDI - pi_dir (where both arms exist)
        dirv = tdi[dir_col].reindex(smi).values.astype(float)
        both_mask = ~np.isnan(tdic) & ~np.isnan(dirv)
        shift_y = (tdic - dirv)[both_mask]
        shift_idx = np.where(both_mask)[0]
        print(f"  shift target: n={len(shift_y)}, mean={shift_y.mean():.3f}, std={shift_y.std():.3f}")

        # Labels
        y_lab = pd.to_numeric(tdi[ycol].reindex(smi), errors="coerce").values
        lab_mask = ~np.isnan(y_lab)
        y = np.zeros(len(smi))
        y[lab_mask] = y_lab[lab_mask]

        # --- Train shift regressor (OOF) ---
        print(f"  Training shift regressor (OOF)...", flush=True)
        shift_params = dict(
            n_estimators=1500, learning_rate=0.02, num_leaves=63,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.3, reg_lambda=5.0, reg_alpha=1.0,
            n_jobs=-1, random_state=42, verbosity=-1
        )
        shift_oof = oof_regressor(
            X, y, folds, shift_idx,
            lambda: LGBMRegressor(**shift_params)
        )
        shift_pred = shift_oof[both_mask]
        print(f"  shift OOF: n={len(shift_pred)}")

        # --- Train gate regressor (OOF) ---
        print(f"  Training gate regressor (OOF)...", flush=True)
        gate_params = dict(
            n_estimators=1500, learning_rate=0.02, num_leaves=63,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.3, reg_lambda=5.0, reg_alpha=1.0,
            n_jobs=-1, random_state=42, verbosity=-1
        )
        gate_oof = oof_regressor(
            X, y, folds, gate_idx,
            lambda: LGBMRegressor(**gate_params)
        )
        gate_pred = gate_oof[gate_mask]
        print(f"  gate OOF: n={len(gate_pred)}")

        # --- Evaluate on labeled rows ---
        # For the conjunction score, we need both gate and shift predictions
        # on the same rows. Use the intersection: labeled AND both-arms.
        eval_mask = lab_mask & both_mask
        eval_idx = np.where(eval_mask)[0]
        y_ev = y[eval_idx]
        f_ev = folds[eval_idx]
        g_ev = gate_oof[eval_idx]
        s_ev = shift_oof[eval_idx]

        # Also evaluate on all labeled rows (gate available for more rows)
        eval_mask_all = lab_mask & gate_mask
        eval_idx_all = np.where(eval_mask_all)[0]
        y_ev_all = y[eval_idx_all]
        f_ev_all = folds[eval_idx_all]
        g_ev_all = gate_oof[eval_idx_all]
        # For shift on rows without direct arm, use the shift OOF (NaN -> fill with median)
        s_ev_all = shift_oof[eval_idx_all].copy()
        s_median = np.nanmedian(shift_oof[both_mask])
        s_ev_all[np.isnan(s_ev_all)] = s_median

        print(f"\n  Eval set (both-arms + labeled): n={len(y_ev)}, pos={y_ev.mean():.3f}")
        print(f"  Eval set (gate + labeled):       n={len(y_ev_all)}, pos={y_ev_all.mean():.3f}")

        # --- Score components (both-arms set) ---
        g_nested = nested_mcc(g_ev, y_ev, f_ev, GATE_GRID)
        s_nested = nested_mcc(s_ev, y_ev, f_ev, SHIFT_GRID)
        g_tuned = tuned_mcc(g_ev, y_ev, GATE_GRID)
        s_tuned = tuned_mcc(s_ev, y_ev, SHIFT_GRID)

        # Product score
        best_prod = (-1, None)
        for k in [0.5, 1, 2, 3, 5, 10]:
            pn = nested_mcc(sigprod(g_ev, s_ev, k), y_ev, f_ev, PROD_GRID)
            if pn > best_prod[0]:
                best_prod = (pn, k)
        prod_tuned = max(tuned_mcc(sigprod(g_ev, s_ev, k), y_ev, PROD_GRID)
                         for k in [0.5, 1, 2, 3, 5, 10])

        # Conj margin
        margin = np.minimum(g_ev - GATE_THR, s_ev - SHIFT_THR)
        margin01 = 1 / (1 + np.exp(-3.0 * margin))
        cm_nested = nested_mcc(margin01, y_ev, f_ev, PROD_GRID)
        cm_tuned = tuned_mcc(margin01, y_ev, PROD_GRID)

        # --- Pearson of shift pred vs true delta ---
        true_delta = (tdic - dirv)[eval_idx]
        shift_pearson = np.corrcoef(s_ev, true_delta)[0, 1]
        gate_pearson = np.corrcoef(g_ev, tdic[eval_idx])[0, 1]

        report[iso] = {
            "n_both_arms": int(len(y_ev)),
            "n_gate_labeled": int(len(y_ev_all)),
            "pos_rate": float(y_ev.mean()),
            "gate_pearson": float(gate_pearson),
            "shift_pearson": float(shift_pearson),
            "gate": {"nested": float(g_nested), "tuned": float(g_tuned)},
            "shift": {"nested": float(s_nested), "tuned": float(s_tuned)},
            "product": {"nested": float(best_prod[0]), "k": best_prod[1], "tuned": float(prod_tuned)},
            "conj_margin": {"nested": float(cm_nested), "tuned": float(cm_tuned)},
        }

        print(f"\n  CYP{short} RESULTS (nested-honest MCC, both-arms set):")
        print(f"    gate  pearson={gate_pearson:.4f}  nested={g_nested:.4f}  tuned={g_tuned:.4f}")
        print(f"    shift pearson={shift_pearson:.4f}  nested={s_nested:.4f}  tuned={s_tuned:.4f}")
        print(f"    product (k={best_prod[1]}):        nested={best_prod[0]:.4f}  tuned={prod_tuned:.4f}")
        print(f"    conj_margin:                nested={cm_nested:.4f}  tuned={cm_tuned:.4f}")

    # --- Macro summary ---
    print("\n" + "=" * 66)
    print("PHASE 1 T1 FRESH SHIFT/GATE - NESTED-HONEST MACRO MCC")
    print("=" * 66)
    print(f"  {'score':<15} {'2D6':>8} {'3A4':>8} {'macro':>8}  {'v6b bar':>9}")
    for s in ["gate", "shift", "product", "conj_margin"]:
        v2d6 = report["CYP2D6"][s]["nested"]
        v3a4 = report["CYP3A4"][s]["nested"]
        macro = (v2d6 + v3a4) / 2
        bar = "0.3448" if s in ("product", "conj_margin") else "-"
        print(f"  {s:<15} {v2d6:>8.4f} {v3a4:>8.4f} {macro:>8.4f}  {bar:>9}")

    # Gate decision
    best = max((report[i]["product"]["nested"] + report[j]["product"]["nested"]) / 2
               for i in TDI_ISO for j in TDI_ISO if i == j)
    # Actually just compute macro properly
    prod_macro = (report["CYP2D6"]["product"]["nested"] + report["CYP3A4"]["product"]["nested"]) / 2
    cm_macro = (report["CYP2D6"]["conj_margin"]["nested"] + report["CYP3A4"]["conj_margin"]["nested"]) / 2
    best_macro = max(prod_macro, cm_macro)

    print(f"\n  v6b capped bar: macro 0.3448 (2D6 0.219, 3A4 0.470)")
    print(f"  product macro: {prod_macro:.4f}")
    print(f"  conj_margin macro: {cm_macro:.4f}")
    print(f"  BEST macro: {best_macro:.4f}")

    # Per-isoform non-regression
    for iso, bar_iso in [("CYP2D6", 0.219), ("CYP3A4", 0.470)]:
        for s in ["product", "conj_margin"]:
            v = report[iso][s]["nested"]
            status = "PASS" if v >= bar_iso else "FAIL"
            print(f"  {iso} {s}: {v:.4f} vs bar {bar_iso} -> {status}")

    if best_macro > 0.3448:
        print(f"\n  *** BEATS THE BAR by {best_macro - 0.3448:.4f} ***")
    else:
        print(f"\n  *** BELOW the bar by {0.3448 - best_macro:.4f} ***")

    # Save
    out = os.path.join(CACHE, "phase1_t1_fresh_shift_gate.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
