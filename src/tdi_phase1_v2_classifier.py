"""Phase 1 T1 v2: FRESH shift/gate CLASSIFIERS for the conjunction TDI.

Lizard Wizard's shipped model uses classifiers (not regressors) on the binary
decisions: gate = P(pi_TDI > 4.301), shift = P(delta > 0.301). The product of
the two probabilities is the conjunction score. This is more robust than
predicting the exact potency/delta and thresholding, because:
  - The classifier directly optimizes for the threshold decision
  - No scale mismatch issues
  - Probability output is naturally in [0,1] for the product

Features: multiple sets tested (UMTM heads, base+UMTM, cpmed+UMTM).
Folds: scaffold seed 7. CPU-only (LightGBM), no GPU.

Gate: nested macro > 0.3448, 2D6 > 0.219, 3A4 > 0.470.
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

warnings.filterwarnings("ignore")

TDI_ISO = ["CYP2D6", "CYP3A4"]
GATE_THR = 4.301
SHIFT_THR = 0.301
K = 5
PROD_GRID = np.linspace(0.01, 0.95, 95)


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


def oof_classifier(X, y, folds, cand_idx, model_fn):
    """OOF classification: train on folds != f, predict proba on fold f."""
    n = len(X)
    oof = np.full(n, np.nan)
    for f in range(K):
        tr = cand_idx[folds[cand_idx] != f]
        va = cand_idx[folds[cand_idx] == f]
        if len(tr) == 0 or len(va) == 0:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def main():
    # --- Load features ---
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

    # Feature sets to try
    feat_sets = {
        "umtm": umtm_avg,
        "base_umtm": np.hstack([base, umtm_avg]),
        "cpmed_umtm": np.hstack([cpmed_mat, umtm_avg]),
    }
    for name, mat in feat_sets.items():
        print(f"  feat set '{name}': {mat.shape}")

    # --- Folds ---
    groups = scaffold_groups(smi.tolist())
    folds = make_folds(groups, seed=7)

    # --- TDI data ---
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES").set_index("SMILES")

    # LightGBM classifier params (stronger regularization for small data)
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

            # Gate: binary (pi_TDI > 4.301)
            tdic = pd.to_numeric(tdi[tdic_col].reindex(smi), errors="coerce").values
            gate_mask = ~np.isnan(tdic)
            gate_y = (tdic > GATE_THR).astype(float)
            gate_idx = np.where(gate_mask)[0]

            # Shift: binary (delta > 0.301)
            dirv = pd.to_numeric(tdi[dir_col].reindex(smi), errors="coerce").values
            both_mask = ~np.isnan(tdic) & ~np.isnan(dirv)
            delta = (tdic - dirv)
            shift_y = (delta > SHIFT_THR).astype(float)
            shift_idx = np.where(both_mask)[0]

            # Labels
            y_lab = pd.to_numeric(tdi[ycol].reindex(smi), errors="coerce").values
            lab_mask = ~np.isnan(y_lab)
            y = np.zeros(len(smi))
            y[lab_mask] = y_lab[lab_mask]

            # Train gate classifier (OOF)
            gate_oof = oof_classifier(X, gate_y, folds, gate_idx,
                                      lambda: LGBMClassifier(**clf_params))

            # Train shift classifier (OOF)
            shift_oof = oof_classifier(X, shift_y, folds, shift_idx,
                                       lambda: LGBMClassifier(**clf_params))

            # Evaluate on labeled + both-arms rows
            eval_mask = lab_mask & both_mask
            eval_idx = np.where(eval_mask)[0]
            y_ev = y[eval_idx]
            f_ev = folds[eval_idx]
            g_ev = gate_oof[eval_idx]
            s_ev = shift_oof[eval_idx]

            # Fill any NaN in gate/shift (rows where the arm is missing)
            g_ev = np.nan_to_num(g_ev, nan=0.5)
            s_ev = np.nan_to_num(s_ev, nan=0.5)

            # Score components
            g_nested = nested_mcc(g_ev, y_ev, f_ev, PROD_GRID)
            s_nested = nested_mcc(s_ev, y_ev, f_ev, PROD_GRID)
            g_tuned = tuned_mcc(g_ev, y_ev, PROD_GRID)
            s_tuned = tuned_mcc(s_ev, y_ev, PROD_GRID)

            # Product
            prod = g_ev * s_ev
            p_nested = nested_mcc(prod, y_ev, f_ev, PROD_GRID)
            p_tuned = tuned_mcc(prod, y_ev, PROD_GRID)

            # Conj margin (soft AND via min of the two probs)
            cm = np.minimum(g_ev, s_ev)
            cm_nested = nested_mcc(cm, y_ev, f_ev, PROD_GRID)

            # AUC
            from sklearn.metrics import roc_auc_score
            g_auc = roc_auc_score(y_ev, g_ev) if len(np.unique(y_ev)) > 1 else np.nan
            s_auc = roc_auc_score(y_ev, s_ev) if len(np.unique(y_ev)) > 1 else np.nan
            p_auc = roc_auc_score(y_ev, prod) if len(np.unique(y_ev)) > 1 else np.nan

            report[feat_name][iso] = {
                "n": int(len(y_ev)),
                "pos": float(y_ev.mean()),
                "gate": {"nested": float(g_nested), "tuned": float(g_tuned), "auc": float(g_auc)},
                "shift": {"nested": float(s_nested), "tuned": float(s_tuned), "auc": float(s_auc)},
                "product": {"nested": float(p_nested), "tuned": float(p_tuned), "auc": float(p_auc)},
                "conj_min": {"nested": float(cm_nested)},
            }

            print(f"  [{feat_name}] {iso}: n={len(y_ev)} pos={y_ev.mean():.3f} "
                  f"gate {g_nested:.4f}(auc {g_auc:.3f}) "
                  f"shift {s_nested:.4f}(auc {s_auc:.3f}) "
                  f"product {p_nested:.4f}(auc {p_auc:.3f}) "
                  f"conj_min {cm_nested:.4f}", flush=True)

    # --- Macro summary per feature set ---
    print("\n" + "=" * 70)
    print("PHASE 1 T1 v2 - CLASSIFIER CONJUNCTION (nested-honest macro MCC)")
    print("=" * 70)
    for feat_name in feat_sets:
        print(f"\n  Feature set: {feat_name}")
        print(f"  {'score':<12} {'2D6':>8} {'3A4':>8} {'macro':>8}  {'v6b bar':>9}")
        for s in ["gate", "shift", "product", "conj_min"]:
            v2d6 = report[feat_name]["CYP2D6"][s]["nested"]
            v3a4 = report[feat_name]["CYP3A4"][s]["nested"]
            macro = (v2d6 + v3a4) / 2
            bar = "0.3448" if s == "product" else "-"
            print(f"  {s:<12} {v2d6:>8.4f} {v3a4:>8.4f} {macro:>8.4f}  {bar:>9}")

    # Best feature set
    best_feat = None
    best_macro = -1
    for feat_name in feat_sets:
        macro = (report[feat_name]["CYP2D6"]["product"]["nested"] +
                 report[feat_name]["CYP3A4"]["product"]["nested"]) / 2
        if macro > best_macro:
            best_macro = macro
            best_feat = feat_name

    print(f"\n  BEST feature set: {best_feat} (product macro {best_macro:.4f})")
    print(f"  v6b capped bar: macro 0.3448 (2D6 0.219, 3A4 0.470)")
    for iso, bar_iso in [("CYP2D6", 0.219), ("CYP3A4", 0.470)]:
        v = report[best_feat][iso]["product"]["nested"]
        status = "PASS" if v >= bar_iso else "FAIL"
        print(f"  {iso} product: {v:.4f} vs bar {bar_iso} -> {status}")

    if best_macro > 0.3448:
        print(f"\n  *** BEATS THE BAR by {best_macro - 0.3448:.4f} ***")
    else:
        print(f"\n  *** BELOW the bar by {0.3448 - best_macro:.4f} ***")

    out = os.path.join(CACHE, "phase1_t1_v2_classifier.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
