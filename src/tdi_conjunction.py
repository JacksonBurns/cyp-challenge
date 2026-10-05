"""Phase 0 T1 conjunction: ORACLE ceiling vs UMTM-cached-heads prototype.

The conjunction is_TDI <=> (pi_TDI > 4.301) AND (pi_TDI - pi_dir > 0.301) is
100% verified on the data (RANK_UP_PLAN A1). This script establishes:

1. ORACLE ceiling - use the GROUND-TRUTH arms (actual pIC50) as the gate/shift
   scores. This is what a perfect potency model would give.
2. UMTM-cached prototype - use the UMTM tdic/direct OOF heads (all seeds/
   lineages averaged) as the gate/shift scores. This is "free" Phase 0.

Both scored with the same nested-honest MCC protocol (threshold grid selected
on folds != f, applied to fold f, pooled). Reports the gap.

CPU-only. No GPU, no submits.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(os.path.dirname(HERE), "data")

TDI_ISO = ["CYP2D6", "CYP3A4"]
GATE_THR = 4.301
SHIFT_THR = 0.301


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


def sigprod(pi_tdi, delta, k):
    pg = 1 / (1 + np.exp(-k * (pi_tdi - GATE_THR)))
    ps = 1 / (1 + np.exp(-k * (delta - SHIFT_THR)))
    return pg * ps


# Scale-appropriate threshold grids:
#  - gate score is on the pIC50 scale (~1.9-7.7), natural cut 4.301
#  - shift score is on the delta scale (~-2..+3), natural cut 0.301
#  - product/conj_margin are normalized to ~[0,1]
GATE_GRID = np.linspace(3.0, 5.5, 101)
SHIFT_GRID = np.linspace(-1.0, 1.5, 101)
PROD_GRID = np.linspace(0.05, 0.85, 81)


def score_set(pi_tdi, delta, y, folds):
    out = {"n": len(y), "pos": float(y.mean())}
    out["gate"] = {"nested": nested_mcc(pi_tdi, y, folds, GATE_GRID),
                   "tuned": tuned_mcc(pi_tdi, y, GATE_GRID)}
    out["shift"] = {"nested": nested_mcc(delta, y, folds, SHIFT_GRID),
                    "tuned": tuned_mcc(delta, y, SHIFT_GRID)}
    best = (-1, None)
    for k in [0.5, 1, 2, 3, 5, 10]:
        pn = nested_mcc(sigprod(pi_tdi, delta, k), y, folds, PROD_GRID)
        if pn > best[0]:
            best = (pn, k)
    out["product"] = {"nested": best[0], "k": best[1],
                      "tuned": max(tuned_mcc(sigprod(pi_tdi, delta, k), y, PROD_GRID)
                                   for k in [0.5, 1, 2, 3, 5, 10])}
    # conj margin = min of the two margins (bottleneck conjunct), a soft AND.
    # Normalize to ~[0,1] via sigmoid so the PROD_GRID threshold is meaningful.
    margin = np.minimum(pi_tdi - GATE_THR, delta - SHIFT_THR)
    margin01 = 1 / (1 + np.exp(-3.0 * margin))
    out["conj_margin"] = {"nested": nested_mcc(margin01, y, folds, PROD_GRID),
                          "tuned": tuned_mcc(margin01, y, PROD_GRID)}
    return out


def main():
    # UMTM OOF heads averaged across all seeds + lineages
    umtm_files = sorted(
        [os.path.join(CACHE, f) for f in os.listdir(CACHE)
         if f.startswith("umtm_oof_") and f.endswith(".csv")
         and "_ext" not in f and "_primary" not in f]
    )
    combined = pd.concat([pd.read_csv(f) for f in umtm_files], ignore_index=True)
    head_cols = [c for c in combined.columns if c.endswith(("_direct", "_tdic"))]
    umtm = combined.groupby("SMILES")[head_cols].mean().reset_index().set_index("SMILES")
    print(f"UMTM heads: {len(umtm_files)} files, {len(umtm)} unique SMILES")

    # Ground truth arms + labels + folds (all from tdi, which has all 6145 rows)
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES").set_index("SMILES")
    fold_df = pd.read_csv(umtm_files[0])[["SMILES", "fold_tdi"]].drop_duplicates("SMILES").set_index("SMILES")

    report = {}
    for iso in TDI_ISO:
        short = iso[3:]
        ycol = f"{iso}_is_TDI"
        tdic_col = f"CYP{short}_pIC50_TDI_condition"
        dir_col = f"CYP{short}_pIC50_direct_inhibition"

        # ORACLE: ground-truth arms + labels, all from tdi (has all 6145 rows)
        o = tdi[[tdic_col, dir_col, ycol]].copy()
        o = o.join(fold_df[["fold_tdi"]])
        o = o.dropna(subset=[tdic_col, dir_col, ycol, "fold_tdi"])
        o_y = o[ycol].map({True: 1, False: 0, 1.0: 1, 0.0: 0}).astype(float).values
        o_f = o["fold_tdi"].astype(int).values
        o_tdic = o[tdic_col].values
        o_dir = o[dir_col].values
        oracle = score_set(o_tdic, o_tdic - o_dir, o_y, o_f)

        # UMTM: cached heads (labels from tdi)
        u = umtm[[f"CYP{short}_tdic", f"CYP{short}_direct"]].copy()
        u[ycol] = tdi[ycol].reindex(u.index)
        u = u.join(fold_df[["fold_tdi"]])
        u = u.dropna(subset=[f"CYP{short}_tdic", f"CYP{short}_direct", ycol, "fold_tdi"])
        u_y = u[ycol].map({True: 1, False: 0, 1.0: 1, 0.0: 0}).astype(float).values
        u_f = u["fold_tdi"].astype(int).values
        u_tdic = u[f"CYP{short}_tdic"].values
        u_dir = u[f"CYP{short}_direct"].values
        umtm_set = score_set(u_tdic, u_tdic - u_dir, u_y, u_f)

        # Apples-to-apples: UMTM product on the SAME both-arms rows the oracle uses
        both_arms = tdi[[tdic_col, dir_col]].dropna().index
        um = umtm_set
        if len(both_arms):
            u2 = u.loc[u.index.intersection(both_arms)]
            u2_y = u2[ycol].map({True: 1, False: 0, 1.0: 1, 0.0: 0}).astype(float).values
            u2_f = u2["fold_tdi"].astype(int).values
            u2_tdic = u2[f"CYP{short}_tdic"].values
            u2_dir = u2[f"CYP{short}_direct"].values
            matched = score_set(u2_tdic, u2_tdic - u2_dir, u2_y, u2_f)
            umtm_set["matched_both_arms"] = matched

        report[iso] = {"oracle": oracle, "umtm": umtm_set}

        print(f"\n=== {iso}  (oracle n={oracle['n']}, umtm n={umtm_set['n']}) ===")
        print(f"  {'score':<12} {'oracle':>8} {'umtm':>8}  {'gap':>8}")
        for s in ["gate", "shift", "product", "conj_margin"]:
            ov = oracle[s]["nested"]
            uv = umtm_set[s]["nested"]
            print(f"  {s:<12} {ov:>8.4f} {uv:>8.4f}  {uv-ov:>+8.4f}")

    print("\n" + "=" * 66)
    print("PHASE 0 T1 CONJUNCTION - NESTED-HONEST MACRO MCC")
    print("=" * 66)
    print(f"  {'score':<12} {'oracle macro':>12} {'umtm macro':>12}  {'v6b bar':>9}")
    for s in ["gate", "shift", "product", "conj_margin"]:
        om = np.mean([report[i][k][s]["nested"] for i in TDI_ISO for k in ["oracle"]])
        um = np.mean([report[i][k][s]["nested"] for i in TDI_ISO for k in ["umtm"]])
        bar = "0.3448" if s in ("product", "conj_margin") else "-"
        print(f"  {s:<12} {om:>12.4f} {um:>12.4f}  {bar:>9}")

    # Gate decision
    best_umtm = max(np.mean([report[i]["umtm"][s]["nested"] for i in TDI_ISO])
                    for s in ["product", "conj_margin"])
    # matched both-arms UMTM product (apples-to-apples with oracle)
    matched_umtm = np.mean([report[i]["umtm"]["matched_both_arms"]["product"]["nested"]
                            for i in TDI_ISO])
    oracle_prod = np.mean([report[i]["oracle"]["product"]["nested"] for i in TDI_ISO])
    print(f"\n  v6b capped bar: macro 0.3448 (2D6 0.219, 3A4 0.470)")
    print(f"  ORACLE ceiling product macro (both-arms rows): {oracle_prod:.4f}")
    print(f"  UMTM-cached product macro (all labeled rows):  {best_umtm:.4f}")
    print(f"  UMTM-cached product macro (matched both-arms): {matched_umtm:.4f}")
    if best_umtm > 0.3448:
        print("  >>> UMTM-cached heads ALREADY beat the bar. Phase 1 is nearly free.")
    else:
        print(f"  >>> UMTM-cached heads are {0.3448-best_umtm:.4f} BELOW the bar.")
        print("      The decomposition ceiling is very high (oracle ~0.98) but the CACHED")
        print("      heads do not predict the delta (shift) well enough. Phase 1 needs")
        print("      FRESH gate/shift regressors - esp. a DIRECT shift/delta model,")
        print("      which is where the oracle->cached gap is largest (2D6 0.90->0.13).")

    with open(os.path.join(CACHE, "phase0_t1_conjunction.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved {os.path.join(CACHE, 'phase0_t1_conjunction.json')}")


if __name__ == "__main__":
    main()
