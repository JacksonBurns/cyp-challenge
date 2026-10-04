"""TDI-track gate eval: honest MCC for UMTM is_TDI heads vs legacy standalone.

Protocol (matches run_tdi.py + tdi_blend_nested honesty rules):
- oof-tuned MCC (thr grid on pooled OOF) == run_tdi's oof_mcc protocol (optimistic
  by construction, used for BOTH legacy and UMTM so the comparison is fair);
- nested MCC: threshold picked on folds != f, applied to fold f, pooled - the
  number we quote as 'honest'.
Usage (cyp env): python src/umtm_eval_tdi.py cache/umtm_oof_L2_s0.csv [--fold-col fold_tdi]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(os.path.dirname(HERE), "data")
TDI_ISO = ["CYP2D6", "CYP3A4"]
GRID = np.linspace(0.05, 0.85, 81)


def mcc_at(p, y, t):
    pr = (p >= t).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum())
    fp = int(((pr == 1) & (y == 0)).sum())
    tn = int(((pr == 0) & (y == 0)).sum())
    fn = int(((pr == 0) & (y == 1)).sum())
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def truth():
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES")
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv")).drop_duplicates("SMILES")
    lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in TDI_ISO]],
                     emx[["SMILES"] + [f"{i}_is_TDI" for i in TDI_ISO]]]).drop_duplicates("SMILES")
    return lab.set_index("SMILES")


def eval_probs(p_by_smi, folds_by_smi, y_by_smi, iso):
    smi = [s for s in p_by_smi.index if not np.isnan(p_by_smi[s]) and s in y_by_smi.index
           and not pd.isna(y_by_smi[s])]
    p = p_by_smi.loc[smi].astype(float).values
    y = y_by_smi.loc[smi].map({True: 1, False: 0, 1.0: 1, 0.0: 0}).astype(float).values
    fo = folds_by_smi.loc[smi].astype(int).values
    tuned = max((mcc_at(p, y, t), t) for t in GRID)
    m05 = mcc_at(p, y, 0.5)
    # nested: thr from OTHER folds only
    nested_preds = np.zeros(len(y), dtype=int)
    for f in np.unique(fo):
        tr, va = fo != f, fo == f
        if tr.sum() == 0 or va.sum() == 0:
            continue
        t = max(GRID, key=lambda t: mcc_at(p[tr], y[tr], t))
        nested_preds[va] = (p[va] >= t).astype(int)
    tp = int(((nested_preds == 1) & (y == 1)).sum())
    fp = int(((nested_preds == 1) & (y == 0)).sum())
    tn = int(((nested_preds == 0) & (y == 0)).sum())
    fn = int(((nested_preds == 0) & (y == 1)).sum())
    nested = (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)
    return {"n": len(y), "pos": float(y.mean()), "mcc_tuned": tuned[0], "thr": tuned[1],
            "mcc_at_0.5": m05, "mcc_nested": float(nested),
            "auc": _auc(y, p)}


def _auc(y, p):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def main(paths, fold_col, avg_group=None):
    lab = truth()
    if avg_group:
        # average probability columns across same-schema (same-lineage) seeds;
        # files with different column sets pass through as separate rows
        groups = {}
        for p in paths:
            d = pd.read_csv(p)
            key = tuple(c for c in d.columns if c.startswith("is_TDI_"))
            groups.setdefault(key, []).append((os.path.basename(p).replace(".csv", ""), d))
        rows = []
        for key, members in groups.items():
            if len(members) == 1:
                rows.append(members[0])
                continue
            base = members[0][1].copy()
            for c in key:
                base[c] = np.mean([d[c].values.astype(float) for _, d in members], axis=0)
            nm = "~avg-" + os.path.basename(members[0][0]).replace(".csv", "")
            rows.append((nm, base))
    else:
        rows = [(os.path.basename(p).replace(".csv", ""), pd.read_csv(p)) for p in paths]
    for name, df in rows:
        fc = fold_col if fold_col in df.columns else ("fold_tdi" if "fold_tdi" in df.columns else "fold")
        out = {}
        for iso in TDI_ISO:
            cands = [f"is_TDI_{iso}", f"is_TDI_{iso[3:]}", f"{iso}_proba"]
            col = next((c for c in cands if c in df.columns), None)
            if col is None:
                continue
            d = df[[col, fc]].copy()
            d.index = df["SMILES"].values
            out[iso] = eval_probs(d[col], d[fc], lab[f"{iso}_is_TDI"], iso)
        line = " | ".join(f"{i}: nested {o['mcc_nested']:.4f} tuned {o['mcc_tuned']:.4f} "
                          f"@0.5 {o['mcc_at_0.5']:.4f} auc {o['auc']:.3f} (n={o['n']}, pos={o['pos']:.2f})"
                          for i, o in out.items())
        mac = np.mean([o["mcc_nested"] for o in out.values()])
        print(f"{name}: macro-nested {mac:.4f}\n  {line}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fold-col", default="fold_tdi")
    ap.add_argument("--avg", action="store_true", help="average is_TDI cols across all paths, eval once")
    a = ap.parse_args()
    main(a.paths, a.fold_col, avg_group=a.avg)
