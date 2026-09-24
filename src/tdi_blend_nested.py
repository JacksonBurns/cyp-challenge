"""Nested-honest TDI blend evaluation: per-fold weight selection, score held-out.

Members per iso: base GBM, emb-aug GBM, TabICL-on-emb, extbase GBM (train +
AID1851 binaries) if present. z-space greedy (with replacement) selected on
OOF rows from folds != f, MCC evaluated on fold f; pooled across folds gives
the honest blend MCC vs the standalone extbase/base MCC. Fold assignment is
identical to run_tdi*.py (scaffold groups on X_train, seed 7).

Writes cache/tdi_blend_nested.json
Usage (cyp env): python src/tdi_blend_nested.py [members_csv]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")

from run_regression import make_folds, scaffold_groups  # noqa: E402

ISO = ["CYP2D6", "CYP3A4"]
K = 5


def mcc(p, y, t):
    pr = (p >= t).astype(int)
    tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum()
    tn = ((pr == 0) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum()
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def best_by_frac(z, yy):
    n = len(yy)
    rows = []
    for f in np.arange(0.05, 0.61, 0.01):
        k = int(n * f)
        t = np.sort(z)[::-1][k]
        rows.append((float(mcc(z, yy, t)), float(f)))
    return max(rows)


def greedy_w(zs, y, rounds=12):
    keys = list(zs)
    w = {k: 0.0 for k in keys}
    cur = None
    for _ in range(rounds):
        bestk, bestr = None, -2
        for k in keys:
            cand = sum((w[j] + (1 if j == k else 0)) * zs[j] for j in keys) / (sum(w.values()) + 1)
            r, _f = best_by_frac(cand, y)
            if r > bestr:
                bestr, bestk = r, k
        w[bestk] += 1
        cur = bestr
    return {k: v / sum(w.values()) for k, v in w.items()}, cur


def main(cand_csv=None):
    members = {"base": np.load(os.path.join(CACHE, "tdi_oof.npz"))}
    for name, fn in [("emb", "tdi_emb_oof.npz"), ("tab", "tdi_tabicl_oof.npz"),
                     ("extbase", "tdi_oof_extbase_w03.npz")]:
        p = os.path.join(CACHE, fn)
        if os.path.exists(p):
            members[name] = np.load(p)
    if cand_csv:
        for part in cand_csv.split(","):
            nm, fn = part.split(":")
            members[nm] = np.load(os.path.join(CACHE, fn))

    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    lab_by_smi = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                            emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab_by_smi.reindex(Xtr["SMILES"])
    folds = make_folds(scaffold_groups(Xtr["SMILES"].tolist()), seed=7)

    out = {}
    for iso in ISO:
        y = Y[f"{iso}_is_TDI"].values
        m = ~pd.isna(y)
        yl = np.array([1 if bool(v) else 0 for v in y[m]])
        zs = {}
        for nm, src in members.items():
            p = src[iso][m].astype(float)
            zs[nm] = (p - p.mean()) / p.std()
        fi = folds[m]
        names = list(zs)
        pooled_blend = np.zeros(m.sum())
        w_hist = []
        for f in range(K):
            trn = fi != f
            va = fi == f
            w, _ = greedy_w({k: v[trn] for k, v in zs.items()}, yl[trn])
            w_hist.append({k: round(v, 2) for k, v in w.items() if v > 0})
            pooled_blend[va] = sum(w[k] * zs[k][va] for k in names)
        rb, fb = best_by_frac(zs["base"], yl)
        rall, fall = best_by_frac(sum(zs[k] for k in names) / len(names), yl)
        nest_mcc, _ = best_by_frac(pooled_blend, yl)
        singles = {}
        for k in names:
            r, _ = best_by_frac(zs[k], yl)
            singles[k] = r
        # oracle (in-sample argmax over simplex step .1) for reference
        best = (-2, None)
        grid = [(a, b) for a in np.arange(0, 1.001, 0.1)
                for b in np.arange(0, 1.001 - a + 1e-9, 0.1)]
        for a, b in grid:
            c = 1 - a - b
            if c < -1e-9 or len(names) < 3:
                continue
            z = a * zs[names[0]] + b * zs[names[1]] + c * zs[names[2]]
            r, ff = best_by_frac(z, yl)
            if r > best[0]:
                best = (r, (names[0], a, names[1], b, names[2], round(c, 2), ff))
        out[iso] = {"base_only": [rb, fb], "equal_avg": [rall, fall],
                    "nested_blend": [nest_mcc, fall], "singles": {k: round(v, 4) for k, v in singles.items()},
                    "fold_weights": w_hist}
        print(iso, json.dumps({k: v for k, v in out[iso].items() if k != 'fold_weights'}), flush=True)

    with open(os.path.join(CACHE, "tdi_blend_nested.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
