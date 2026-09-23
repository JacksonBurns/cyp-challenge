"""TDI variant: regress the IC50 shift (TDI_arm - direct_arm) then threshold at 0.301.
Handles inferred positives by predicting the TDI arm pIC50 directly for low-activity cpds:
pred_positive if (pred_tdi_arm > 4.301) OR (direct labeled>4 and shift>0.301). Simplify:
score = pred_shift adjusted; positive iff pred_tdi_arm - direct_obs_or_pred > 0.301,
with pred_tdi_arm from a regression on TDI-arm pIC50 and direct = observed when available
else predicted. Compare OOF MCC vs classifier.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

warnings.filterwarnings("ignore")
ISO = ["CYP2D6", "CYP3A4"]


def mcc(tp, fp, tn, fn):
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def main():
    Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
    R.S_ALL, R.STE_ALL = S, Ste
    Xbase = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    Xte_base = Xte.drop(columns=["SMILES"]).values.astype(np.float32)
    groups = R.scaffold_groups(Xtr["SMILES"].tolist())
    folds = R.make_folds(groups, seed=7)

    out = {"test_tdi_arm": {}, "oof": {}}
    # predict TDI-arm pIC50 for scored isoforms (that encodes shift given known direct)
    for iso in ISO:
        y = targets[R.TDIARM[iso]].values
        cand = np.where(~np.isnan(y))[0]
        nb_tr = R.nn_block(S, cand, y, q_idx=np.arange(len(y))).astype(np.float32)
        nb_te = R.nn_block(Ste, cand, y).astype(np.float32)
        Xl = np.hstack([Xbase, nb_tr])
        Xq = np.hstack([Xte_base, nb_te])
        oof = np.full(len(y), np.nan)
        te_p = 0.0
        nte = 0
        for f in range(R.K):
            tr = cand[folds[cand] != f]
            va = cand[folds[cand] == f]
            m = R.lgbm(random_state=42)
            m.fit(Xl[tr], y[tr])
            oof[va] = m.predict(Xl[va])
        for s in range(3):
            m = R.lgbm(random_state=42 + s)
            m.fit(Xl[cand], y[cand])
            te_p = te_p + m.predict(Xq)
            nte += 1
        out["oof"][iso] = oof
        out["test_tdi_arm"][iso] = te_p / nte
        print(iso, "tdi-arm oof pearson", float(pearsonr(y[cand], oof[cand]).statistic), flush=True)

    # label rule check vs observed direct arm
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES")
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv")).drop_duplicates("SMILES")
    lab = pd.concat([tdi[[c for c in ["SMILES"] + [f"{i}_is_TDI" for i in ISO]]],
                     emx[[c for c in ["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab.reindex(Xtr["SMILES"])
    res = {}
    for iso in ISO:
        yd = targets[R.DIRECT[iso]].values
        ytdi = targets[R.TDIARM[iso]].values
        yt = Y[f"{iso}_is_TDI"].values
        m = ~pd.isna(yt)
        yy = np.array([1 if bool(v) else 0 for v in yt[m]])  # noqa: E741
        shift_hat = out["oof"][iso][m] - yd[m]  # use observed direct arm where known
        # rule A: shift>0.301 (only meaningful where direct known); where direct unknown, use tdi arm>4.301
        pred = np.where(np.isnan(yd[m]), out["oof"][iso][m] > 4.301, shift_hat > 0.301)
        scores = []
        for t in np.linspace(-0.2, 0.9, 45):
            pr = np.where(np.isnan(yd[m]), out["oof"][iso][m] > 4.301, shift_hat > t)
            tp = int(((pr) & (yy == 1)).sum()); fp = int(((pr) & (yy == 0)).sum())
            tn = int(((~pr) & (yy == 0)).sum()); fn = int(((~pr) & (yy == 1)).sum())
            scores.append(mcc(tp, fp, tn, fn))
        bi = int(np.argmax(scores))
        res[iso] = {"thr": float(np.linspace(-0.2, 0.9, 45)[bi]), "mcc_shift_rule": float(scores[bi])}
        print(iso, res[iso], flush=True)
    with open(os.path.join(CACHE, "tdi_shift_cv.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    np.savez(os.path.join(CACHE, "tdi_shift_preds.npz"),
             **{f"oof_{i}": out["oof"][i] for i in ISO},
             **{f"test_{i}": out["test_tdi_arm"][i] for i in ISO})


if __name__ == "__main__":
    main()
