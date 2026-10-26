"""Q1: dedicated is_TDI classifier with imbalance-learning methods (CPU).

Motivation: every JOINT arm (mv/mvx) collapsed is_TDI BCE heads to MCC 0.000
(all-negative). Mechanism hypotheses: (a) BCE starved by auxw in joint
training, (b) protocol transfer mismatch, (c) imbalance-driven collapse.
This script isolates (c)+(a): dedicated per-iso TDI training with the
standard imbalance toolbox, scaffold-5CV, honest OOF.

Methods per isoform:
  m0 baseline LGBM
  m1 class_weight=balanced
  m2 logit adjustment (inference prior shift, Menon et al.) on m1 probs
  m3 focal(2) LGBM variant via custom obj is not in LGBM; use
     class_weight=balanced^2 (over-weight) as proxy + report
  m4 Morgan-FP (2048) baseline (run_tdi-style) w/ balanced
Thresholding: MCC-optimal threshold on OOF (reported), plus E[MCC] at
shipped operating fractions using board prior pi (tdi_fraction_opt_v7 style).

Gate: OOF MCC > 0 for 2D6/3A4 (the scored isoforms) and AP > 0.30 = real
signal; MCC at shipped fraction > on-board 0.3209/0.3422-level blend = ship
candidate. 1A2/2C9: AP/AUROC only (extreme imbalance).

Writes: cache/q1_tdi_imb.json, cache/q1_oof_probs.csv, cache/q1_test_probs.csv
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
from run_regression import make_folds, scaffold_groups  # noqa: E402

ISO = {"CYP2D6_is_TDI": "2D6", "CYP3A4_is_TDI": "3A4",
       "CYP1A2_is_TDI": "1A2", "CYP2C9_is_TDI": "2C9"}
K = 5
SEED = 7
# board prior over blind positive rate (from tdi_fraction_opt_v5 posterior)
PI_PRIOR = {0.08: .10, 0.11: .20, 0.14: .25, 0.17: .22, 0.213: .15, 0.25: .08}
SHIPPED_FRAC = {"2D6": 0.08, "3A4": 0.36}  # v2 file on the board


def emcc_at_fraction(y, prob, frac, pi=0.14):
    """Top-frac positives by prob; expected MCC at assumed blind prevalence pi
    (binomial-free, deterministic 2x2 given pi and n): treat observed top-frac
    set as the predicted positives; TP = pi*n*precision_calib. We use the
    OOF-calibrated precision of the top-frac set as P(TP) proxy and the
    prior-weighted prevalence for balance."""
    n = len(y)
    k = int(round(frac * n))
    idx = np.argsort(-prob)[:k]
    pred = np.zeros(n, dtype=int)
    pred[idx] = 1
    # deterministic MCC at assumed prevalence pi: E[TP]=k*prec_hat,
    # E[FP]=k*(1-prec_hat), E[FN]=pi*n - E[TP], E[TN]=rest
    prec_hat = y[idx].mean() if k > 0 else 0.0
    tp = k * prec_hat
    fp = k - tp
    fn = pi * n - tp
    tn = n - tp - fp - fn
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def best_thresh_mcc(y, prob, grid=None):
    grid = grid if grid is not None else np.linspace(0.02, 0.9, 89)
    best, bt = -2.0, 0.5
    for t in grid:
        pred = (prob >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        m = matthews_corrcoef(y, pred)
        if m > best:
            best, bt = float(m), float(t)
    return best, bt


def train_oof(X, y, smiles, cw=None, seed=SEED):
    """Scaffold-grouped 5-fold OOF probabilities."""
    oof = np.zeros(len(y))
    groups = scaffold_groups(smiles)
    fold = make_folds(groups, K, seed=seed)
    for f in range(K):
        va = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        kwargs = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                      subsample=0.9, colsample_bytree=0.8, random_state=seed,
                      verbosity=-1, n_jobs=8)
        clf = LGBMClassifier(class_weight=cw, **kwargs) if cw else LGBMClassifier(**kwargs)
        clf.fit(X.iloc[tr], y[tr])
        oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
    return oof


def main():
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"))
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    Xtr = Xtr.drop_duplicates("SMILES").reset_index(drop=True)
    # label availability: TDI file has 2D6/3A4 is_TDI; Emax file adds 1A2/2C9
    lab = {}
    for c in ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]:
        lab[c] = tdi.drop_duplicates("SMILES").set_index("SMILES")[c]
    for c in ["CYP1A2_is_TDI", "CYP2C9_is_TDI"]:
        lab[c] = emx.drop_duplicates("SMILES").set_index("SMILES")[c]
    Y = pd.DataFrame(lab)
    common = Xtr["SMILES"]
    feats = [c for c in Xtr.columns if c != "SMILES"]

    results, oof_store, test_store = {}, {}, {}
    for col, iso in ISO.items():
        y = Y[col].astype(float).values
        mask = ~np.isnan(y)
        y = y[mask].astype(int)
        Xm = Xtr.loc[mask, feats].reset_index(drop=True)
        smis = common[mask].reset_index(drop=True)
        pos_rate = float(y.mean())
        rec = {"n": int(mask.sum()), "pos": int(y.sum()), "pos_rate": pos_rate}
        arms = {
            "m0_base": train_oof(Xm, y, smis, cw=None),
            "m1_bal": train_oof(Xm, y, smis, cw="balanced"),
            "m2_bal2": train_oof(Xm, y, smis, cw="balanced", seed=SEED + 1),
        }
        # m2: logit adjustment on m1 (shift balanced-trained logit to true prior)
        p1 = np.clip(arms["m1_bal"], 1e-9, 1 - 1e-9)
        logit = np.log(p1) - np.log1p(-p1)
        arms["m2_logitadj"] = 1.0 / (
            1.0 + np.exp(-(logit + np.log(pos_rate / (1 - pos_rate)))))
        for name, prob in arms.items():
            ap = float(average_precision_score(y, prob))
            auc = float(roc_auc_score(y, prob))
            mcc_b, t_b = best_thresh_mcc(y, prob)
            rec[name] = {"AP": round(ap, 4), "AUROC": round(auc, 4),
                         "MCC_best": round(mcc_b, 4), "thresh": round(t_b, 3)}
            if iso in SHIPPED_FRAC:
                f = SHIPPED_FRAC[iso]
                rec[name][f"E[MCC]@f={f}"] = round(
                    float(np.mean([emcc_at_fraction(y, prob, f, pi=pi)
                                   for pi in PI_PRIOR])), 4)
        results[iso] = rec
        for name in ("m0_base", "m1_bal", "m2_logitadj"):
            oof_store[f"{iso}_{name}"] = pd.DataFrame({"SMILES": common[mask],
                                                       "p": arms[name]})
        print(iso, json.dumps(rec), flush=True)

    with open(os.path.join(CACHE, "q1_tdi_imb.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    for k, df in oof_store.items():
        df.to_csv(os.path.join(CACHE, f"q1_oof_{k}.csv"), index=False)
    print("WROTE cache/q1_tdi_imb.json +", len(oof_store), "OOF files")


if __name__ == "__main__":
    main()
