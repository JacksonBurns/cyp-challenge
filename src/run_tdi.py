"""TDI classifier: LightGBM per scored isoform, scaffold-grouped CV, MCC threshold tuning.

Writes: cache/tdi_test_probs.csv, cache/tdi_cv.json
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
from run_regression import make_folds, scaffold_groups  # noqa: E402

warnings.filterwarnings("ignore")

ISO = ["CYP2D6", "CYP3A4"]
FP_COLS = [f"m2_{i}" for i in range(2048)]
K = 5


def mcc(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def main():
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"))
    Xte = pd.read_parquet(os.path.join(CACHE, "X_test.parquet"))
    Xtr = Xtr.drop_duplicates("SMILES").reset_index(drop=True)
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))

    # labels: union of TDI + Emax files (identical where overlapping; Emax adds none for scored isoforms)
    lab_by_smi = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                            emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab_by_smi.reindex(Xtr["SMILES"])

    Xbase = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    Xte_base = Xte.drop(columns=["SMILES"]).values.astype(np.float32)
    groups = scaffold_groups(Xtr["SMILES"].tolist())
    folds = make_folds(groups, seed=7)

    # NN-sim features against in-fold labeled positives: distance-to-positive class signal
    Btr = (Xtr[FP_COLS].values > 0).astype(np.float32)
    Bte = (Xte[FP_COLS].values > 0).astype(np.float32)

    def sim(A, Bb):
        d = A @ Bb.T
        ca = (A > 0).sum(1)[:, None]
        cb = (Bb > 0).sum(1)[None, :]
        return d / np.maximum(ca + cb - d, 1e-9)

    S = sim(Btr, Btr)
    Ste = sim(Bte, Btr)

    out_cv, out_test = {}, {}
    oof_store = {}
    for iso in ISO:
        y = Y[f"{iso}_is_TDI"].values
        lab_mask = ~pd.isna(y)
        yl = np.zeros(len(y))
        yl[lab_mask] = y[lab_mask].astype(bool).astype(float)
        proba = np.full(len(y), np.nan)
        te_preds = []
        cand_all = np.where(lab_mask)[0]
        for f in range(K):
            trc = cand_all[folds[cand_all] != f]
            vac = cand_all[folds[cand_all] == f]
            # per-query NN features vs positives in trc (self-excluded)
            def nnfeat(qS, q_idx=None):
                pos = trc[yl[trc] > 0]
                Sp = qS[:, pos]
                pos_of = {c: j for j, c in enumerate(pos)}
                if q_idx is not None:
                    for i, q in enumerate(q_idx):
                        j = pos_of.get(q)
                        if j is not None:
                            Sp[i, j] = -1.0
                maxp = Sp.max(1)
                topk = np.partition(-Sp, min(9, Sp.shape[1] - 1), axis=1)[:, :10]
                return np.column_stack([maxp, -topk.mean(1), (Sp > 0.7).sum(1)]).astype(np.float32)
            qtr = S[trc]
            qva = S[vac]
            Xf_tr = np.hstack([Xbase[trc], nnfeat(qtr, trc)])
            Xf_va = np.hstack([Xbase[vac], nnfeat(qva, vac)])
            m = LGBMClassifier(n_estimators=1200, learning_rate=0.03, num_leaves=63,
                               min_child_samples=15, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, reg_lambda=2.0, n_jobs=-1,
                               random_state=42, verbosity=-1)
            m.fit(Xf_tr, yl[trc])
            proba[vac] = m.predict_proba(Xf_va)[:, 1]
            te_preds.append(m.predict_proba(np.hstack([Xte_base, nnfeat(Ste)]))[:, 1])
        # threshold tune MCC on OOF
        cand_th = np.linspace(0.05, 0.85, 81)
        scores = []
        yy = yl[lab_mask].astype(int)
        for t in cand_th:
            p = (proba[lab_mask] >= t).astype(int)
            tp = int(((p == 1) & (yy == 1)).sum()); fp = int(((p == 1) & (yy == 0)).sum())
            tn = int(((p == 0) & (yy == 0)).sum()); fn = int(((p == 0) & (yy == 1)).sum())
            scores.append(mcc(tp, fp, tn, fn))
        bi = int(np.argmax(scores))
        oof_store[iso] = proba
        out_cv[iso] = {"best_thr": float(cand_th[bi]), "oof_mcc": float(scores[bi]),
                       "mcc_at_0.5": float(scores[int(np.argmin(np.abs(cand_th - 0.5)))]),
                       "pearson": float(pearsonr(yy, proba[lab_mask]).statistic),
                       "pos_rate": float(yy.mean()), "n": int(lab_mask.sum())}
        out_test[iso] = np.mean(te_preds, axis=0)
        print(iso, json.dumps(out_cv[iso]), flush=True)
    pd.DataFrame({"SMILES": Xte["SMILES"], **{f"{i}_proba": out_test[i] for i in ISO}}).to_csv(
        os.path.join(CACHE, "tdi_test_probs.csv"), index=False)
    np.savez(os.path.join(CACHE, "tdi_oof.npz"), **{i: v for i, v in oof_store.items()})
    with open(os.path.join(CACHE, "tdi_cv.json"), "w") as fh:
        json.dump(out_cv, fh, indent=2)


if __name__ == "__main__":
    main()
