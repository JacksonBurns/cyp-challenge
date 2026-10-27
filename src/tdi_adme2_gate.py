"""ADME2-4: do predicted assay-physics features help TDI CLASSIFICATION?

Mirrors run_tdi.py (LGBM, scaffold folds, sim-NN features, MCC threshold
tune) with ADME2 features appended. Baseline = run WITHOUT ADME2 (same
machinery, in-run) so comparison is apples-to-apples; ship baseline
v4_candidate field E[MCC]: 3A4 0.409 / 2D6 0.163.

Gate: field MCC improves on >=1 isoform by >0.02 with baseline in-run.
CPU only.
"""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
from run_regression import make_folds, scaffold_groups  # noqa: E402

ISO = ["CYP2D6", "CYP3A4"]
FP_COLS = [f"m2_{i}" for i in range(2048)]
K = 5


def mcc(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def main():
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"))
    Xtr = Xtr.drop_duplicates("SMILES").reset_index(drop=True)
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                     emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab.reindex(Xtr["SMILES"])

    A2 = np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy"))
    # adme2 feats built on H.Xtr order; align to dedup Xtr by SMILES join
    sys.path.insert(0, HERE)
    import hinge_blend as H
    key = pd.Index(H.Xtr["SMILES"].values)
    pos = key.get_indexer(Xtr["SMILES"].values)
    A2 = A2[pos]
    print("adme2 aligned", A2.shape)

    Xbase = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    groups = scaffold_groups(Xtr["SMILES"].tolist())
    folds = make_folds(groups, seed=7)
    Btr = (Xtr[FP_COLS].values > 0).astype(np.float32)

    def sim(A, Bb):
        d = A @ Bb.T
        ca = (A > 0).sum(1)[:, None]
        cb = (Bb > 0).sum(1)[None, :]
        return d / np.maximum(ca + cb - d, 1e-9)

    S = sim(Btr, Btr)
    res = {}
    for iso in ISO:
        y = Y[f"{iso}_is_TDI"].values
        lab_mask = ~pd.isna(y)
        yl = np.zeros(len(y))
        yl[lab_mask] = y[lab_mask].astype(bool).astype(float)
        cand_all = np.where(lab_mask)[0]
        for tag, XA in [("base", np.zeros((len(y), 0), np.float32)),
                        ("adme2", A2.astype(np.float32))]:
            proba = np.full(len(y), np.nan)
            for f in range(K):
                trc = cand_all[folds[cand_all] != f]
                vac = cand_all[folds[cand_all] == f]

                def nnfeat(qS, q_idx):
                    pos = trc[yl[trc] > 0]
                    Sp = qS[:, pos]
                    pos_of = {c: j for j, c in enumerate(pos)}
                    for i, q in enumerate(q_idx):
                        j = pos_of.get(q)
                        if j is not None:
                            Sp[i, j] = -1.0
                    maxp = Sp.max(1)
                    topk = np.partition(-Sp, min(9, Sp.shape[1] - 1), axis=1)[:, :10]
                    return np.column_stack([maxp, -topk.mean(1), (Sp > 0.7).sum(1)]).astype(np.float32)

                Xf_tr = np.hstack([Xbase[trc], nnfeat(S[trc], trc), XA[trc]])
                Xf_va = np.hstack([Xbase[vac], nnfeat(S[vac], vac), XA[vac]])
                m = LGBMClassifier(n_estimators=1200, learning_rate=0.03, num_leaves=63,
                                   min_child_samples=15, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.5, reg_lambda=2.0, n_jobs=-1,
                                   random_state=42, verbosity=-1)
                m.fit(Xf_tr, yl[trc])
                proba[vac] = m.predict_proba(Xf_va)[:, 1]
            yy = yl[lab_mask].astype(int)
            scores = []
            for t in np.linspace(0.05, 0.85, 81):
                p = (proba[lab_mask] >= t).astype(int)
                tp = int(((p == 1) & (yy == 1)).sum()); fp = int(((p == 1) & (yy == 0)).sum())
                tn = int(((p == 0) & (yy == 0)).sum()); fn = int(((p == 0) & (yy == 1)).sum())
                scores.append(mcc(tp, fp, tn, fn))
            bi = int(np.argmax(scores))
            # field threshold: fixed f=0.33 (2D6) / 0.26 (3A4) quantile rule
            q = 0.33 if iso == "CYP2D6" else 0.26
            tf = np.quantile(proba[lab_mask], 1 - q)
            p = (proba[lab_mask] >= tf).astype(int)
            tp = int(((p == 1) & (yy == 1)).sum()); fp = int(((p == 1) & (yy == 0)).sum())
            tn = int(((p == 0) & (yy == 0)).sum()); fn = int(((p == 0) & (yy == 1)).sum())
            res[(iso, tag)] = (scores[bi], mcc(tp, fp, tn, fn))
            print(f"{iso} {tag}: best-thr MCC {scores[bi]:.4f} | field MCC {mcc(tp, fp, tn, fn):.4f}", flush=True)
    for iso in ISO:
        d = res[(iso, "adme2")][1] - res[(iso, "base")][1]
        print(f"FIELD DELTA {iso}: {d:+.4f}")


if __name__ == "__main__":
    main()
