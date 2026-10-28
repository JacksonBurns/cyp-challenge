"""R3-1: TDI classification - bioactivation mechanism features + MCC-native loss.

Q: does injecting TDI MECHANISM (bioactivation motifs: thiophene->S-oxide,
furan->enedialdehyde, MDA->catechol->quinone, aniline->quinone imine,
alkyne mechanism-based inactivation, quinone imine...) and a soft-MCC
surrogate objective (direct metric optimization, never tried) beat the
logloss LGBM baseline?

Harness identical to tdi_adme2_gate.py (dedup Xtr, scaffold folds,
sim-NN feats, threshold tune, field MCC). Baselines in-run.

Modes:
  base            logloss, no extras
  mech            logloss + bioactivation block
  adme2           logloss + ADME2 block (known: 2D6 +.0103, 3A4 +.0062)
  mech_adme2      logloss + both
  mcc_mech        soft-MCC custom objective + mech block
Gate: field E[MCC] gain > +0.02 over base on >=1 isoform.
"""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import Chem, RDLogger

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
from run_regression import make_folds, scaffold_groups  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ISO = ["CYP2D6", "CYP3A4"]
FP_COLS = [f"m2_{i}" for i in range(2048)]
K = 5

BIOACT = {
    "thiophene": "s1cccc1",
    "alkyl_thiophene": "c1ccccc1?;s1ccccc1",  # replaced below
    "alkyl_thio2": "[#6X4]s1cccc1",
    "thienyl_pyridine": "s1cccc1c2ccccn2",  # nefazodone core
    "furan": "c1ccoc1",
    "vinyl_furan": "c1ccoc1[CX3H2X2]=C",
    "mda": "O1COc2ccccc21",  # methylenedioxy
    "catechol": "c1cc(O)c(O)cc1",
    "hydroquinone": "c1cc(O)cc(O)c1",
    "aniline": "N[#6]1cccc1",
    "anilide": "c1ccccc1N[CX3](=[OX1])",
    "paminophenol": "Nc1ccc(O)cc1",
    "alkyne_terminal": "C#C[#6X4]",
    "alkyne_aryl": "c[CX3][CX3]#C[#6]",
    "enone": "[#6]C=C[CX3](=[OX1])[#6]",
    "acrylonitrile": "C=CC#N",
    "acrylamide": "C=C[CX3](=[OX1])N",
    "vinyl_sulfone": "C=C[SX4](=[OX1])(=[OX1])",
    "benzylic_halide": "[CX3H1X3]([F,Cl,Br,I])c",
    "gem_dihalide": "[CX3H1]([F,Cl,Br,I])[F,Cl,Br,I]",
    "nitroso_arom": "N=Oa",
    "quinone_direct": "O=C1C=CC(=O)C=C1",
    "imine": "[CX3H1]=NX2",
    "oxime_ether": "C=N[OX2]C",
    "hydrazide": "[CX3](=[OX1])[NX3][NX3]",
    "benzothiazole_thiol": "c1cc2[ns]cnc2s1",
    "aryl_hydrazine": "N[NX3]c1ccccc1",
    "nitroparaffin": "[#6X4][N+](=O)[O-]",
}
# fix junk entry
del BIOACT["alkyl_thiophene"]
PATS = {k: Chem.MolFromSmarts(v) for k, v in BIOACT.items()}
PATS = {k: v for k, v in PATS.items() if v is not None}
NAMES = list(PATS)
print("bioact patterns:", len(PATS), flush=True)


def bioact_feats(smiles):
    X = np.zeros((len(smiles), len(NAMES)), np.float32)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for j, k in enumerate(NAMES):
            X[i, j] = min(m.GetSubstructMatches(PATS[k]).__len__(), 4)
    return X


def mcc(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def soft_mcc_obj(y_true):
    """Custom LGBM objective: maximize soft MCC (grad/hess)."""
    def obj(preds, _train):
        p = 1.0 / (1.0 + np.exp(-preds))
        y = y_true
        n = len(y)
        TP = float(np.sum(p * y)); FP = float(np.sum(p * (1 - y)))
        TN = float(np.sum((1 - p) * (1 - y))); FN = float(np.sum((1 - p) * y))
        P, Q, R, S = TP + FP, TP + FN, TN + FP, TN + FN
        eps = 1e-6
        D = np.sqrt(max(P, eps) * max(Q, eps) * max(R, eps) * max(S, eps))
        M = (TP * TN - FP * FN) / D
        dN = y * n - Q  # dN/dsigma_i
        dP, dR, dS = 2 * y - 1, 2 * (1 - y), -np.ones(n)
        dD = D / 2 * (dP / max(P, eps) + dR / max(R, eps) + dS / max(S, eps))
        dM = (dN - M * dD) / D
        g = -dM * p * (1 - p)  # chain rule dsigma/dpreds
        h = np.full(n, 1.0)
        return g, h
    return obj


def main():
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"))
    Xtr = Xtr.drop_duplicates("SMILES").reset_index(drop=True)
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                     emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab.reindex(Xtr["SMILES"])

    MECH = bioact_feats(Xtr["SMILES"].tolist())
    print("mech", MECH.shape, "prevalence>0:", (MECH > 0).mean(0).max().round(3), flush=True)
    A2 = np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy"))
    import hinge_blend as H
    key = pd.Index(H.Xtr["SMILES"].values)
    A2 = A2[key.get_indexer(Xtr["SMILES"].values)]

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
    for iso in ISO:
        y = Y[f"{iso}_is_TDI"].values
        lab_mask = ~pd.isna(y)
        yl = np.zeros(len(y))
        yl[lab_mask] = y[lab_mask].astype(bool).astype(float)
        cand_all = np.where(lab_mask)[0]
        for tag, XA, loss in [("base", np.zeros((len(y), 0), np.float32), "logloss"),
                              ("mech", MECH, "logloss"),
                              ("adme2", A2.astype(np.float32), "logloss"),
                              ("mech_adme2", np.hstack([MECH, A2]), "logloss"),
                              ("mcc_mech", MECH, "softmcc")]:
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
                if loss == "logloss":
                    m = LGBMClassifier(n_estimators=1200, learning_rate=0.03, num_leaves=63,
                                       min_child_samples=15, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.5, reg_lambda=2.0, n_jobs=-1,
                                       random_state=42, verbosity=-1, class_weight="balanced")
                    m.fit(Xf_tr, yl[trc])
                else:
                    m = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=63,
                                       min_child_samples=15, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.5, reg_lambda=2.0, n_jobs=-1,
                                       random_state=42, verbosity=-1,
                                       objective=soft_mcc_obj(yl[trc]))
                    m.fit(Xf_tr, yl[trc])
                proba[vac] = m.predict_proba(Xf_va)[:, 1]
            # threshold tune on labeled rows (field: mean over thresholds of OOF proba)
            pv = proba[cand_all]; yv = yl[cand_all]
            ths = np.quantile(pv, np.linspace(0.3, 0.8, 41))
            best, field = -1, []
            for t in ths:
                pred = (pv >= t)
                tp = (pred & (yv > 0.5)).sum(); fp = (pred & (yv < 0.5)).sum()
                fn = (~pred & (yv > 0.5)).sum(); tn = (~pred & (yv < 0.5)).sum()
                mm = mcc(tp, fp, tn, fn)
                field.append(mm)
                if mm > best:
                    best = mm
            print(f"{iso} {tag:11s} best-thr MCC {best:.4f} | field MCC {np.mean(field):.4f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
