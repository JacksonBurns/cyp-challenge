"""TDI base classifier retrained on challenge train + PubChem AID1851 binary rows.

Fixes the precision deficit (blind prec 0.408 vs top-10 0.55-0.71) by adding
~14k qHTS-labeled compounds. External rows join TRAINING only (never scored);
challenge rows keep identical features to src/run_tdi.py (NN-sim features vs
in-fold challenge positives), external rows get the same feature definition
computed against the same positive pool. sample_weight for external = W_EXT
(population shift: qHTS is not DRC/TDI-calibrated, ranking transfer only).

Outputs (cyp env):
  cache/tdi_oof_extbase.npz        challenge-only OOF proba per iso
  cache/tdi_extbase_test_probs.csv mean-of-folds test proba
  cache/tdi_cv_extbase.json        MCC/pearson + chosen W_EXT
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")
EXT = os.path.join(ROOT, "external")

from featurize import featurize_smiles_list  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from run_regression import make_folds, scaffold_groups  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ISO = ["CYP2D6", "CYP3A4"]
FP_COLS = [f"m2_{i}" for i in range(2048)]
K = 5
W_EXT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
TAG = sys.argv[2] if len(sys.argv) > 2 else "extbase"


def mcc(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def load_external(exclude):
    pc = pd.read_csv(os.path.join(EXT, "pubchem_cyp_qhts_aid1851.csv"))
    pc["canon"] = pc["smiles"].map(canon)
    pc = pc.dropna(subset=["canon"])
    # collapse duplicate canonical SMILES: drop iso-label conflicts, else median
    rows = []
    for iso in ISO:
        c = pc[["canon", f"{iso}_active"]].dropna()
        g = c.groupby("canon")[f"{iso}_active"].agg(["mean", "nunique"])
        g = g[(g["nunique"] == 1) & (g["mean"] <= 1.0)].reset_index()
        g[f"{iso}_lab"] = g["mean"].astype(int)
        rows.append(g[["canon", f"{iso}_lab"]])
    ext = rows[0].merge(rows[1], on="canon", how="outer")
    ext = ext[~ext["canon"].isin(exclude)]
    ext = ext.drop_duplicates("canon").reset_index(drop=True)
    X, names, keep, feat_names = featurize_smiles_list(ext["canon"].tolist())
    Xe = pd.DataFrame(X, columns=feat_names)
    Xe["SMILES"] = names
    Xe = Xe[keep.astype(bool)].reset_index(drop=True)
    ext = ext.iloc[np.where(keep)[0]].reset_index(drop=True)
    print(f"external rows: {len(ext)} (2D6 pos {(ext['CYP2D6_lab'] == 1).sum()} / "
          f"3A4 pos {(ext['CYP3A4_lab'] == 1).sum()})", flush=True)
    return ext, Xe


def main():
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    Xte = pd.read_parquet(os.path.join(CACHE, "X_test.parquet"))
    exclude = set(c for c in (canon(s) for s in Xtr["SMILES"]) if c) | \
              set(c for c in (canon(s) for s in Xte["SMILES"]) if c)
    ext, Xe = load_external(exclude)

    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    lab_by_smi = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                            emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab_by_smi.reindex(Xtr["SMILES"])

    Xbase = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    Xte_base = Xte.drop(columns=["SMILES"]).values.astype(np.float32)
    Xext = Xe.drop(columns=["SMILES"]).values.astype(np.float32)

    Btr = (Xtr[FP_COLS].values > 0).astype(np.float32)
    Bte = (Xte[FP_COLS].values > 0).astype(np.float32)
    Bex = (Xe[FP_COLS].values > 0).astype(np.float32)

    def sim(A, Bb):
        d = A @ Bb.T
        ca = (A > 0).sum(1)[:, None]
        cb = (Bb > 0).sum(1)[None, :]
        return d / np.maximum(ca + cb - d, 1e-9)

    S = sim(Btr, Btr)
    Ste = sim(Bte, Btr)
    Sex = sim(Bex, Btr)  # external vs challenge train only (same pos pool semantics)
    groups = scaffold_groups(Xtr["SMILES"].tolist())
    folds = make_folds(groups, seed=7)

    out_cv, out_test, oof_store = {}, {}, {}
    for iso in ISO:
        y = Y[f"{iso}_is_TDI"].values
        lab_mask = ~pd.isna(y)
        yl = np.zeros(len(y))
        yl[lab_mask] = y[lab_mask].astype(bool).astype(float)
        yext = ext[f"{iso}_lab"].values.astype(float)
        elab = ~np.isnan(yext)
        proba = np.full(len(y), np.nan)
        te_preds = []
        cand_all = np.where(lab_mask)[0]
        for f in range(K):
            trc = cand_all[folds[cand_all] != f]
            vac = cand_all[folds[cand_all] == f]

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

            Xf_tr = np.hstack([Xbase[trc], nnfeat(S[trc], trc)])
            Xf_va = np.hstack([Xbase[vac], nnfeat(S[vac], vac)])
            Xf_ex = np.hstack([Xext[elab], nnfeat(Sex[elab])])
            Xf_all = np.vstack([Xf_tr, Xf_ex])
            y_all = np.concatenate([yl[trc], yext[elab]])
            w_all = np.concatenate([np.ones(len(trc)), np.full(int(elab.sum()), W_EXT)])
            m = LGBMClassifier(n_estimators=1200, learning_rate=0.03, num_leaves=63,
                               min_child_samples=15, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, reg_lambda=2.0, n_jobs=-1,
                               random_state=42, verbosity=-1)
            m.fit(Xf_all, y_all, sample_weight=w_all)
            proba[vac] = m.predict_proba(Xf_va)[:, 1]
            te_preds.append(m.predict_proba(np.hstack([Xte_base, nnfeat(Ste)]))[:, 1])
            print(f"{iso} fold {f} done", flush=True)
        cand_th = np.linspace(0.05, 0.85, 81)
        yy = yl[lab_mask].astype(int)
        pv = proba[lab_mask]
        scores = [mcc(int(((pv >= t) & (yy == 1)).sum()), int(((pv >= t) & (yy == 0)).sum()),
                      int(((pv < t) & (yy == 0)).sum()), int(((pv < t) & (yy == 1)).sum()))
                  for t in cand_th]
        bi = int(np.argmax(scores))
        oof_store[iso] = proba
        out_cv[iso] = {"best_thr": float(cand_th[bi]), "oof_mcc": float(scores[bi]),
                       "pearson": float(pearsonr(yy, proba[lab_mask]).statistic),
                       "pos_rate": float(yy.mean()), "n": int(lab_mask.sum()),
                       "n_ext": int(elab.sum())}
        out_test[iso] = np.mean(te_preds, axis=0)
        print(iso, json.dumps(out_cv[iso]), flush=True)

    pd.DataFrame({"SMILES": Xte["SMILES"], **{f"{i}_proba": out_test[i] for i in ISO}}).to_csv(
        os.path.join(CACHE, f"tdi_{TAG}_test_probs.csv"), index=False)
    np.savez(os.path.join(CACHE, f"tdi_oof_{TAG}.npz"), **oof_store)
    with open(os.path.join(CACHE, f"tdi_cv_{TAG}.json"), "w") as fh:
        json.dump({"w_ext": W_EXT, "cv": out_cv}, fh, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
