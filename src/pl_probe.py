"""PL: pseudo-labeling on the 750 BLINDED test molecules (transductive self-training).

Never-tried lever (not on the dead-lever list). Protocol (honest):
  per scaffold fold f:
    1. fit ridge on train-fold labeled rows
    2. predict the 750 test molecules (they appear in NO fold, labels are self-generated)
    3. refit ridge on train-fold + test rows with pseudo-labels (confidence = model pred)
    4. predict val-fold, Pearson
Two flavors:
  PL_all   - all 750 test rows added with pseudo-labels
  PL_top   - only the test rows the fold-model ranks in the top 50% of predicted
             pIC50 magnitude (heuristic: extremes anchor the regression)
Baseline A = plain cpmed ridge (known: 0.5544/0.7058/0.3810/0.8035, macro 0.3991).

Gate: (PL - A) nested macro >= +0.003 AND no isoform delta < -0.01.

Env: mordred. Writes cache/ft_oof_pl{all,top}.csv (blend-ready) + cache/pl_probe.json.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
ALPHAS = [100.0, 1000.0, 10000.0]
K = 5


def nested_pearson(rs):
    rs = np.clip(rs, -0.999, 0.999)
    return float(np.tanh(np.mean(np.arctanh(rs))))


def fit_pred(Xtr, ytr, Xp, alphas=ALPHAS):
    sc = StandardScaler().fit(Xtr)
    r = RidgeCV(alphas=alphas)
    r.fit(sc.transform(Xtr), ytr)
    return r.predict(sc.transform(Xp))


def main():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    smi = ft["SMILES"].values
    folds = ft["fold"].values
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
    test_smi = test["SMILES"].values
    print(f"train {len(smi)}  test {len(test_smi)}", flush=True)

    # Embeddings: 6145 train + 750 test, same parquet
    emb = pd.read_parquet(os.path.join(CACHE, "emb_cpmed_all.parquet"))
    if "SMILES" in emb.columns:
        emb = emb.set_index("SMILES")
    Etr = emb.reindex(smi).values.astype(np.float64)
    Ete = emb.reindex(test_smi).values.astype(np.float64)
    Etr = np.nan_to_num(Etr)
    Ete = np.nan_to_num(Ete)
    # PCA 128 like tabpfn_probe for a conditioned linear read
    pca = PCA(n_components=128, random_state=0).fit(Etr)
    Xtr, Xte = pca.transform(Etr), pca.transform(Ete)
    print(f"Xtr {Xtr.shape}  Xte {Xte.shape}", flush=True)

    variants = {"A_plain": None, "PL_all": "all", "PL_top": "top"}
    per_iso = {v: {} for v in variants}
    oof_store = {v: pd.DataFrame(np.nan, index=ft.index, columns=ISO)
                 for v in ("PL_all", "PL_top")}

    for iso in ISO:
        y = ft[iso].values
        m = ~np.isnan(y)
        line = f"  {iso} (n={m.sum()})"
        for v, flavor in variants.items():
            oof = np.full(len(y), np.nan)
            rs = []
            for f in range(K):
                trn = m & (folds != f)
                va = m & (folds == f)
                if va.sum() < 5 or trn.sum() < 10:
                    rs.append(0.0)
                    continue
                # 1. base model on train-fold
                base = fit_pred(Xtr[trn], y[trn], Xte)
                if flavor is None:
                    pred = fit_pred(Xtr[trn], y[trn], Xtr[va])
                else:
                    keep = np.ones(len(Xte), dtype=bool)
                    if flavor == "top":
                        keep = np.abs(base - np.median(y[trn])) >= np.median(
                            np.abs(base - np.median(y[trn])))
                    Xp = np.vstack([Xtr[trn], Xte[keep]])
                    yp = np.concatenate([y[trn], base[keep]])
                    pred = fit_pred(Xp, yp, Xtr[va])
                oof[va] = pred
                rs.append(float(pearsonr(y[va], pred).statistic))
                if flavor is not None:
                    oof_store[v].loc[va, iso] = pred
            nested = nested_pearson(rs)
            per_iso[v][iso] = nested
            line += f"  | {v} {nested:.4f}"
        print(line, flush=True)

    print("\n=== NESTED MACRO R2 ===", flush=True)
    res = {}
    for v in variants:
        mac = float(np.mean([per_iso[v][i] ** 2 for i in ISO]))
        res[v] = {"nested": {i: round(per_iso[v][i], 4) for i in ISO},
                  "macro_R2": round(mac, 4)}
        row = f"{v:10s}" + "".join(f"{per_iso[v][i]:.4f}  " for i in ISO) + f"macro {mac:.4f}"
        print(row, flush=True)

    deltas = {i: per_iso["PL_all"][i] - per_iso["A_plain"][i] for i in ISO}
    deltas_t = {i: per_iso["PL_top"][i] - per_iso["A_plain"][i] for i in ISO}
    dA = res["PL_all"]["macro_R2"] - res["A_plain"]["macro_R2"]
    dT = res["PL_top"]["macro_R2"] - res["A_plain"]["macro_R2"]
    print(f"\ndeltas PL_all {deltas} macro {dA:+.4f}", flush=True)
    print(f"deltas PL_top {deltas_t} macro {dT:+.4f}", flush=True)
    gate = (dA >= 0.003 and min(deltas.values()) >= -0.01)
    gate_t = (dT >= 0.003 and min(deltas_t.values()) >= -0.01)
    print(f"GATE PL_all: {'PASS' if gate else 'NOT MET'} | PL_top: {'PASS' if gate_t else 'NOT MET'}",
          flush=True)

    for v in ("PL_all", "PL_top"):
        out = oof_store[v].fillna(0)
        out.insert(0, "SMILES", ft["SMILES"])
        out["fold"] = folds
        out.to_csv(os.path.join(CACHE, f"ft_oof_pl{'all' if v == 'PL_all' else 'top'}.csv"),
                   index=False)

    json.dump({"variants": res,
               "gate": {"PL_all": gate, "PL_top": gate_t},
               "recipe": "PCA128(cpmed); per-fold RidgeCV base -> pseudo-label test rows -> refit -> val; folds seed0 K=5"},
              open(os.path.join(CACHE, "pl_probe.json"), "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
