"""Floor-hazard gate: (1) AUC of FLOOR-membership from chemistry block;
(2) CV gain from shrink-toward-floor on floor rows / overall ST-RAE.

If hazard AUC >= 0.7 per isoform AND shrinkage improves CV metric,
this becomes the ship candidate from the chemistry line.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402
from chem_block import block  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

CACHE = H.CACHE
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def soft_rae(y, yhat, lo, hi):
    err = np.clip(yhat - hi, 0, None) + np.clip(lo - yhat, 0, None)
    return float(np.nanmean(err))


def main():
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
    smi = H.Xtr["SMILES"].values
    recs, Pred = [], {}
    for iso in ISO:
        P = np.zeros(len(smi))
        tot = np.zeros(len(smi))
        for fam, w in W[iso]["w"].items():
            if not isinstance(w, (int, float)) or abs(w) < 1e-9:
                continue
            fo = H.raw_oof_fam(fam)
            col = iso if iso in fo.columns else f"{iso}_pIC50_direct_inhibition"
            v = fo[col].reindex(smi).values.astype(float)
            ok = ~np.isnan(v)
            P[ok] += w * v[ok]
            tot[ok] += w
        P = P / np.maximum(tot, 1e-9)
        Pred[iso] = P
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        flr = np.percentile(y[~np.isnan(y)], 1.5)
        for i, s in enumerate(smi):
            if np.isnan(y[i]):
                continue
            recs.append((s, iso, y[i], P[i], flr))
    t = pd.DataFrame(recs, columns=["SMILES", "iso", "y", "P", "flr"])
    mols = {s: __import__("rdkit.Chem", fromlist=["Chem"]).MolFromSmiles(s)
            for s in t.SMILES.unique()}
    from rdkit import Chem
    mols = {s: Chem.MolFromSmiles(s) for s in mols if s}
    BL = pd.DataFrame({s: block(mols[s]) for s in mols}).T
    bt = t.merge(BL, left_on="SMILES", right_index=True, how="left")
    bl = list(BL.columns)

    print("== floor-hazard: AUC (block -> is FLOOR) ==")
    aucs = {}
    for iso in ISO:
        sub = bt[bt.iso == iso].dropna(subset=bl)
        lab = (sub.y <= sub.flr).astype(int).values
        X = sub[bl].values.astype(float)
        if lab.sum() < 10:
            print(f" {iso}: n_floor={lab.sum()} skip")
            continue
        oof = np.zeros(len(lab))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, lab):
            clf = LGBMClassifier(n_estimators=200, learning_rate=0.06,
                                 num_leaves=15, min_child_samples=20,
                                 class_weight="balanced", verbose=-1)
            clf.fit(X[tr], lab[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(lab, oof)
        aucs[iso] = auc
        # prevalence at P>0.5
        thr = np.percentile(oof[lab == 0], 99)
        print(f" {iso} n={len(lab)} n_floor={lab.sum()} AUC={auc:.3f} "
              f"flag@fp1%={np.mean(oof > thr):.3f}")

    print("\n== shrink-to-floor post-process (OOF hazard, honest by 5-fold) ==")
    base = {iso: [] for iso in ISO}
    adj = {iso: [] for iso in ISO}
    for iso in ISO:
        sub = bt[bt.iso == iso].dropna(subset=bl)
        idx = np.where(np.isin(smi, sub.SMILES.values))[0]
        lab = (sub.y <= sub.flr).astype(int).values
        X = sub[bl].values.astype(float)
        oof = np.zeros(len(lab))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, lab):
            clf = LGBMClassifier(n_estimators=200, learning_rate=0.06,
                                 num_leaves=15, min_child_samples=20,
                                 class_weight="balanced", verbose=-1)
            clf.fit(X[tr], lab[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        P = pd.Series(sub.P.values, index=sub.SMILES).reindex(smi).values
        y = pd.Series(sub.y.values, index=sub.SMILES).reindex(smi).values
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        hz = pd.Series(oof, index=sub.SMILES).reindex(smi).values
        ok = ~np.isnan(y)
        flr = float(sub.flr.iloc[0])
        for alpha in [1.0]:
            h = np.clip(np.nan_to_num(hz, nan=0.0), 0, 1)
            Pa = P + alpha * h * (flr - P)
            base[iso].append(soft_rae(y[ok], P[ok], lo[ok], hi[ok]))
            adj[iso].append(soft_rae(y[ok], Pa[ok], lo[ok], hi[ok]))
    for iso in ISO:
        b = np.mean(base[iso])
        a = np.mean(adj[iso])
        print(f" {iso} soft_rae base={b:.4f} shrink={a:.4f} delta={a-b:+.4f}")
    mb = np.mean([np.mean(base[i]) for i in ISO])
    ma = np.mean([np.mean(adj[i]) for i in ISO])
    print(f" MACRO base={mb:.4f} shrink={ma:.4f} delta={ma-mb:+.4f}")
    print("GATE:", "PASS" if all(aucs.get(i, 0) >= 0.7 for i in ISO) else "FAIL",
          "(floor-hazard AUC >= 0.7 all isoforms)")


if __name__ == "__main__":
    main()
