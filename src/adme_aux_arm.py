"""ADME auxiliary-task arm (R1 route: external-data aux, zero overlap).

Idea under test: teach the model assay-physics by auxiliary tasks --
logP (Sangster 23.5k), logD / kinetic+thermodynamic solubility (KSOL) /
microsomal CLint / Caco-2 (user expansion 5k), logS (ESOL), amide
rotational barrier dG (Xin Li 885). These datasets share ~0 molecules
with the challenge set, so transfer must come through SHARED DESCRIPTORS.

Pipeline (honest, CPU, GBM -- no GPU):
 1. compute ~208 RDKit descriptors for all external molecules (cached).
 2. per-property GBM with 5-fold CV R2 (property predictability).
 3. predict each property for ALL 6895 challenge molecules -> ADME block.
 4. GATE A: append block to the base stack; 5-fold CV MAE gain per isoform.
 5. GATE B: does the block improve HARD-molecule detection (AUC of
    |residual|>=1 from features vs base)?
Pass gate = MAE gain >= 0.005 on >=1 isoform, or HARD-AUC +0.03.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402
import featurize as FZ  # noqa: E402

CACHE = "/home/jackson/cyp-challenge/cache/adme_aux"
USER = "/home/jackson/cyp-challenge/external/user_supplied"
os.makedirs(CACHE, exist_ok=True)
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def desc_frame(smiles):
    out = []
    names = FZ._desc_names
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            out.append([0.0] * len(names))
        else:
            out.append(FZ.features_for_mol(m))
    return pd.DataFrame(out, columns=names)


def load_sources():
    frames = {}
    sang = pd.read_excel(f"{USER}/sangster_datasets.xlsx",
                         sheet_name="SangsterLogP (N=23,520)")
    c = {x.lower(): x for x in sang.columns}
    sc = [x for x in sang.columns if "smiles" in x.lower()][0]
    lc = [x for x in sang.columns if "logp" in x.lower() and "pred" not in x.lower()][0]
    frames["logP_sang"] = sang[[sc, lc]].dropna().rename(
        columns={sc: "SMILES", lc: "y"})
    exp = pd.read_csv(f"{USER}/expansion_data_train.csv")
    ec = [x for x in exp.columns if "smiles" in x.lower()][0]
    for prop in ["LogD", "KSOL", "HLM CLint", "MLM CLint",
                 "Caco-2 Permeability Papp A>B"]:
        if prop in exp.columns:
            d = exp[[ec, prop]].dropna()
            d.columns = ["SMILES", "y"]
            d["y"] = pd.to_numeric(d["y"], errors="coerce")
            frames[prop.replace(" ", "_")] = d.dropna()
    es = pd.read_csv(f"{USER}/delaney-processed.csv")
    smc = [x for x in es.columns if x.upper().startswith("SMILES")][0]
    frames["logS_esol"] = es[[smc, "measured log solubility in mols per litre"]].dropna()
    frames["logS_esol"].columns = ["SMILES", "y"]
    rot = pd.read_csv(f"{USER}/rotational_barrier_xinli.csv", encoding="utf-8-sig")
    rc = [x for x in rot.columns if "smiles" in x.lower()][0]
    gc = [x for x in rot.columns if "g(" in x.lower()][0]
    frames["rotbar_dG"] = rot[[rc, gc]].dropna()
    frames["rotbar_dG"].columns = ["SMILES", "y"]
    return frames


def numify(v):
    import re
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(v))
    return float(m.group(1)) if m else np.nan


def main():
    frames = load_sources()
    for k, f in frames.items():
        f["y"] = f["y"].map(numify)
        frames[k] = f.dropna(subset=["y"])
    # one shared descriptor matrix over union of external smiles
    allsm = sorted(set().union(*[set(f.SMILES) for f in frames.values()]))
    dpath = f"{CACHE}/ext_desc.parquet"
    if os.path.exists(dpath):
        XD = pd.read_parquet(dpath)
    else:
        XD = desc_frame(allsm)
        XD["SMILES"] = allsm
        XD.to_parquet(dpath)
    XD = XD.set_index("SMILES")
    print(f"external molecules featurized: {len(XD)}")

    prop_feats = {}
    for name, f in frames.items():
        f = f.drop_duplicates("SMILES")
        f = f[f.SMILES.isin(XD.index)]
        X = XD.loc[f.SMILES].values.astype(float)
        y = f.y.values.astype(float)
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isfinite(X), X, med)
        # GBM 5-fold CV R2
        preds = np.zeros(len(y))
        k = 5
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(X, (y > np.median(y)).astype(int)):
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                  num_leaves=31, verbose=-1)
            m.fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        ss = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - preds) ** 2) / ss
        print(f"  prop {name:24s} n={len(y):6d} CV_R2={r2:.3f}")
        # refit on all, predict challenge molecules
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                              num_leaves=31, verbose=-1)
        m.fit(X, y)
        prop_feats["adme_" + name.replace(" ", "_")] = m

    # challenge molecule descriptors: reuse base stack's RDKit portion.
    # H.Xtr columns are ~200 RDKit desc + 3072 autocorr; RDKit names match FZ._desc_names
    base_cols = [c for c in H.Xtr.columns if c != "SMILES"]
    rdcols = [c for c in FZ._desc_names if c in base_cols]
    print(f"RDKit-named cols found in stack: {len(rdcols)}/{len(FZ._desc_names)}")
    Xtr_r = H.Xtr[rdcols].values.astype(float)
    Xte_r = H.Xte[rdcols].values.astype(float)
    med = np.nanmedian(np.where(np.isfinite(Xtr_r), Xtr_r, np.nan), axis=0)
    Xtr_r = np.where(np.isfinite(Xtr_r), Xtr_r, med)
    Xte_r = np.where(np.isfinite(Xte_r), Xte_r, med)

    A_tr, A_te = [], []
    for cname, m in prop_feats.items():
        A_tr.append(m.predict(Xtr_r))
        A_te.append(m.predict(Xte_r))
    A_tr = np.column_stack(A_tr)
    np.save(f"{CACHE}/adme_feats_tr.npy", A_tr)
    np.save(f"{CACHE}/adme_feats_te.npy", np.column_stack(A_te))
    print("ADME block tr", A_tr.shape)

    # GATE A: MAE gain on base stack
    Xbase = H.Xtr[base_cols].values.astype(float)
    med2 = np.nanmedian(np.where(np.isfinite(Xbase), Xbase, np.nan), axis=0)
    Xbase = np.where(np.isfinite(Xbase), Xbase, med2)
    smi = H.Xtr["SMILES"].values
    W = __import__("json").load(open("cache/hinge_w_zoo6.json"))
    gains = {}
    for iso in ISO:
        P = np.zeros(len(smi))
        tot = 0.0
        for fam, w in W[iso]["w"].items():
            if not isinstance(w, (int, float)) or abs(w) < 1e-9:
                continue
            fo = H.raw_oof_fam(fam)
            col = iso if iso in fo.columns else f"{iso}_pIC50_direct_inhibition"
            v = fo[col].reindex(smi).values.astype(float)
            ok = ~np.isnan(v)
            P[ok] += w * v[ok]
            tot += w * ok
        P = P / np.maximum(tot, 1e-9)
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        m = ~np.isnan(y)
        y = y[m]
        Xbase_i, A_i = Xbase[m], A_tr[m]
        resid = np.abs(y - np.nan_to_num(P[m], nan=np.nanmedian(y)))
        hard = (resid >= 1.0).astype(int)
        Xa = np.hstack([Xbase_i, A_i])
        maes = []
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, te in skf.split(Xbase_i, hard):
            yv = y[tr]
            m0 = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                   num_leaves=31, verbose=-1)
            m0.fit(Xbase_i[tr], y[tr])
            e0 = mean_absolute_error(y[te], m0.predict(Xbase_i[te]))
            m1 = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                   num_leaves=31, verbose=-1)
            m1.fit(Xa[tr], y[tr])
            e1 = mean_absolute_error(y[te], m1.predict(Xa[te]))
            maes.append((e0, e1))
        e0 = np.mean([a for a, b in maes])
        e1 = np.mean([b for a, b in maes])
        gains[iso] = (e0, e1)
        print(f" GATE A {iso}: MAE base={e0:.4f} +adme={e1:.4f} gain={e0-e1:+.4f}")

    pos = [iso for iso in ISO if gains[iso][0] - gains[iso][1] >= 0.005]
    print("GATE A pass isoforms:", pos if pos else "NONE")

    # GATE B: FLOOR-membership hazard AUC (assay-physics detection)
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    skf2 = StratifiedKFold(5, shuffle=True, random_state=1)
    for iso in ISO:
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        m = ~np.isnan(y)
        yv = y[m]
        lo = np.nanpercentile(yv, 1.5)
        hb = (yv <= lo + 1e-9).astype(int)
        if hb.sum() < 10:
            continue
        Xb, Xa = Xbase[m], A_tr[m]
        aucs = []
        for tr, te in skf2.split(Xb, hb):
            lr = LogisticRegression(max_iter=2000).fit(Xb[tr], hb[tr])
            a0 = roc_auc_score(hb[te], lr.decision_function(Xb[te]))
            lr2 = LogisticRegression(max_iter=2000).fit(Xa[tr], hb[tr])
            a1 = roc_auc_score(hb[te], lr2.decision_function(Xa[te]))
            aucs.append((a0, a1))
        print(f" GATE B {iso}: floor-hazard AUC stack={np.mean([a for a,b in aucs]):.3f} "
              f"adme={np.mean([b for a,b in aucs]):.3f} n_floor={hb.sum()}")


if __name__ == "__main__":
    main()
