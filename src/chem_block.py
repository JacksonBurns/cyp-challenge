"""Part 4: candidate CYP-mechanism descriptor block + honest lift gate.

Proposed block (the 'what to add' answer, built concretely):
 1. pKa proxies: strongest basic center (aliphatic amine count, aryl-
    piperazine, guanidine), acidic centers (COOH, tetrazole, sulfonamide,
    sulfone-imide), zwitterion flag.
 2. Isoform substrate-motif counts (MediChem-flavored, from literature):
    2D6: basic-N + aromatic ring distance features (count-based proxy);
    1A2: planar fused aromatics, flatness (n_fused_arom, ring planarity
    proxy = n_arom/nrot);
    2C9: acids, halogenated aryl ketones, aryl-acetic acids;
    3A4: large flexible hydrophobic, esters, cyclic carbonates.
 3. Reactivity alerts (covalent liability): aldehyde, Michael, epoxide,
    quinone, acyl-hydrazide, benzothiazole-thiol, haloketone.
 4. Stereo/3D-lite: n_stereocenters, n_stereo_unknown (undefined = the
    real data problem), molecular flexibility entropy proxy (nrot*logP?).

GATE: does this block predict HARD membership (AUC>0.6 per iso) and
reduce |residual| when added to GBM trained on Xtr?  Honest 5-fold CV.
"""
import os
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402
from rdkit import Chem, RDLogger, DataStructs  # noqa: E402
from rdkit.Chem import rdMolDescriptors, Descriptors, rdPartialCharges  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator as rfg  # noqa: E402
from lightgbm import LGBMRegressor, LGBMClassifier  # noqa: E402

RDLogger.DisableLog("rdApp.*")
CACHE = H.CACHE
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
GEN = rfg.GetMorganGenerator(radius=2, fpSize=2048)
DATA = H.R.DATA

MOTIFS = {
    # basic / acid centers
    "guanidine": "NC(=[NX2])[NX3]",
    "amidine": "C(=N)[NX3]",
    "tert_aliph_N": "N(#C)(#C)#C",
    "sec_aliph_N": "[NX3H1](#C)#C",
    "prim_aliph_N": "[NX3H2]#C",
    "aryl_piperazine": "c1ccccc1N1CCNCC1",
    "benzyl_N": "[NX3][CH2]c",
    "amine_2c_from_arom": "c[CH2,#6X4]N",  # rough 2D6 pharmacophore proxy
    "cooh": "[CX3](=O)[OX2H1]",
    "tetrazole": "c1nn[nH]n1",
    "sulfonamide": "[NX3][SX4](=[OX1])(=[OX1])",
    "sulfone_imide": "[SX6](=[OX1])(=[OX1])[NX3][CX3](=[OX1])",
    "phenol": "[OX2H]a",
    "acylsulfonamide": "C[SX4](=[OX1])(=[OX1])[NX3]",
    "hydroxamic": "[CX3](=[OX1])[OX2H]",
    # isoform motifs
    "naphthyl": "c1ccc2ccccc2c1",
    "biphenyl": "c1ccccc1c2ccccc2",
    "indole": "c1ccc2[nH]ccc2c1",
    "carbazole_like": "c1ccc2c(c1)[nH]c1ccccc21",
    "quinoline": "c1ccc2cnccc2c1",
    "acoflavone_like_O": "O=c1cc2ccccc2oc2ccccc12",
    "haloketone": "[#6](=[OX1])[CX3]([F,Cl,Br,I])",
    "aryl_alkyl_ketone": "c[CX3](=[OX1])[#6]",
    "arylacetic": "c[CX3H2][CX3](=[OX1])",
    "ester": "[#6][OX2][CX3](=[OX1])",
    "cyclic_carbonate": "O1C(=O)OC1",
    "lactam": "N1[CX3](=[OX1])",
    "steroid_like": "C1CC2CCC1CC3CCCC23",
    "macrocyclic": "[R7,R8,R9,R10,R11,R12]",
    "quaternary_N": "[N+]",
    "pyridyl_ketone": "c1ccncc1[CX3](=[OX1])",
    "long_alkyl": "CCCCC",
    # reactivity
    "aldehyde": "[CX3H1](=O)",
    "michael": "C=CC(=O)[#6]",
    "epoxide": "C1OC1",
    "quinone": "O=C1C=CC(=O)C=C1",
    "acyl_hydrazide": "[CX3](=O)[NX3][NX3]",
    "benzothiazole": "c1cc2nscnc2s1",
    "thiol": "[SX2H]",
    "disulfide": "[SX2][SX2]",
    "alkyl_halide": "[CX4][F,Cl,Br,I]",
    "vinyl_halide": "C=C[F,Cl,Br,I]",
    "sulfonyl_fluoride": "S(=O)(=O)F",
    "boronic": "[BX3]",
    "nitroso": "[NX2]=O",
    "azo": "N=N",
    "anilide": "c[NX3][CX3](=[OX1])",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in MOTIFS.items()}


def block(m):
    out = {f"mt_{k}": int(bool(m.GetSubstructMatches(p)))
           for k, p in PATS.items() if p is not None}
    out["n_stereo"] = len(Chem.FindMolChiralCenters(
        m, includeUnassigned=True, useLegacyImplementation=False))
    out["n_stereo_unassigned"] = sum(
        1 for a in m.GetAtoms()
        if a.GetChiralTag().name in ("CHI_UNSPECIFIED",))
    rot = rdMolDescriptors.CalcNumRotatableBonds(m)
    nar = rdMolDescriptors.CalcNumAromaticRings(m)
    out["flatness"] = nar / max(rot, 1)
    out["basic_centers"] = out["mt_guanidine"] + out["mt_amidine"] + \
        out["mt_tert_aliph_N"] + out["mt_sec_aliph_N"] + out["mt_prim_aliph_N"] + \
        out["mt_aryl_piperazine"] + out["mt_benzyl_N"]
    out["acid_centers"] = out["mt_cooh"] + out["mt_tetrazole"] + \
        out["mt_sulfonamide"] + out["mt_sulfone_imide"] + out["mt_phenol"] + \
        out["mt_acylsulfonamide"]
    out["zwit"] = int(out["basic_centers"] > 0 and out["acid_centers"] > 0)
    out["pharm2d6"] = int(out["basic_centers"] > 0 and nar > 0)
    out["reactive"] = out["mt_aldehyde"] + out["mt_michael"] + out["mt_epoxide"] + \
        out["mt_quinone"] + out["mt_acyl_hydrazide"] + out["mt_alkyl_halide"] + \
        out["mt_vinyl_halide"] + out["mt_sulfonyl_fluoride"] + out["mt_boronic"]
    return out


def main():
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
    smi = H.Xtr["SMILES"].values
    recs = []
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
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        flr = np.percentile(y[~np.isnan(y)], 1.5)
        for i, s in enumerate(smi):
            if np.isnan(y[i]):
                continue
            r = y[i] - P[i]
            grp = ("FLOOR" if y[i] <= flr else "HARD" if abs(r) >= 1.0
                   else "EASY" if abs(r) <= 0.25 else "MID")
            recs.append((s, iso, y[i], P[i], r, grp))
    t = pd.DataFrame(recs, columns=["SMILES", "iso", "y", "P", "res", "grp"])

    mols = {s: Chem.MolFromSmiles(s) for s in t.SMILES.unique()}
    BL = pd.DataFrame({s: block(mols[s]) for s in mols if mols[s] is not None}).T
    bt = t.merge(BL, left_on="SMILES", right_index=True, how="left")
    bl = [c for c in BL.columns]

    # prevalence of motifs in FLOOR vs HARD vs EASY per iso
    print("== motif prevalence FLOOR / HARD / EASY / MID (iso: top HARD-vs-EASY gaps) ==")
    for iso in ISO:
        sub = bt[bt.iso == iso]
        pr = sub.groupby("grp")[bl].mean().fillna(0)
        need = ["HARD", "EASY", "FLOOR", "MID"]
        pr = pr.reindex([g for g in need if g in pr.index])
        def col(g):
            return pr.loc[g] if g in pr.index else pd.Series(0.0, index=bl)
        hg, eg = col("HARD"), col("EASY")
        gaps = (hg - eg).abs().sort_values(ascending=False)
        print(f" {iso}: " + ", ".join(
            f"{k} {hg[k]:.2f}/{eg[k]:.2f}" for k in gaps.index[:6]))
        fg, mg = col("FLOOR"), col("MID")
        print("   FLOOR top: " + ", ".join(
            f"{k} {fg[k]:.2f}/{mg[k]:.2f}"
            for k in (fg - mg).abs().sort_values(ascending=False).index[:4]))

    # GATE A: HARD-membership AUC from block alone (5-fold)
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    print("\n== GATE A: HARD AUC, block-only (5-fold) ==")
    for iso in ISO:
        sub = bt[bt.iso == iso].dropna(subset=bl)
        lab = (sub.grp == "HARD").astype(int).values
        X = sub[bl].values.astype(float)
        if lab.sum() < 20:
            continue
        oof = np.zeros(len(lab))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, lab):
            clf = LGBMClassifier(n_estimators=250, learning_rate=0.06,
                                 num_leaves=15, min_child_samples=30, verbose=-1)
            clf.fit(X[tr], lab[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        print(f" {iso} n={len(lab)} hard={lab.sum()} AUC={roc_auc_score(lab, oof):.3f}")

    # GATE B: does block reduce GBM residual on y? (Xtr vs Xtr+block, 5-fold)
    print("\n== GATE B: GBM y-prediction MAE, Xtr vs Xtr+block (5-fold) ==")
    Xbase = H.Xtr[[c for c in H.Xtr.columns if c != "SMILES"]].values.astype(float)
    med = np.nanmedian(np.where(np.isfinite(Xbase), Xbase, np.nan), axis=0)
    Xbase = np.where(np.isfinite(Xbase), Xbase, med)
    for iso in ISO:
        sub = bt[bt.iso == iso].dropna(subset=bl)
        idx = np.where(np.isin(smi, sub.SMILES.values))[0]
        yv = pd.Series(sub.y.values, index=sub.SMILES).reindex(smi[idx]).values
        B = pd.DataFrame(sub[bl].values, index=sub.SMILES).reindex(smi[idx]).values.astype(float)
        B = np.nan_to_num(B, nan=0.0)
        Xa = np.hstack([Xbase[idx], B])
        maes = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xa, (yv > np.median(yv)).astype(int)):
            m0 = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
            m0.fit(Xbase[idx][tr], yv[tr])
            e0 = np.mean(np.abs(yv[te] - m0.predict(Xbase[idx][te])))
            m1 = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
            m1.fit(Xa[tr], yv[tr])
            e1 = np.mean(np.abs(yv[te] - m1.predict(Xa[te])))
            maes.append((e0, e1))
        e0 = np.mean([m[0] for m in maes]); e1 = np.mean([m[1] for m in maes])
        print(f" {iso} MAE base={e0:.3f} +block={e1:.3f} delta={e1-e0:+.3f}")


if __name__ == "__main__":
    main()
