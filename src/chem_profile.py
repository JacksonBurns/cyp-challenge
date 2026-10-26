"""Chemistry forensics: who falls at the LLOD floor, who is hard to predict?

Groups per isoform (train rows with direct_inhibition labels):
 - FLOOR: y <= 1.10  (assay bottomed out at the global reporting floor
   ~1.03 pIC50 = 10 nM -> censored, structure-poor measurements)
 - HARD:  |y - P_blend| >= 1.0 log unit (validated zoo6 OOF blend)
 - EASY:  |y - P_blend| <= 0.25
 - MID:   everything else
Outputs cache/chem_groups.csv: per-row per-iso flags + RDKit profile, so
group comparison and descriptor-lift tests reuse one extraction.
"""
import os
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator as rfg  # noqa: E402

RDLogger.DisableLog("rdApp.*")
CACHE = H.CACHE
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

# ---- RDKit profile (the axes we will interpret on) ----
def profile(m):
    if m is None:
        return {}
    nN = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N")
    basic_n = 0
    for a in m.GetAtoms():
        if a.GetSymbol() == "N" and a.GetHybridization().name in ("SP3", "SP2") \
           and not a.GetIsAromatic() and a.GetTotalValence() == 3:
            basic_n += 1
    qn = 0
    for a in m.GetAtoms():
        if a.GetFormalCharge() >= 1:
            qn = max(qn, 1)
    return dict(
        mw=Descriptors.MolWt(m),
        logp=Crippen.MolLogP(m),
        tpsa=rdMolDescriptors.CalcTPSA(m),
        nrot=rdMolDescriptors.CalcNumRotatableBonds(m),
        nring=rdMolDescriptors.CalcNumRings(m),
        narom=rdMolDescriptors.CalcNumAromaticRings(m),
        nN=nN,
        basic_n=basic_n,
        has_basic=1 if basic_n > 0 else 0,
        frac_csp3=rdMolDescriptors.CalcFractionCSP3(m),
        nhalo=sum(1 for a in m.GetAtoms() if a.GetHalogen() in (True,)) if False
                else sum(1 for a in m.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53)),
        chg=qn,
        nheavy=m.GetNumHeavyAtoms(),
    )


# Structural-alert SMARTS: known CYP binder motifs + promiscuity/reactivity
# liabilities (PAINS / reactivity-flavored, curated for CYP work).
ALERTS = {
    "tertiary_amine": "[NX3;H0,1,2;!$(N[#6]=[O,N,S])]"
                      "$(N([#6])([#6])[#6])",
    "benzylamine": "[NX3][CH2]c",
    "pyridine": "c1ccncc1",
    "quinazoline": "c1cc2c(ncn2...)nc1",
    "imidazole": "c1c[nH]cn1|c1cnc[nH]1",
    "indole": "c1ccc2[nH]ccc2c1",
    "aniline": "[NX3]c",
    "amide": "[NX3][CX3](=[OX1])",
    "sulfonamide": "[NX3][SX4](=[OX1])(=[OX1])",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "epoxide": "C1OC1",
    "Michael_acceptor": "C=CC(=O)...[1,2]...X",
    "aldehyde": "[CX3H1](=O)",
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "isothiocyanate": "N=C=S",
    "acyl_hydrazide": "C(=O)NN",
    "nitroso": "N=O",
    "aromatic_noN": None,  # placeholder, skipped
}
BASIC_SMARTS = "[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N[a]);!$(N[+]);!$([N+]);!$(N=*)]"
AROM_N = "[n]"
COOH = "[CX3](=O)[OX2H1]"
SULF = "[SX4](=[OX1])(=[OX1])[OX2H1]"


def alerts(m):
    out = {}
    for name, sma in ALERTS.items():
        if not sma:
            continue
        try:
            out[f"al_{name}"] = 1 if m.GetSubstructMatches(
                Chem.MolFromSmarts(sma)) else 0
        except Exception:
            out[f"al_{name}"] = 0
    out["basic_amine"] = 1 if m.GetSubstructMatches(
        Chem.MolFromSmarts(BASIC_SMARTS)) else 0
    out["aromN"] = 1 if m.GetSubstructMatches(
        Chem.MolFromSmarts(AROM_N)) else 0
    out["cooh"] = 1 if m.GetSubstructMatches(
        Chem.MolFromSmarts(COOH)) else 0
    out["sulfone_acid"] = 1 if m.GetSubstructMatches(
        Chem.MolFromSmarts(SULF)) else 0
    return out


def main():
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
    smi = H.Xtr["SMILES"].values
    FLR = {}
    for iso in ISO:
        yv = H.targets[f"{iso}_pIC50_direct_inhibition"].values.astype(float)
        lab = yv[~np.isnan(yv)]
        FLR[iso] = float(np.percentile(lab, 1.5))
    print("floor cut per iso:", {k: round(v, 2) for k, v in FLR.items()})
    rows = []
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
        for i, s in enumerate(smi):
            if np.isnan(y[i]):
                continue
            r = abs(y[i] - P[i])
            grp = ("FLOOR" if y[i] <= FLR[iso] else
                   "HARD" if r >= 1.0 else
                   "EASY" if r <= 0.25 else "MID")
            rows.append((s, iso, y[i], P[i], r, grp))
    g = pd.DataFrame(rows, columns=["SMILES", "iso", "y", "P", "absres", "grp"])
    print("group counts per iso:")
    print(g.groupby(["iso", "grp"]).size().unstack(fill_value=0))

    # molecule-level: FLOOR in >=2 isoforms = pan-LLOD;
    # HARD in >=2 = pan-hard; EASY everywhere = easy.
    fl = g[g.grp == "FLOOR"].groupby("SMILES").size()
    hd = g[g.grp == "HARD"].groupby("SMILES").size()
    ea = g[g.grp == "EASY"].groupby("SMILES").size()
    allsmi = g["SMILES"].unique()
    labels = []
    for s in allsmi:
        nfl, nhd = int(fl.get(s, 0)), int(hd.get(s, 0))
        if nfl >= 2:
            lab = "PAN_FLOOR"
        elif nhd >= 2:
            lab = "PAN_HARD"
        elif nfl == 1:
            lab = "ISO_FLOOR"
        elif nhd == 1:
            lab = "ISO_HARD"
        elif ea.get(s, 0) >= 3:
            lab = "EASY"
        else:
            lab = "MID"
        labels.append(lab)
    mol_lab = pd.Series(labels, index=allsmi, name="molgrp")
    print("molecule-level labels:")
    print(mol_lab.value_counts())

    mols = {s: Chem.MolFromSmiles(s) for s in allsmi}
    prof = pd.DataFrame({s: {**profile(mols[s]), **alerts(mols[s])}
                         for s in allsmi if mols[s] is not None}).T
    prof = prof.join(mol_lab)
    prof.to_csv(os.path.join(CACHE, "chem_groups.csv"))
    print("profiled", prof.shape)

    num = prof.select_dtypes(include=[np.number]).columns
    print("\n== molecule-level group means ==")
    order = ["PAN_FLOOR", "ISO_FLOOR", "PAN_HARD", "ISO_HARD", "MID", "EASY"]
    print(prof.groupby("molgrp")[num].mean().loc[
        [g for g in order if g in prof["molgrp"].unique()]].round(2).T)


if __name__ == "__main__":
    main()
