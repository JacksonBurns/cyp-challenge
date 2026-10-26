"""Chemistry forensics part 3 (fixed SMARTS + mechanism tests).

A. Signed residuals: is HARD a LOW-side (under-prediction) phenomenon per
   isoform? mean(y - P) among hard rows.
B. Fixed reactivity/binder SMARTS -> prevalence in FLOOR vs HARD vs EASY.
C. 2D6 pharmacophore: basic-amine (pKa-like) requirement. HARD 2D6 rows
   without a basic aliphatic amine? Also quaternary-N, big-ring.
D. Label ambiguity: molecules with MULTIPLE training label sources
   (direct_inhibition vs chembl vs pubchem vs TDI_condition) whose
   labels disagree by >1 log unit -> is HARD enriched there?
E. Near-neighbor ambiguity: max-Tanimoto >=0.9 neighbors' potency spread
   (std of y among near neighbors) for HARD vs EASY rows.
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
from rdkit.Chem import rdFingerprintGenerator as rfg  # noqa: E402

RDLogger.DisableLog("rdApp.*")
CACHE = H.CACHE
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
GEN = rfg.GetMorganGenerator(radius=2, fpSize=2048)

SMARTS = {
    "tert_ammine": "N(#C)(#C)#C",
    "aliphatic_N": "[NX3;H0,H1,H2;!$(N=*);!$(N[a]);!$(N[#6]=[O,N,S])]",
    "aryl_piperazine": "c1ccccc1N1CCNCC1",
    "benzyl_amine": "[NX3][CH2]c",
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1ncncc1",
    "quinoline": "c1ccc2cnccc2c1",
    "quinazoline": "c1cc2c(ncn2[nH])nc1",
    "imidazole": "c1c[nH]cn1",
    "imidazole_any": "[c][n]1c[nH]cn1",
    "indole": "c1ccc2[nH]ccc2c1",
    "aniline": "[NX3]c",
    "amide": "[NX3][CX3](=[OX1])",
    "sulfonamide": "[NX3][SX4](=[OX1])(=[OX1])",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "tetrazole": "c1nn[nH]n1",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitrile": "C#N",
    "sulfoxide": "[SX4](=[OX1])",
    "sulfone": "[SX6](=[OX1])(=[OX1])",
    "phosphate": "P(=O)([O,F])",
    "epoxide": "C1OC1",
    "aziridine": "C1NC1",
    "aldehyde": "[CX3H1](=O)",
    "Michael_ketone": "C=CC(=O)[#6]",
    "ester": "[#6][OX2][CX3](=[OX1])",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "guanidine": "NC(=[NX2])[NX3]",
    "quaternary_N": "[N+]",
    "bigring": "[R8,R9,R10,R11,R12]",
    "spiro": "[XD4;$(*1(~*)(~*)~*1)]",
    "alkynyl": "C#C",
    "azo": "N=N",
    "hydrazine": "[NX3][NX3]",
    "halogen_Csp2": "[F,Cl,Br,I]=,a",
    "gem_dimethyl": "[CX3](C)(C)",
    "saccharide_like": "C1OC(O)C(O)C(O)C1O",
}


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

    print("== A. signed residual mean by group (neg = blend predicts HIGH)")
    for iso in ISO:
        sub = t[t.iso == iso]
        g = sub.groupby("grp").res.agg(["mean", "std", "count"])
        print(f" {iso}:", {k: (round(v['mean'], 2), int(v['count']))
                           for k, v in g.iterrows()})

    # B: SMARTS prevalence by group
    mols = {s: Chem.MolFromSmiles(s) for s in t.SMILES.unique()}
    pats = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
    hits = {}
    for s, m in mols.items():
        if m is None:
            continue
        hits[s] = {k: int(bool(m.GetSubstructMatches(p))) for k, p in pats.items() if p is not None}
    Hdf = pd.DataFrame(hits).T
    for k in pats:
        if k not in Hdf.columns:
            Hdf[k] = 0
    t2 = t.merge(Hdf, left_on="SMILES", right_index=True, how="left")
    print("\n== B. SMARTS prevalence HARD vs EASY vs FLOOR (per iso, top gaps) ==")
    cols = list(pats)
    for iso in ISO:
        sub = t2[t2.iso == iso]
        pr = sub.groupby("grp")[cols].mean()
        need = ["HARD", "EASY", "FLOOR", "MID"]
        pr = pr.reindex([g for g in need if g in pr.index])
        def col(g):
            return pr.loc[g] if g in pr.index else pd.Series(0.0, index=cols)
        hg, eg = col("HARD"), col("EASY")
        gaps = (hg - eg).abs().sort_values(ascending=False)
        line = ", ".join(f"{k} {hg[k]:.2f}v{eg[k]:.2f}" for k in gaps.index[:5])
        fg, mg = col("FLOOR"), col("MID")
        fline = ", ".join(f"{k} {fg[k]:.2f}v{mg[k]:.2f}"
                          for k in (fg - mg).abs().sort_values(ascending=False).index[:4])
        print(f" {iso} HARDvsEASY: {line}")
        print(f"      FLOORvMID:  {fline}")

    # C: 2D6 pharmacophore
    d6 = t2[t2.iso == "CYP2D6"]
    print("\n== C. CYP2D6: aliphatic_N prevalence HARD={:.2f} EASY={:.2f} "
          "FLOOR={:.2f}; quaternary HARD={:.2f} EASY={:.2f}".format(
              d6[d6.grp == "HARD"].aliphatic_N.mean(),
              d6[d6.grp == "EASY"].aliphatic_N.mean(),
              d6[d6.grp == "FLOOR"].aliphatic_N.mean(),
              d6[d6.grp == "HARD"].quaternary_N.mean(),
              d6[d6.grp == "EASY"].quaternary_N.mean()))

    # D: label ambiguity across sources
    emx = pd.read_csv(os.path.join(H.R.DATA, "cyp-challenge-TRAIN_Emax.csv"))
    emx = emx.drop_duplicates("SMILES").set_index("SMILES")
    print("\n== D. label-source disagreement (DI vs chembl/pubchem aux) ==")
    for iso in ISO:
        auxc = [c for c in emx.columns if c.startswith(f"{iso}_pIC50")
                and "conf" not in c and "direct" not in c]
        if not auxc:
            continue
        ydi = t[t.iso == iso].set_index("SMILES").y
        aux = emx[auxc[0]].reindex(ydi.index).astype(float)
        both = ydi.notna() & aux.notna()
        gap = (ydi - aux).abs()[both]
        sub = t[t.iso == iso].set_index("SMILES")
        hardm = (sub.grp == "HARD").reindex(gap.index).astype(bool)
        print(f" {iso} n_both={int(both.sum())} mean|gap|={gap.mean():.2f} "
              f"hard_frac_gap>1={ (gap>1)[hardm].mean():.2f} "
              f"easy_frac_gap>1={(gap>1)[(sub.grp=='EASY').reindex(gap.index).astype(bool)].mean():.2f}")

    # E: near-neighbor potency spread
    fps = {s: GEN.GetFingerprint(mols[s]) for s in mols if mols[s] is not None}
    print("\n== E. near-neighbor (Tanimoto>=0.9, excl self) y-spread ==")
    for iso in ISO:
        sub = t[t.iso == iso]
        by = sub.groupby("SMILES").y.first()
        act = [s for s in by.index if s in fps]
        fpsL = [fps[s] for s in act]
        res = defaultdict(list)
        for i, s in enumerate(act):
            sims = DataStructs.BulkTanimotoSimilarity(fps[s], fpsL)
            nb = [by[act[j]] for j in range(len(act))
                  if j != i and sims[j] >= 0.9]
            if len(nb) >= 2:
                g = sub[sub.SMILES == s].grp.iloc[0]
                res[g].append(np.std(nb))
        for g in ["HARD", "EASY", "MID", "FLOOR"]:
            if res[g]:
                print(f"  {iso} {g:6s} n={len(res[g]):3d} median_neighbor_std={np.median(res[g]):.2f}")


if __name__ == "__main__":
    main()
