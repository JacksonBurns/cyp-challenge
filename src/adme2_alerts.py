"""Assay-interference alert features (SMARTS counts) for the ADME2 block.

Motivated by chemistry forensics: floor rows are hydroxamic-acid/acid-rich,
aggregation-prone planars; assay bottoms are often precipitation/aggregation/
reactive-depletion phenomena, not target pharmacology. Counts of known
assay-interference + reactivity alerts as ~25 features for all 6895 mols.
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
BASE = "/home/jackson/cyp-challenge"
CACHE = os.path.join(BASE, "cache")
OUTD = os.path.join(CACHE, "adme2")

SMARTS = {
    "pains_hydroxamate": "[$(C(=O)N[OX2H])]",
    "pains_catechol": "c1cc(O)c(O)cc1O",
    "pains_rhodanine": "O=C1NC(=O)C(=C)SC1",
    "pains_barbitur": "O=C1NC(=O)NC(=O)C1",
    "pains_quinone": "O=C1C=CC(=O)C=C1",
    "pains_michael": "[$(C=CC(=O)[N,O,S,C])]",
    "pains_benzoquinone_n": "N=CC1=CC(=O)C=CC1",
    "reactive_acrolein": "C=CC=O",
    "reactive_sulfonyl": "S(=O)(=O)[O,N]",
    "reactive_isocyanate": "N=C=O",
    "reactive_epoxide": "C1OC1",
    "reactive_aziridine": "C1CC1",
    "reactive_acidhalide": "C(=O)Cl",
    "reactive_anhydride": "C(=O)OC(=O)",
    "reactive_disulfide": "SS",
    "reactive_thiol": "[SH]",
    "agg_nitro_aromatic": "[N+](=O)[O-]",
    "agg_azo": "N=N",
    "agg_polycyclic": "a1aa2aaa(aa1)aaa2",
    "dye_triazenyl": "N=NN",
    "hydrazide": "NNC(=O)",
    "semicarb": "NNC(=O)N",
    "big_planar4": "a1a2aaa3aaaa(a2a(a1)3)",
    "logP_heavy_hal": "[$(C(F)(F)F)]",
    "quat_ammonium": "[N+](C)(C)C",
    "sulfonium": "[S+]",
    "n_oxide": "[N+]([O-])",
    "nitroso": "N=O",
    "acyl_hydrazine_thio": "NNC(=S)",
}


def main():
    import hinge_blend as H
    sys.path.insert(0, "/home/jackson/cyp-challenge/src")
    pats = {k: Chem.MolFromSmarts(v.split(" | ")[0].strip())
            for k, v in SMARTS.items()}
    bad = [k for k, v in pats.items() if v is None]
    print("bad smarts:", bad)
    pats = {k: v for k, v in pats.items() if v is not None}

    def block(smi_list):
        rows = []
        for s in smi_list:
            m = Chem.MolFromSmiles(s)
            if m is None:
                rows.append([np.nan] * len(pats))
            else:
                rows.append([float(len(m.GetSubstructMatches(p)))
                             for p in pats.values()])
        return np.array(rows, dtype=np.float32)

    Atr = block(list(H.Xtr["SMILES"].values))
    te = pd.read_csv(os.path.join(BASE, "data/cyp-challenge-TEST-BLINDED.csv"))
    Ate = block(list(te["SMILES"].values))
    np.save(os.path.join(OUTD, "alerts_tr.npy"), Atr)
    np.save(os.path.join(OUTD, "alerts_te.npy"), Ate)
    open(os.path.join(OUTD, "alerts_names.txt"), "w").write(
        "\n".join(pats.keys()))
    print("alerts block", Atr.shape, "prevalence top:",
          {k: round(float(np.nanmean(Atr[:, j])), 3)
           for j, k in enumerate(pats.keys())
           if np.nanmean(Atr[:, j]) > 0.02})


if __name__ == "__main__":
    main()
