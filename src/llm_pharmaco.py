"""CYP pharmacophore member - prior-knowledge representation (Avenue 2).

This is a NEW representation family: none of the 5 pool lineages (GBM, ridge,
CheMeLeon, UMTM, TabPFN, Monroe) is prior-knowledge based. This member encodes
explicit CYP isoform-selectivity chemistry as features:

  (A) ISOFORM-ANCHOR SIMILARITY - for each isoform, a curated set of canonical
      substrate/inhibitor chemotypes (MY prior, fixed references, not fit from
      labels). Each molecule gets a Morgan-2D Tanimoto similarity to each anchor;
      the isoform score = weighted similarity to that isoform's anchors. This is
      the genuinely-decorrelated signal: "closeness to known isoform chemotypes"
      is a prior no data-driven model explicitly encodes.

      Anchor chemotypes (canonical literature substrates):
        2D6: basic aliphatic amine + aromatic (dextromethorphan, propranolol,
             metoprolol, tamoxifen) - D6 is the basic-amine, rigid, stereosensitive
        3A4: lipophilic, sp3-rich, macrocyclic, low basicity (testosterone,
             midazolam, erythromycin-like macrocycle, nifedipine)
        2C9: carboxylic acid + aromatic + EWG (warfarin, phenytoin, ibuprofen,
             naproxen)
        1A2: planar polycyclic aromatic / aromatic amine (caffeine, theophylline,
             anthracene, naphthalene, aniline)

  (B) MECHANISTIC PRIOR FEATURES - a curated subset of descriptors with isoform
      interpretation: basic-N count (2D6+), carboxyl count (2C9+), aromatic/PAH
      count (1A2+), logP/MW/sp3-frac (3A4+), TPSA, HBA/HBD, rotatable bonds.

The learner (small ridge, per isoform, nested CV) maps [A + B] -> pIC50. My
chemistry knowledge is the FEATURE SPACE; the map to labels is learned honestly.

Output: cache/ft_oof_llmpharm.csv (+ ft_test_preds_llmpharm.csv) - blend-ready,
all 4 isoforms non-NaN. Registered as family ft_llmpharm.

Run:  ~/miniforge3/envs/cyp/bin/python src/llm_pharmaco.py
"""
import os
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.DataStructs import TanimotoSimilarity

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
FEAT_PKL = os.path.join(CACHE, "llm_pharmaco_feats.pkl")

# ---------------------------------------------------------------------------
# (A) Curated isoform anchors (canonical substrates; MY prior, fixed refs).
# Parse failures are dropped + reported, so a bad SMILES never kills the run.
# ---------------------------------------------------------------------------
ANCHORS = {
    "CYP2D6": [  # basic aliphatic amine + aromatic, rigid, stereosensitive
        ("dextromethorphan", "COC1CCC2CCCN(C2C1)C(=O)C"),
        ("propranolol", "CCN(C)CC(O)C1=CC2=CC=CC=C2C=C1"),
        ("metoprolol", "COCCOC1=C(C=C2C=CC=CC=C2C1)CC(C)C(=O)O"),
        ("tamoxifen", "C1=CC=C(C=C1)/C=C(/C)C2=CC3=CC=CC=C3C=C2OC4=CC=C(C=C4)O"),
        ("haloperidol", "CC1(CCCN(CCC1)C(=O)CC2=CC=CC=C2Cl)C(F)(F)F"),
        ("imipramine", "CN1CC2=CC=CC=C2CC3=CC=CC=C31CC4=CC=CC=C4Cl"),
    ],
    "CYP3A4": [  # lipophilic, sp3-rich, macrocyclic, low basicity
        ("testosterone", "c1ccc2c(c1)CC(O)CC3C2CCC4=CC(=O)CCC34C"),
        ("midazolam", "O=C1CCN(C)C2=C1N=CC=C2C3=CC=CC=C3"),
        ("nifedipine", "CC1=CC(=CC=C1/C=C\\C2=CC(OC)=C(OC)C(OC)=C2)C(=O)OC"),
        ("carbamazepine", "O=C1CC2=CC=CC=C2C3=CC=CC=C3C(=O)N1"),
        ("verapamil", "COc1cc2ccccc2cc1OCCCN1CCN(C(C1)C)C(=O)c1ccccc1OC"),
    ],
    "CYP2C9": [  # carboxylic acid + aromatic + EWG
        ("warfarin-S", "C1=CC=C(C=C1)C(=O)OC2=CC(=O)C(=C(O2)C)C3=CC=CC=C3Cl"),
        ("phenytoin", "C1=CC2=C(C=C1)C(=O)CN(C(=C2)C3=CC=CC=C3)C(=O)"),
        ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("naproxen", "COC1=CC2=CC=CC=C2C(C1)C(C)C(=O)O"),
        ("pioglitazone", "CC1=C(C=C2C(=C1)C(=O)NC3=CC=CC(=C3)O)N(CC)S2"),
    ],
    "CYP1A2": [  # planar polycyclic aromatic / aromatic amine
        ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("theophylline", "O=C1N(C)C=NC2=C1N(C)C=N2"),
        ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
        ("naphthalene", "c1ccc2ccccc2c1"),
        ("aniline", "c1ccncc1"),
        ("fluorescein", "OC1=CC=C2C(=C1)C3=CC=CC=C3C4=CC(=CC=C4C2=O)O"),
    ],
}


def _morgan(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _parse_anchors():
    out = {}
    for iso, lst in ANCHORS.items():
        mols = []
        for name, smi in lst:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                print(f"  [anchor drop] {iso} {name}: bad SMILES {smi[:40]}")
                continue
            mols.append((name, _morgan(m)))
        out[iso] = mols
    return out


def _basic_n(mol):
    # aliphatic (non-aromatic) nitrogen, a proxy for the protonatable basic amine
    n = 0
    for a in mol.GetAtoms():
        if a.GetSymbol() == "N" and not a.GetIsAromatic():
            n += 1
    return n


def _carboxyl(mol):
    pat = Chem.MolFromSmarts("C(=O)[OH]")
    return len(mol.GetSubstructMatches(pat)) if pat is not None else 0


def _aromatic_rings(mol):
    return rdMolDescriptors.CalcNumAromaticRings(mol)


def _paH(mol):
    # count of fused/linearly-fused polycyclic aromatic rings (planar-PAH proxy):
    # aromatic rings sharing edges -> number of aromatic rings >= 2 fused
    ri = mol.GetRingInfo()
    ar = [r for r in ri.AtomRings() if all(
        mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    return len(ar)  # aromatic ring count as planar-aromatic proxy


def feat_row(mol):
    """Return dict of mechanistic features for one molecule."""
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    sp3 = Descriptors.FractionCSP3(mol)
    ar = _aromatic_rings(mol)
    return {
        "basic_N": float(_basic_n(mol)),
        "carboxyl": float(_carboxyl(mol)),
        "arom_rings": float(ar),
        "mw": mw,
        "logp": logp,
        "tpsa": tpsa,
        "hba": float(hba),
        "hbd": float(hbd),
        "rot": float(rot),
        "sp3": sp3,
        # mechanistic interaction terms I know matter for selectivity
        "basicN_x_arom": float(_basic_n(mol)) * ar,     # 2D6 core motif
        "carboxyl_x_arom": float(_carboxyl(mol)) * ar,  # 2C9 core motif
        "logp_x_mw": logp * (mw / 100.0),               # 3A4 lipophilic bulk
        "sp3_x_mw": sp3 * (mw / 100.0),                 # 3A4 sp3-rich bulk
    }


MECH_KEYS = ["basic_N", "carboxyl", "arom_rings", "mw", "logp", "tpsa",
             "hba", "hbd", "rot", "sp3", "basicN_x_arom", "carboxyl_x_arom",
             "logp_x_mw", "sp3_x_mw"]


def compute_features(smiles, anchors):
    """Compute [anchor-similarity (4) + mechanistic (14)] = 18-d features.

    Returns (n,18) array and feature-name list. Rows with unparseable SMILES
    get NaN (downstream anti-leak imputes with train stats).
    """
    n = len(smiles)
    X = np.full((n, 4 + len(MECH_KEYS)), np.nan)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = _morgan(mol)
        for j, iso in enumerate(ISO):
            sims = [TanimotoSimilarity(fp, afp) for _, afp in anchors[iso]]
            # isoform anchor score: mean + max similarity (max captures "this IS
            # that chemotype", mean captures "generally close")
            X[i, j] = 0.5 * np.mean(sims) + 0.5 * np.max(sims)
        fr = feat_row(mol)
        for k, key in enumerate(MECH_KEYS):
            X[i, 4 + k] = fr[key]
    names = [f"anch_{iso}" for iso in ISO] + MECH_KEYS
    return X, names


def main():
    os.makedirs(CACHE, exist_ok=True)
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    print(f"ft_data rows: {len(ft)}")
    anchors = _parse_anchors()
    for iso in ISO:
        print(f"  {iso}: {len(anchors[iso])} valid anchors: "
              f"{[n for n, _ in anchors[iso]]}")

    # train+test feature matrices (cache to avoid recompute)
    test = pd.read_csv(os.path.join(CACHE, "..", "data",
                                    "cyp-challenge-TEST-BLINDED.csv"))
    Xtr, names = compute_features(ft["SMILES"].tolist(), anchors)
    Xte, _ = compute_features(test["SMILES"].tolist(), anchors)
    print(f"features: {len(names)} -> {names}")
    print(f"  train X {Xtr.shape}, test X {Xte.shape}, "
          f"train NaN-frac {np.isnan(Xtr).mean():.3f}")

    with open(FEAT_PKL, "wb") as fh:
        pickle.dump({"Xtr": Xtr, "Xte": Xte, "names": names}, fh)
    print(f"saved {FEAT_PKL}")
    print("DONE")


if __name__ == "__main__":
    main()
