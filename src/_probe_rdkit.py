"""Quick RDKit capability probe for the pharmacophore member (v2, simple SMARTS)."""
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
import numpy as np

m = Chem.MolFromSmiles("CCN(C)Cc1ccc(cc1)C(=O)O")  # propranolol-like, basic
# simple basic-amine patterns
for name, smi in [
    ("aliphatic_N", "N"),
    ("basic_amine_NH", "[NH1,NH2]"),
    ("aromatic_N", "n"),
    ("tertiary_N", "N(C)(C)C"),
]:
    pat = Chem.MolFromSmarts(smi)
    n = len(m.GetSubstructMatches(pat)) if pat is not None else -1
    print(f"{name:14s} ({smi}): {n}")

print("FractionCSP3:", Descriptors.FractionCSP3(m))
print("HBA:", rdMolDescriptors.CalcNumHBA(m), "HBD:", rdMolDescriptors.CalcNumHBD(m))
print("RotBonds:", rdMolDescriptors.CalcNumRotatableBonds(m))
print("MolLogP:", Descriptors.MolLogP(m), "TPSA:", Descriptors.TPSA(m))
print("NumAromaticHeterocycles:", rdMolDescriptors.CalcNumAromaticHeterocycles(m))
print("NumHAcceptors:", rdMolDescriptors.CalcNumHBA(m))

# 3D conformer (needed for the 3D avenue)
m2 = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1O"))
cids = AllChem.EmbedMolecule(m2, randomSeed=7)
print("3D embed:", "ok" if cids == 0 else "fail", "atoms:", m2.GetNumAtoms())

# Morgan fingerprint (for exemplar similarity)
fp = Chem.AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
print("Morgan2048 ok, bits:", fp.GetNumOnBits())
print("DONE")
