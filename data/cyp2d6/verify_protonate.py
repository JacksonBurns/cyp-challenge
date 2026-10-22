#!/usr/bin/env python3
"""Force-protonate basic aliphatic amines to their pH-7.4 state (N_D in AD4).

Open Babel's `-p 7.4` does NOT reliably protonate aliphatic amines (they stay
NA/neutral). The CYP2D6 pharmacophore is the PROTONATED basic amine, so we
protonate explicitly in RDKit: any aliphatic (non-aromatic) N that is not an
amide and not a nitrile and has a lone pair (<=3 heavy neighbors) gets +1 charge
+ an explicit H. This is the standard basic-amine heuristic.

Verify:
  python verify_protonate.py
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def is_basic_amine_n(atom):
    if atom.GetSymbol() != "N":
        return False
    if atom.GetIsAromatic():
        return False  # pyridine-like: neutral at 7.4
    # a basic amine N is ALWAYS single-bonded. Any double/triple bond
    # (nitrile C#N, imine C=N, azide) disqualifies it.
    for b in atom.GetBonds():
        if b.GetBondTypeAsDouble() >= 2.0:
            return False
    # amide: N bonded to a carbonyl C (C(=O))
    for b in atom.GetBonds():
        other = b.GetOtherAtom(atom)
        if other.GetSymbol() == "C":
            for b2 in other.GetBonds():
                o2 = b2.GetOtherAtom(other)
                if o2.GetSymbol() == "O" and b2.GetBondTypeAsDouble() >= 2.0:
                    return False  # amide N: not basic
    # must have a lone pair: <= 3 heavy-atom neighbors
    heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() != "H")
    if heavy_neighbors > 3:
        return False
    return True


def protonate_basic_amines(mol):
    """Return a copy of mol with basic aliphatic amines protonated (+1, +H)."""
    mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
    mol = Chem.AddHs(mol)
    em = Chem.RWMol(mol)
    n_protonated = 0
    # snapshot: identify basic-amine N indices BEFORE mutating
    targets = [a.GetIdx() for a in em.GetAtoms()
               if is_basic_amine_n(a) and a.GetFormalCharge() == 0]
    for idx in targets:
        atom = em.GetAtomWithIdx(idx)
        hidx = em.AddAtom(Chem.Atom(1))
        em.AddBond(idx, hidx, Chem.BondType.SINGLE)
        atom.SetFormalCharge(1)
        n_protonated += 1
    out = em.GetMol()
    Chem.SanitizeMol(out)
    return out, n_protonated


def test():
    cases = {
        "metoprolol (basic 2nd-amine, pKa~9.4)": "CNCCc1ccc(CC(O)CNC(C)C)cc1",
        "dextromethorphan (tertiary amine)": "CN1CC[C@@]2(c3ccccc3O2)C1",
        "haloperidol (tertiary amine)": "CC1(CN(C)C)C(c2ccccc2Cl)CC(c3ccc(O)cc3F)N1C(=O)C",
        "acetamide (amide N, NOT basic)": "CC(=O)N",
        "acetonitrile (nitrile, NOT basic)": "CC#N",
        "pyridine (aromatic N, NOT basic at 7.4)": "c1ccncc1",
        "trigonal N in guanidine-ish": "NC(N)=N",
    }
    for name, smi in cases.items():
        mol, n = protonate_basic_amines(smi)
        charges = [a.GetFormalCharge() for a in mol.GetAtoms() if a.GetSymbol() == "N"]
        print(f"{name:45s} -> {n} protonated, N charges={charges}")


if __name__ == "__main__":
    test()
