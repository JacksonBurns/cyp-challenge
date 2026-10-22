#!/usr/bin/env python3
"""Direct RDKit -> AD4 PDBQT writer (no Open Babel), so protonated basic amines
keep their N_D type + positive partial charge. Open Babel neutralizes them,
which would defeat the CYP2D6 basic-amine pharmacophore.

AD4 atom-type assignment:
  H            -> HD
  C aromatic   -> A      (aromatic carbon)
  C aliphatic  -> C
  N aromatic   -> N_P    (pyridine-like, acceptor)
  N amide/nitrile/imine (multiple bond, no lone pair) -> N_T
  N amine w/ H -> N_D    (donor; covers -NH-, -NH2, and protonated -NH2+/NH3+)
  N amine no H -> N_A    (tertiary amine, acceptor)
  O            -> O_A
  S            -> S
Partial charges: Gasteiger (respects formal charge, so protonated N is +).
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def _ad4_type(atom):
    el = atom.GetSymbol()
    if el == "H":
        return "HD"
    if el == "C":
        return "A" if atom.GetIsAromatic() else "C"
    if el == "N":
        if atom.GetIsAromatic():
            return "N_P"
        if any(b.GetBondTypeAsDouble() >= 2.0 for b in atom.GetBonds()):
            return "N_T"  # amide / nitrile / imine
        return "N_D" if atom.GetTotalNumHs() > 0 else "N_A"
    if el == "O":
        return "O_A"
    if el == "S":
        return "S"
    return "C"


def mol_to_pdbqt(mol):
    """mol (with 3D + H + protonation) -> PDBQT string with Gasteiger partial
    charges; protonated N (formal charge +1) gets a positive partial charge
    override so the basic-amine electrostatics are correct."""
    try:
        Chem.AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    lines = ["TORSDOF 0"]
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms(), 1):
        pos = conf.GetAtomPosition(i - 1)
        el = atom.GetSymbol()
        name = el
        typ = _ad4_type(atom)
        q = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        # protonated basic amine: force a positive partial charge (Gasteiger
        # can give the lone-pair-rich N a negative value, which is backwards)
        if atom.GetFormalCharge() > 0:
            q = 0.4
        lines.append(
            f"ATOM  {i:5d}  {name:<2s} UNL 1    0    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  0.00  0.00    {q:+7.3f} {typ:<3s}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/jackson/cyp-challenge/src")
    from dock_features import _protonate_basic_amines
    smi = "CN1CC[C@@]2(c3ccccc3O2)C1"  # dextromethorphan
    mol = _protonate_basic_amines(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    pq = mol_to_pdbqt(mol)
    print(pq)
    print("=== N atoms (type + charge) ===")
    for l in pq.splitlines():
        if l.startswith("ATOM") and l[12:14].strip() == "N":
            print(f"  type={l.split()[-1]}  charge={l.split()[-2]}")
