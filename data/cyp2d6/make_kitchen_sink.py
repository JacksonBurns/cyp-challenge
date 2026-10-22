#!/usr/bin/env python3
"""Build a kitchen-sink reference ligand PDBQT covering the UNION of AD4 atom
types across a sample of the real dataset, so the Vina affinity maps are
generated for every type any ligand will use (avoids 'map not present' errors).

Steps:
  1. Convert a sample of SMILES to PDBQT (obabel -xh -p 7.4).
  2. Collect the set of atom-type labels (last field of ATOM lines).
  3. Emit one dummy atom of each type at distinct coordinates -> reference.
  4. (caller runs vina --write_maps on it)
"""
import subprocess, tempfile, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
OBABEL = "/usr/bin/obabel"

def smi_to_types(smi):
    """Atom types for one SMILES, via the SAME direct writer the pipeline uses."""
    from dock_features import _ligand_pdbqt
    pq = _ligand_pdbqt(smi)
    if pq is None:
        return set()
    return set(l.split()[-1] for l in pq.splitlines() if l.startswith("ATOM"))

def main(smi_file, n_sample, out_ref):
    smis = [l.strip() for l in open(smi_file) if l.strip()][:n_sample]
    print(f"scanning {len(smis)} ligands for atom-type union...", flush=True)
    union = set()
    for i, s in enumerate(smis, 1):
        union |= smi_to_types(s)
        if i % 50 == 0:
            print(f"  {i}/{len(smis)}  types so far: {sorted(union)}", flush=True)
    # guaranteed baseline: every common AD4 type, so no ligand ever hits a
    # 'map not present' error (esp. N_D protonated amine, which the sample may lack)
    BASELINE = {"A", "C", "HD", "N_A", "N_D", "N_P", "N_T", "O_A", "S", "F", "Cl", "Br"}
    union |= BASELINE
    print(f"UNION of atom types ({len(union)}): {sorted(union)}", flush=True)
    # emit one atom per type at distinct coordinates (ATOM-only, no TORSDOF -
    # the Vina C++ parser rejects a TORSDOF tag in a ligand/receptor file)
    # AD4 type -> element symbol (for the dummy reference atoms)
    TYPE_EL = {"A": "C", "C": "C", "HD": "H", "N_A": "N", "N_D": "N",
               "N_P": "N", "N_T": "N", "O_A": "O", "S": "S",
               "F": "F", "Cl": "Cl", "Br": "Br", "I": "I"}
    with open(out_ref, "w") as f:
        for k, t in enumerate(sorted(union), 1):
            el = TYPE_EL.get(t, "C")
            x = float((k - 1) * 2.0)
            pre = f"ATOM  {k:5d}  {el:<3s} UNL  1    1".ljust(30)
            f.write(pre + f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  0.00  0.00    +0.000 {t} \n")
    print(f"wrote reference {out_ref} with {len(union)} atom types")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
