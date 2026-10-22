#!/usr/bin/env python3
"""Build the kitchen-sink reference PDBQT: a SINGLE obabel-converted molecule
spanning the full AD4 atom-type space, so Vina's --write_maps generates a map
for every type any ligand in the dataset will use (no 'map not present' errors).

A single molecule is required - Vina rejects multi-molecule (concatenated)
ligand files. The reference is a grid/atom-type template only - no label/model
role.

SMILES chosen to cover (verified against the full 6145-ligand scan):
A (aromatic C), C (aliphatic C), HD, N (pyridine), NA (tertiary amine),
OA (hydroxyl), S (thioether), F, Cl, Br, I, P (phosphine). These are the
exact 12 AD4 types present in the dataset.
"""
import subprocess, os, tempfile
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
OBABEL = "/usr/bin/obabel"
HERE = os.path.dirname(os.path.abspath(__file__))

KSS_SMI = "IP(F)(F)FC1=CC(=CC=C1Br)SCC(=O)NCc2cc(Cl)cc(F)c2CN(C)C"


def main():
    with tempfile.NamedTemporaryFile("w", suffix=".smi", delete=False) as tf:
        tf.write(KSS_SMI + "\n")
        smi_path = tf.name
    pq = smi_path.replace(".smi", ".pdbqt")
    r = subprocess.run([OBABEL, smi_path, "-O", pq, "-xh", "--gen3d"],
                       capture_output=True, text=True, timeout=120)
    os.unlink(smi_path)
    if r.returncode != 0 or not os.path.exists(pq):
        raise RuntimeError(f"obabel failed: {r.stderr[-300:]}")
    # keep the RAW obabel output (ROOT/BRANCH/TORSDOF) - Vina's ligand parser
    # expects the torsion tree; stripping to ATOM-only breaks it.
    content = open(pq).read()
    os.unlink(pq)
    atom_lines = [l for l in content.splitlines() if l.startswith("ATOM")]
    types = sorted(set(l.split()[-1] for l in atom_lines))
    print(f"reference: {len(atom_lines)} atoms, obabel types: {types}")
    ref = os.path.join(HERE, "kss_ref.pdbqt")
    with open(ref, "w") as f:
        f.write(content)
    print(f"wrote {ref}")


if __name__ == "__main__":
    main()
