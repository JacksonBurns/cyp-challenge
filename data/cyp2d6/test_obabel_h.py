#!/usr/bin/env python3
"""Definitive test: does obabel preserve explicit H from a molblock?"""
import subprocess, tempfile, os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
OBABEL = "/usr/bin/obabel"

smi = "CN1CC[C@@]2(c3ccccc3O2)C1"  # dextromethorphan: 13 heavy + 13 H = 26
mol = Chem.MolFromSmiles(smi)
mol_h = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol_h, randomSeed=0)
AllChem.MMFFOptimizeMolecule(mol_h)
print(f"mol with H: {mol_h.GetNumAtoms()} atoms")
mb = Chem.MolToMolBlock(mol_h)
n_molblock_atoms = int(mb.splitlines()[3].split()[0])
print(f"molblock declares: {n_molblock_atoms} atoms")

tf = tempfile.NamedTemporaryFile("w", suffix=".mol", delete=False)
tf.write(mb); tf.close()

for flags in [["-xh"], ["-x"], ["--nohydrogens", "-x"], ["-xh", "--gen3d"]]:
    pq = tf.name.replace(".mol", ".pdbqt")
    if os.path.exists(pq):
        os.unlink(pq)
    r = subprocess.run([OBABEL, tf.name, "-O", pq] + flags,
                       capture_output=True, text=True, timeout=60)
    if os.path.exists(pq):
        n = sum(1 for l in open(pq) if l.startswith("ATOM"))
        n_h = sum(1 for l in open(pq) if l.startswith("ATOM") and l[77:79].strip() == "H")
        print(f"obabel {' '.join(flags):25s} -> {n:3d} atoms ({n_h} H)  rc={r.returncode}")
        os.unlink(pq)
    else:
        print(f"obabel {' '.join(flags):25s} -> FAILED rc={r.returncode} {r.stderr[-80:]}")
os.unlink(tf.name)
