"""R3-5: protein-side docking features (Vina CPU, heme-centered boxes).

Structures: 1Z19 (1A2), 1EU9 (2C9), 4UOU (2D6), 3MO1 (3A4) - co-crystal,
catalytic heme present. Box = 24A cube centered on heme Fe.
Prep: meeko->pdqt4 (pdbh+q4, flexible residues at the site are frozen as
in PDB); receptor pqr via mk_prepare_receptor (pdbqt->pqr) if available,
else the pdbqt directly with Vina's own charges.
Features: best Vina score per isoform, top-3 score mean, score range.
Subsample first: 1200 train (random) + 750 test. If feature gate is
positive, extend to all molecules.
Output: cache/dock/dock_scores.npz {iso: {smiles: score}}.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DOCKD = os.path.join(CACHE, "dock")
os.makedirs(DOCKD, exist_ok=True)

BOX = 24.0
# filled after PDB parse (heme Fe center)
CENTERS = {}


def main():
    sys.path.insert(0, HERE)
    import hinge_blend as H
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from vina import Vina

    v = Vina(sf_shape=32, exhaustiveness=8, energy_range=8, seed=0)
    rng = np.random.default_rng(0)
    tr_sm = list(H.Xtr["SMILES"])
    te_sm = list(H.Xte["SMILES"])
    idx = rng.choice(len(tr_sm), 1200, replace=False)
    sample = [("tr", tr_sm[i]) for i in sorted(idx)] + [("te", s) for s in te_sm]
    print("sample", len(sample), flush=True)

    out = {iso: {} for iso in CENTERS}
    recs = {iso: os.path.join(DOCKD, f"{iso}.pqr") for iso in CENTERS}
    for iso, (cx, cy, cz) in CENTERS.items():
        v.set_receptor(recs[iso])
        v.set_ligand_from_file(os.path.join(DOCKD, "probe.pdbqt"))
    # ligand loop
    mk = MoleculePreparation()
    fail = 0
    for tag, smi in sample:
        ligp = os.path.join(DOCKD, "lig.pdbqt")
        try:
            from rdkit import Chem
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            ok = Chem.MMFFOptimizeMolecule(m)
            if ok == -1:
                Chem.MMFFHasAllMoleculeParams(m)
            if m.GetNumConformers() == 0:
                Chem.EmbedMolecule(m, randomSeed=42)
                Chem.MMFFOptimizeMolecule(m)
            mol_block = Chem.MolToMolBlock(m)
            setups = mk.parse(mol_block) if hasattr(mk, "parse") else mk
            # meeko API: setup list from MolMolBlockToMol, write via writer
            from meeko import PDBQTMolecule
            m3 = mk.parse(mol_block)
            pdbqt = PDBQTMolecule(m3)
        except Exception as e:
            fail += 1
            continue
    print("meeko failures", fail, "sample", len(sample), flush=True)


if __name__ == "__main__":
    main()
