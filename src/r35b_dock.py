"""R3-5b: Vina CPU docking of 1200 train sample + 750 test into 4 CYPs.

Ligands: RDKit ETKDG+MMFF conformer -> meeko PDBQT (flexible).
Receptors: cache/dock/{iso}.pdbqt, box 24 A centered on heme Fe.
Scores: best pose Vina score. Checkpoint JSON; resumable; 12 workers.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
DOCKD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "dock")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CENTERS = json.load(open(os.path.join(DOCKD, "dock_centers.json")))
OUT = os.path.join(DOCKD, "dock_scores.json")
N_TR = 1200


def lig_pdbqt(smi, tag):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    if AllChem.MMFFHasAllMoleculeParams(m):
        AllChem.EmbedMolecule(m, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    else:
        AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    sdf = os.path.join(DOCKD, f"tmp_{tag}.sdf")
    Chem.MolToMolFile(m, sdf)
    import subprocess
    out = sdf.replace(".sdf", ".pdbqt")
    r = subprocess.run(
        [os.path.expanduser("~/miniforge3/envs/dock/bin/mk_prepare_ligand.py"),
         "-i", sdf, "-o", out], capture_output=True)
    os.remove(sdf)
    if r.returncode != 0 or not os.path.exists(out):
        return None
    txt = open(out).read()
    os.remove(out)
    return txt


def dock_one(args):
    tag, smi = args
    from vina import Vina
    txt = lig_pdbqt(smi, tag)
    if txt is None:
        return tag, smi, {}
    res = {}
    ligf = os.path.join(DOCKD, f"lig_{tag}.pdbqt")
    open(ligf, "w").write(txt)
    for iso in ISO:
        c = CENTERS[iso]["center"]
        v = Vina(sf_name="vina", cpu=1, seed=0)
        try:
            v.set_receptor(os.path.join(DOCKD, f"{iso}.rigid.pdbqt"))
            v.set_ligand_from_file(ligf)
            v.compute_vina_maps(center=c, box_size=[24, 24, 24])
            v.dock(exhaustiveness=8, n_poses=3)
            res[iso] = float(v.score()[0])
        except Exception:
            res[iso] = None
    os.remove(ligf)
    return tag, smi, res


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import hinge_blend as H
    rng = np.random.default_rng(0)
    tr = list(H.Xtr["SMILES"])
    te = list(H.Xte["SMILES"])
    idx = sorted(rng.choice(len(tr), N_TR, replace=False))
    todo = [("tr", tr[i]) for i in idx] + [("te", s) for s in te]
    done = {}
    if os.path.exists(OUT):
        done = json.load(open(OUT))
    todo = [t for t in todo if t[1] not in done]
    print("todo", len(todo), "cached", len(done), flush=True)
    with Pool(12) as p:
        for i, (tag, smi, res) in enumerate(p.imap_unordered(dock_one, todo, chunksize=8)):
            done[smi] = res
            if i % 100 == 0:
                json.dump(done, open(OUT, "w"))
                print(f"  {i}/{len(todo)}", flush=True)
    json.dump(done, open(OUT, "w"))
    print("DONE", len(done), flush=True)


if __name__ == "__main__":
    main()
