"""R3-5b v2: receptor-pinned Vina docking (maps built ONCE per worker).

Phase 1: prepare ligand PDBQT (12 workers), cached per SMILES -> lig/<md5>.pdbqt
Phase 2: per receptor (4 procs x 2 workers): load maps once, stream ligands,
         checkpoint json per receptor.
Usage: python r35b_dock.py [prep|dock CYP1A2|done]
"""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
DOCKD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "dock")
LIGD = os.path.join(DOCKD, "lig")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CENTERS = json.load(open(os.path.join(DOCKD, "dock_centers.json")))
os.makedirs(LIGD, exist_ok=True)


def uid(smi):
    return hashlib.md5(smi.encode()).hexdigest()[:14]


def prep_one(smi):
    p = os.path.join(LIGD, uid(smi) + ".pdbqt")
    if os.path.exists(p):
        return smi, True
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi, False
    m = Chem.AddHs(m)
    ok = AllChem.EmbedMolecule(m, randomSeed=42, useRandomCoords=True)
    if ok != 0:
        return smi, False
    AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    sdf = os.path.join(LIGD, f"_t_{uid(smi)}.sdf")
    Chem.MolToMolFile(m, sdf)
    import subprocess
    r = subprocess.run(
        [os.path.expanduser("~/miniforge3/envs/dock/bin/mk_prepare_ligand.py"),
         "-i", sdf, "-o", p], capture_output=True)
    if os.path.exists(sdf):
        os.remove(sdf)
    return smi, (r.returncode == 0 and os.path.exists(p))


def dock_worker(args):
    iso, smis = args
    from vina import Vina
    c = CENTERS[iso]["center"]
    v = Vina(sf_name="vina", cpu=1, seed=0, verbosity=0)
    v.set_receptor(os.path.join(DOCKD, f"{iso}.rigid.pdbqt"))
    v.compute_vina_maps(center=c, box_size=[24, 24, 24])
    out = {}
    for smi in smis:
        p = os.path.join(LIGD, uid(smi) + ".pdbqt")
        if not os.path.exists(p):
            out[smi] = None
            continue
        try:
            v.set_ligand_from_file(p)
            v.dock(exhaustiveness=8, n_poses=1)
            out[smi] = float(v.score()[0])
        except Exception:
            out[smi] = None
    return iso, out


def all_mols():
    mol = json.load(open(os.path.join(DOCKD, "mols.json")))
    return mol["tr"] + mol["te"]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "prep"
    if mode == "prep":
        smis = all_mols()
        todo = [s for s in smis if not os.path.exists(os.path.join(LIGD, uid(s) + ".pdbqt"))]
        print("prep todo", len(todo), flush=True)
        with Pool(12) as p:
            for i, (smi, ok) in enumerate(p.imap(prep_one, todo, chunksize=16)):
                if i % 200 == 0:
                    print(f"  prep {i}/{len(todo)}", flush=True)
        print("PREP DONE", flush=True)
    elif mode == "dock":
        iso = sys.argv[2]
        cp = os.path.join(DOCKD, f"dock_{iso}.json")
        done = json.load(open(cp)) if os.path.exists(cp) else {}
        smis = [s for s in all_mols() if s not in done]
        print(iso, "dock todo", len(smis), flush=True)
        chunks = [smis[i::2] for i in range(2)]  # 2 workers
        with Pool(2) as p:
            for j, (isox, out) in enumerate(p.imap_unordered(dock_worker, [(iso, c) for c in chunks])):
                done.update(out)
                json.dump(done, open(cp, "w"))
                print(f"  {iso} chunk done {len(done)}/{len(all_mols())}", flush=True)
        print("DOCK DONE", iso, len(done), flush=True)


if __name__ == "__main__":
    main()
