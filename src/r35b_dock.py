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
    iso, wid, smis = args
    from vina import Vina
    c = CENTERS[iso]["center"]
    v = Vina(sf_name="vina", cpu=1, seed=0, verbosity=0)
    v.set_receptor(os.path.join(DOCKD, f"{iso}.rigid.pdbqt"))
    v.compute_vina_maps(center=c, box_size=[24, 24, 24])
    out = {}
    cp = os.path.join(DOCKD, f"dock_{iso}_w{wid}.json")
    for i, smi in enumerate(smis):
        p_ = os.path.join(LIGD, uid(smi) + ".pdbqt")
        if not os.path.exists(p_):
            out[smi] = None
            continue
        try:
            v.set_ligand_from_file(p_)
            v.dock(exhaustiveness=8, n_poses=1)
            out[smi] = float(v.score()[0])
        except Exception:
            out[smi] = None
        if i % 100 == 99:
            done_ = json.load(open(cp)) if os.path.exists(cp) else {}
            done_.update(out)
            json.dump(done_, open(cp, "w"))
            print(f"  {iso} w{wid} {len(done_)}", flush=True)
            out = {}
    done_ = json.load(open(cp)) if os.path.exists(cp) else {}
    done_.update(out)
    json.dump(done_, open(cp, "w"))
    return iso, wid


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
    elif mode.startswith("dock"):
        iso = sys.argv[2] if mode == "dock" else "ALL"
        isos = ISO if iso == "ALL" else [iso]
        allsm = all_mols()
        NW = 12 // len(isos)
        jobs = []
        for iz in isos:
            cp = os.path.join(DOCKD, f"dock_{iz}.json")
            done = json.load(open(cp)) if os.path.exists(cp) else {}
            smis = [s for s in allsm if s not in done]
            print(iz, "dock todo", len(smis), flush=True)
            for i in range(NW):
                jobs.append((iz, i, smis[i::NW]))
        with Pool(12) as p_:
            for j, (isox, wid) in enumerate(p_.imap_unordered(dock_worker, jobs)):
                print(f"  {isox} w{wid} finished", flush=True)
        print("DOCK DONE", flush=True)


if __name__ == "__main__":
    main()
