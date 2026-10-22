#!/usr/bin/env python3
"""Build the CYP2D6 docking affinity maps by UNION over several single-molecule
references.

Vina's grid map name depends on the atom's charge state (e.g. O -> O_A or O_DA,
N -> N_A / N_D / N_DA / N_P), so no single reference ligand produces the full
set of map names that docking ligands look up. We generate maps from several
single-molecule references (each Vina-valid) and UNION the .map files into one
directory. Docking any dataset ligand then finds its grid type.

Each reference is a grid/atom-type template only - no label/model role.
"""
import subprocess, os, tempfile, shutil, glob
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
OBABEL = "/usr/bin/obabel"
HERE = os.path.dirname(os.path.abspath(__file__))
VINA = os.path.expanduser("~/miniforge3/envs/cyp/bin/vina")
RECEPTOR = os.path.join(HERE, "cyp2d6_target_flat.pdbqt")
MAPS_PREFIX = os.path.join(HERE, "cyp2d6_maps")
CENTER = (-12.796, 50.784, -6.164)
SIZE = (22, 20, 20)

# references chosen to span all charge states of O, N + all halogens + S + P:
REFS = {
    "core":  "OC(=O)CC(=O)Nc1ccnc(NCC)cc1OS",        # C_H C_P O_A O_DA N_A N_D N_P S_P
    "acid":  "OC(=O)c1ccccc1O",                       # O_DA (carboxyl) + O_A (phenol)
    "pyrid": "c1ccncc1",                               # N_A (aromatic N, simple ring)
    "np1":   "CCOC(=O)N1CSCC1C(=O)O",                 # N_P (thiazolidinedione aromatic N)
    "np2":   "O=C1NN=C(c2ccccc2)N=C1C(=O)O",          # N_P (barbiturate fused aromatic N)
    "np3":   "c1ncnc2ccccc12",                         # N_P (quinoline fused aromatic N)
    "amine": "CN(C)CCN",                               # N_A / N_D / N_DA (aliphatic amines)
    "amide": "CC(=O)NCC(=O)O",                         # N_T/O_DA amide states
    "halo":  "IP(F)(F)FC1=CC(=CC=C1Br)SCCCl",         # I F Br Cl halogen maps
    "phos":  "CPC(F)(F)F",                             # P_P phosphine
}


def gen_maps_for(smi, tag):
    with tempfile.NamedTemporaryFile("w", suffix=".smi", delete=False) as tf:
        tf.write(smi + "\n")
        sp = tf.name
    pq = sp.replace(".smi", ".pdbqt")
    subprocess.run([OBABEL, sp, "-O", pq, "-xh", "--gen3d"],
                   capture_output=True, text=True, timeout=60)
    os.unlink(sp)
    if not os.path.exists(pq):
        return []
    tmp_prefix = f"/tmp/refmaps_{tag}"
    cmd = [VINA, "--receptor", RECEPTOR, "--ligand", pq, "--write_maps", tmp_prefix,
           "--center_x", str(CENTER[0]), "--center_y", str(CENTER[1]),
           "--center_z", str(CENTER[2]),
           "--size_x", str(SIZE[0]), "--size_y", str(SIZE[1]), "--size_z", str(SIZE[2]),
           "--spacing", "0.375", "--force_even_voxels"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    os.unlink(pq)
    produced = [os.path.basename(p) for p in glob.glob(tmp_prefix + ".*.map")]
    # copy into the union dir, normalizing the name to cyp2d6_maps.<TYPE>.map
    for p in glob.glob(tmp_prefix + ".*.map"):
        base = os.path.basename(p)           # e.g. refmaps_core.C_H.map
        typ = base.split(".", 1)[1]          # C_H.map (everything after first dot)
        dest = os.path.join(HERE, f"cyp2d6_maps.{typ}")
        shutil.copy(p, dest)
        os.unlink(p)
    return produced


def main():
    # clean
    for p in glob.glob(MAPS_PREFIX + ".*.map"):
        os.unlink(p)
    union = set()
    for tag, smi in REFS.items():
        produced = gen_maps_for(smi, tag)
        print(f"  ref {tag:6s}: {sorted(os.path.basename(p) for p in produced)}")
        union |= set(os.path.basename(p) for p in produced)
    allmaps = sorted(os.path.basename(p) for p in glob.glob(MAPS_PREFIX + ".*.map"))
    print(f"\nUNION of maps ({len(allmaps)}): {allmaps}")


if __name__ == "__main__":
    main()
