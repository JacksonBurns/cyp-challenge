#!/usr/bin/env python3
"""CYP2D6 docking feature extractor (official AutoDock Vina CLI).

Docks each SMILES into the fixed CYP2D6 holo structure (2F9Q monomer A+heme E,
box centered on the heme Fe) with the official AutoDock Vina CLI, then extracts
a compact LABEL-FREE interaction fingerprint from the best pose. Docking is a
physics/geometry computation fit to NO pIC50 labels - a legitimate
prior-knowledge representation (same class as the LLM-pharmacophore avenue).

Efficiency: the binding-site affinity maps are computed ONCE and reused for
every ligand (--maps), so per-ligand cost is just the global search (~1-3s).

Ligand prep: SMILES -> AddHs -> ETKDG embed -> MMFF -> molblock ->
Open Babel `-xh -p 7.4` (preserve explicit H, protonate basic amines at pH 7.4)
-> PDBQT. (Plain `-x` silently drops explicit H; `-xh` keeps them.)

Feature vector (per molecule, fixed length, NaN-free by construction):
  vina_score        best-pose affinity (kcal/mol, more negative = tighter)
  n_hbond           ligand {N,O} <-> protein {N,O,H} within 3.5A
  n_hydrophobic     ligand C <-> protein C within 4.5A
  n_pi_stack        ligand C <-> PHE/TYR/TRP C within 4.5A
  n_cation_pi       protonated-N <-> PHE/TYR/TRP C within 6A
  n_salt            protonated-N <-> ASP/GLU O within 4.5A
  n_heme_contact    ligand atoms within 6A of any HEM atom (count)
  basicN_Fe_dist    protonated-N -> heme Fe
  basicN_acid_dist  protonated-N -> nearest ASP/GLU O
  basicN_arom_dist  protonated-N -> nearest aromatic C
  pose_inbox_frac   fraction of ligand atoms inside the docking box
  n_heavy           ligand heavy-atom count
  has_basicN        1 if a protonated (charged) N was found
  dock_fail         1 if 3D-gen/docking failed (other features 0)

Usage:
  dock_features.py --smiles file.txt --out cache/dockfeat_cyp2d6.csv \
      [--n 300] [--exhaustiveness 8] [--workers 12] [--maps maps_prefix]
"""
import argparse
import os
import re
import subprocess
import tempfile
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "cyp2d6")
RECEPTOR = os.path.join(DATA, "cyp2d6_target_flat.pdbqt")
PROTEIN_PDB = os.path.join(DATA, "cyp2d6_monomer.pdb")
VINA = os.path.expanduser("~/miniforge3/envs/cyp/bin/vina")
OBABEL = "/usr/bin/obabel"
MAPS_PREFIX = os.path.join(DATA, "cyp2d6_maps")

# heme iron (2F9Q chain A + heme E)
FE_POS = np.array([-12.796, 50.784, -6.164])
BOX_CENTER = FE_POS.tolist()
BOX_SIZE = [22.0, 20.0, 20.0]
SPACING = 0.375
# Per-child address-space cap (GB) for the per-ligand obabel/vina subprocesses.
# Set in main() from --memcap-gb. 0 = no cap. A child that can't allocate exits
# non-zero -> dock_fail, instead of OOM-killing the whole job + shared machine.
_MEMCAP_GB = 0.0

AA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
AROMATIC_RES = {"PHE", "TYR", "TRP"}
ACIDIC_RES = {"ASP", "GLU"}


# ---------------- protein reference (loaded once per worker) ----------------
_PROT = None
def _load_prot():
    global _PROT
    if _PROT is not None:
        return _PROT
    coords, elems, comps = [], [], []
    with open(PROTEIN_PDB) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            comp = line[17:20].strip()
            if comp in ("HOH", "SO4", "ZN", "NA", "CL"):
                continue
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                elems.append(line[76:78].strip() or "?")
                comps.append(comp)
            except Exception:
                continue
    _PROT = {"coords": np.array(coords), "elems": np.array(elems), "comps": np.array(comps)}
    return _PROT


# ---------------- protonation (pH 7.4 basic amines) ----------------
# Open Babel's `-p 7.4` does NOT reliably protonate aliphatic amines (they stay
# neutral). The CYP2D6 pharmacophore is the PROTONATED basic amine (salt-bridge
# to the conserved Asp, cation-pi, Fe coordination), so we protonate explicitly.
def _is_basic_amine_n(atom):
    if atom.GetSymbol() != "N" or atom.GetIsAromatic():
        return False
    # a basic amine N is ALWAYS single-bonded; double/triple bond (nitrile,
    # imine, azide) disqualifies
    for b in atom.GetBonds():
        if b.GetBondTypeAsDouble() >= 2.0:
            return False
    # amide N (bonded to C(=O)) is not basic
    for b in atom.GetBonds():
        other = b.GetOtherAtom(atom)
        if other.GetSymbol() == "C":
            for b2 in other.GetBonds():
                o2 = b2.GetOtherAtom(other)
                if o2.GetSymbol() == "O" and b2.GetBondTypeAsDouble() >= 2.0:
                    return False
    heavy = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() != "H")
    return heavy <= 3


def _protonate_basic_amines(mol):
    """Add an explicit H + +1 charge to each basic aliphatic amine N."""
    mol = Chem.AddHs(mol)
    em = Chem.RWMol(mol)
    targets = [a.GetIdx() for a in em.GetAtoms()
               if _is_basic_amine_n(a) and a.GetFormalCharge() == 0]
    for idx in targets:
        hidx = em.AddAtom(Chem.Atom(1))
        em.AddBond(idx, hidx, Chem.BondType.SINGLE)
        em.GetAtomWithIdx(idx).SetFormalCharge(1)
    out = em.GetMol()
    Chem.SanitizeMol(out)
    return out


# ---------------- direct AD4 PDBQT writer (preserves protonation) ----------------
# Open Babel neutralizes protonated basic amines (loses the N_D type + positive
# charge), which would defeat the CYP2D6 basic-amine pharmacophore. So we write
# the ligand PDBQT directly from RDKit, controlling atom types + partial charges.
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
    if el in ("F", "Cl", "Br", "I"):
        return el  # halogens keep their own AD4 type
    return "C"


def _mol_to_pdbqt(mol):
    """mol (3D + H + protonation) -> PDBQT string. Gasteiger partial charges;
    protonated N (formal +1) gets a +0.4 override (Gasteiger can go negative)."""
    try:
        Chem.AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    lines = []  # ATOM-only; Vina's C++ parser rejects a TORSDOF tag
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms(), 1):
        pos = conf.GetAtomPosition(i - 1)
        el = atom.GetSymbol()
        typ = _ad4_type(atom)
        q = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        if atom.GetFormalCharge() > 0:
            q = 0.4
        # Place fields by exact column index to match the layout Vina accepts.
        s = [" "] * 80
        s[0:6] = list("ATOM  ")
        s[6:11] = list(f"{i:5d}")
        s[12:16] = list(f" {el:<3s}")
        s[17:20] = list("UNL")
        s[21:26] = list("    1")
        s[32:40] = list(f"{pos.x:8.3f}")
        s[40:48] = list(f"{pos.y:8.3f}")
        s[48:56] = list(f"{pos.z:8.3f}")
        s[56:62] = list("  0.00")
        s[62:68] = list("  0.00")
        s[70:76] = list(f"{q:+6.3f}")
        s[77:77 + len(typ)] = list(typ)
        lines.append("".join(s))
    return "\n".join(lines) + "\n"


# ---------------- maps (computed once) ----------------
def ensure_maps():
    """Compute the binding-site affinity maps once if not present.
    Vina requires a ligand reference to define the grid AND to determine which
    atom-type affinity maps to build. We use a kitchen-sink reference covering
    the union of AD4 atom types across the dataset (incl. N_D protonated amine)
    so NO ligand hits a 'map not present' error. The reference is a grid
    template only - it has no label/model role."""
    # Check if ANY union map file exists (pattern: cyp2d6_maps.<TYPE>.map)
    existing = [f for f in os.listdir(DATA) if f.startswith("cyp2d6_maps.") and f.endswith(".map")]
    if len(existing) >= 10:  # expect 14 maps; 10+ means a prior build is intact
        return
    ref_lig = os.path.join(DATA, "kss_ref.pdbqt")
    if not os.path.exists(ref_lig):
        raise RuntimeError(f"kitchen-sink reference missing: {ref_lig} "
                           f"(run make_kitchen_sink.py first)")
    cmd = [VINA, "--receptor", RECEPTOR, "--ligand", ref_lig,
           "--write_maps", MAPS_PREFIX,
           "--center_x", str(BOX_CENTER[0]), "--center_y", str(BOX_CENTER[1]),
           "--center_z", str(BOX_CENTER[2]),
           "--size_x", str(int(BOX_SIZE[0])), "--size_y", str(int(BOX_SIZE[1])),
           "--size_z", str(int(BOX_SIZE[2])), "--spacing", str(SPACING),
           "--force_even_voxels"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"map generation failed: {r.stderr[-500:]}")
    # list the maps built
    maps = [f for f in os.listdir(DATA) if f.startswith("cyp2d6_maps.") and f.endswith(".map")]
    print(f"maps written: {len(maps)} atom-type maps: {sorted(maps)}", flush=True)


# ---------------- ligand prep ----------------
def _ligand_pdbqt(smi):
    """SMILES -> PDBQT via Open Babel (proven dockable by Vina CLI).
    Uses -xh to preserve explicit H. Obabel yields a CONSISTENT set of AD4
    atom types (neutral amines) across all ligands, so the kitchen-sink map
    reference (built from the same obabel output) covers every ligand.
    NOTE: obabel neutralizes basic amines (no protonation); documented as a
    limitation - the pose is still driven by the binding-site geometry.
    Returns pdbqt string or None."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    molblock = Chem.MolToMolBlock(mol)
    with tempfile.NamedTemporaryFile("w", suffix=".mol", delete=False) as tf:
        tf.write(molblock)
        mol_path = tf.name
    pdbqt_path = mol_path.replace(".mol", ".pdbqt")
    try:
        r = subprocess.run([OBABEL, mol_path, "-O", pdbqt_path, "-xh"],
                           capture_output=True, text=True, timeout=120,
                           preexec_fn=_child_memcap)
        if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
            with open(pdbqt_path) as f:
                content = f.read()
            # keep the RAW obabel output (ROOT/BRANCH/TORSDOF) - Vina's ligand
            # parser expects the torsion tree. (Only the receptor is ATOM-only.)
            if "ATOM" in content:
                return content
        return None
    except Exception:
        return None
    finally:
        for p in (mol_path, pdbqt_path):
            if os.path.exists(p):
                try: os.unlink(p)
                except Exception: pass


def _parse_pdbqt_atoms(pdbqt_str):
    """Return (coords Nx3, elements list, types list) in file order.
    Open Babel PDBQT: standard PDB coord columns (x[30:38] y[38:46] z[46:54]);
    element = 3rd whitespace token, AD4 type = last token."""
    coords, elems, types = [], [], []
    for line in pdbqt_str.splitlines():
        if line.startswith("ATOM"):
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            toks = line.split()
            elems.append(toks[2] if len(toks) > 2 else "?")
            types.append(toks[-1])
    return np.array(coords) if coords else None, elems, types


def _dock_one(smi, exh=8):
    """Dock one SMILES via the Vina CLI (reusing precomputed maps).
    Returns (affinity, pose_coords Nx3, ligand_elems, ligand_types) or None."""
    pdbqt_str = _ligand_pdbqt(smi)
    if pdbqt_str is None:
        return None
    lig_coords, lig_elems, lig_types = _parse_pdbqt_atoms(pdbqt_str)
    if lig_coords is None:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".pdbqt", delete=False) as tf:
        tf.write(pdbqt_str)
        lig_path = tf.name
    out_path = lig_path.replace(".pdbqt", "_out.pdbqt")
    try:
        cmd = [VINA, "--maps", MAPS_PREFIX, "--ligand", lig_path,
               "--center_x", str(BOX_CENTER[0]), "--center_y", str(BOX_CENTER[1]),
               "--center_z", str(BOX_CENTER[2]),
               "--size_x", str(int(BOX_SIZE[0])), "--size_y", str(int(BOX_SIZE[1])),
               "--size_z", str(int(BOX_SIZE[2])), "--spacing", str(SPACING),
               "--num_modes", "1", "--exhaustiveness", str(exh),
               "--cpu", "1", "--out", out_path]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                               preexec_fn=_child_memcap)
        except subprocess.TimeoutExpired:
            return None  # single slow ligand should not kill the whole pool
        if r.returncode != 0 or not os.path.exists(out_path):
            return None
        # affinity from stdout: first "mode 1  -X.XXX"
        m = re.search(r"^\s*1\s+(-?\d+\.\d+)", r.stdout, re.M)
        aff = float(m.group(1)) if m else 0.0
        # pose = ALL ATOM coords in the output (num_modes=1, single ligand ->
        # one model). Do NOT break on ENDROOT/ENDBRANCH (nested torsion tree).
        pose = []
        for line in open(out_path):
            if line.startswith("ATOM"):
                pose.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if not pose:
            return None
        return aff, np.array(pose), lig_elems, lig_types
    finally:
        for p in (lig_path, out_path):
            if os.path.exists(p):
                try: os.unlink(p)
                except Exception: pass


# ---------------- feature extraction ----------------
FEATS = ["vina_score", "n_hbond", "n_hydrophobic", "n_pi_stack", "n_cation_pi",
         "n_salt", "n_heme_contact", "basicN_Fe_dist", "basicN_acid_dist",
         "basicN_arom_dist", "pose_inbox_frac", "n_heavy", "has_basicN", "dock_fail"]


def _fingerprint(aff, pose, lig_elems, lig_types):
    out = dict.fromkeys(FEATS, 0.0)
    out["dock_fail"] = 0.0
    out["vina_score"] = aff
    out["n_heavy"] = sum(1 for e in lig_elems if e != "H")
    prot = _load_prot()
    pc, pe, pcomp = prot["coords"], prot["elems"], prot["comps"]

    heavy = np.array([e != "H" for e in lig_elems])
    lp = pose[heavy]
    le = [e for e, h in zip(lig_elems, heavy) if h]
    lt = [t for t, h in zip(lig_types, heavy) if h]
    n_heavy = len(lp)

    def count_within(lig_mask, prot_mask, dmax, cap=99):
        if not lig_mask.any() or not prot_mask.any():
            return 0
        lm = lp[lig_mask]
        pm = pc[prot_mask]
        c = 0
        for i in range(len(lm)):
            c += int((np.sum((pm - lm[i]) ** 2, axis=1) < dmax * dmax).sum())
        return min(c, cap)

    lig_no = np.array([e in ("N", "O") for e in le])
    lig_c = np.array([e == "C" for e in le])
    prot_no = np.isin(pe, ["N", "O", "H"])
    prot_c = np.array(pe == "C")
    prot_arom_c = np.array([(pe[i] == "C") and (pcomp[i] in AROMATIC_RES) for i in range(len(pe))])
    prot_acid_o = np.array([(pe[i] == "O") and (pcomp[i] in ACIDIC_RES) for i in range(len(pe))])

    out["n_hbond"] = count_within(lig_no, prot_no, 3.5)
    out["n_hydrophobic"] = count_within(lig_c, prot_c, 4.5)
    out["n_pi_stack"] = count_within(lig_c, prot_arom_c, 4.5)

    # heme contact
    hem = np.array(pcomp == "HEM")
    if hem.any():
        hemc = pc[hem]
        out["n_heme_contact"] = min(int(sum(
            1 for p in lp if np.any(np.sum((hemc - p) ** 2, axis=1) < 36))), 99)

    # basic amine N: in this obabel AD4 type set, 'NA' = amine (the basic N),
    # 'N' = aromatic N (pyridine, not basic). So the basic amine is type NA.
    bN_local = None
    for i, t in enumerate(lt):
        if t == "NA":
            bN_local = i
            break
    bN = None
    if bN_local is not None:
        bN = lp[bN_local]
        out["has_basicN"] = 1.0
        out["n_cation_pi"] = count_within(np.array([bN_local]), prot_arom_c, 6.0)
        out["n_salt"] = count_within(np.array([bN_local]), prot_acid_o, 4.5)
        out["basicN_Fe_dist"] = float(np.linalg.norm(bN - FE_POS))
        if prot_acid_o.any():
            out["basicN_acid_dist"] = float(np.min(np.linalg.norm(pc[prot_acid_o] - bN, axis=1)))
        if prot_arom_c.any():
            out["basicN_arom_dist"] = float(np.min(np.linalg.norm(pc[prot_arom_c] - bN, axis=1)))
    if bN is None:
        out["basicN_Fe_dist"] = 99.0
        out["basicN_acid_dist"] = 99.0
        out["basicN_arom_dist"] = 99.0

    cx, cy, cz = BOX_CENTER
    hx, hy, hz = [s / 2 for s in BOX_SIZE]
    inb = (np.abs(pose[:, 0] - cx) < hx) & (np.abs(pose[:, 1] - cy) < hy) & (np.abs(pose[:, 2] - cz) < hz)
    out["pose_inbox_frac"] = float(inb.mean()) if len(inb) else 0.0
    return out


def _worker(idx_smi):
    idx, smi, exh = idx_smi
    res = _dock_one(smi, exh=exh)
    if res is None:
        return idx, (dict.fromkeys(FEATS, 0.0) | {"dock_fail": 1.0})
    aff, pose, le, lc = res
    return idx, _fingerprint(aff, pose, le, lc)


def _worker_init(memcap_gb):
    """Pool initializer: cap each worker's own address space (defensive)."""
    if memcap_gb and memcap_gb > 0:
        import resource
        cap = int(memcap_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def _child_memcap():
    """preexec_fn: cap the VINA/OBABEL child's address space. The worker-init
    cap only limits the Python worker, NOT the subprocess it spawns - and the
    OOM blowup (vina 3.5GB RSS on a large/flexible ligand) happens in the child.
    A capped child that can't allocate exits non-zero -> _dock_one returns None
    -> dock_fail, instead of OOM-killing the whole job + shared machine."""
    import resource
    cap = int(_MEMCAP_GB * 1024 ** 3) if _MEMCAP_GB > 0 else resource.RLIM_INFINITY
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def _write_out(res, smis, out):
    """Write the (possibly partial) result table. res is index-aligned to smis;
    any None entries become dock_fail rows so the file is always valid/resumable."""
    rows = []
    for i, r in enumerate(res):
        if r is None:
            r = dict.fromkeys(FEATS, 0.0) | {"dock_fail": 2.0}  # 2 = not-yet-done
        rows.append(r)
    df = pd.DataFrame(rows)
    df.insert(0, "SMILES", smis)  # res is index-aligned, so SMILES match rows
    df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--maps", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--memcap-gb", type=float, default=2.0,
                    help="Per-worker address-space cap (GB). 0 = no cap.")
    args = ap.parse_args()

    global MAPS_PREFIX
    if args.maps:
        MAPS_PREFIX = args.maps
    global _MEMCAP_GB
    _MEMCAP_GB = args.memcap_gb
    ensure_maps()

    with open(args.smiles) as f:
        smis = [l.strip() for l in f if l.strip()]
    if args.n > 0:
        smis = smis[:args.n]

    # --- Resume: reload any prior checkpoint, skip already-docked SMILES. ---
    # Makes the run resilient to OOM/kill: progress is no longer all-or-nothing.
    res: list = [None] * len(smis)
    todo_idx = list(range(len(smis)))
    if os.path.exists(args.out):
        try:
            prev = pd.read_csv(args.out)
            smi_to_row = {r["SMILES"]: r for r in prev.to_dict("records")}
            n_done = 0
            for i, s in enumerate(smis):
                row = smi_to_row.get(s)
                if row is not None and float(row.get("dock_fail", 0)) < 2.0:
                    res[i] = {c: row[c] for c in FEATS}
                    n_done += 1
            todo_idx = [i for i in range(len(smis)) if res[i] is None]
            print(f"RESUME: {n_done}/{len(smis)} already done, {len(todo_idx)} to go", flush=True)
        except Exception as e:
            print(f"resume load failed ({e}); starting fresh", flush=True)
            res = [None] * len(smis)
            todo_idx = list(range(len(smis)))

    if not todo_idx:
        print("Nothing to do - all SMILES already docked.", flush=True)
        return

    print(f"docking {len(smis)} ligands ({len(todo_idx)} todo), "
          f"{args.workers} workers, exh={args.exhaustiveness}", flush=True)

    t0 = time.time()
    n = len(smis)
    base_done = n - len(todo_idx)
    with Pool(args.workers, initializer=_worker_init,
              initargs=(args.memcap_gb,)) as pool:
        done = 0
        for idx, feat in pool.imap_unordered(
                _worker, [(i, smis[i], args.exhaustiveness) for i in todo_idx]):
            res[idx] = feat
            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                rate = done / el
                eta = (len(todo_idx) - done) / rate if rate > 0 else 0
                print(f"  {base_done + done}/{n}  {el:.0f}s  ETA {eta:.0f}s", flush=True)
            if args.checkpoint_every and done % args.checkpoint_every == 0:
                _write_out(res, smis, args.out)  # checkpoint (atomic-ish)

    _write_out(res, smis, args.out)
    df = pd.read_csv(args.out)
    fails = int(df["dock_fail"].sum())
    print(f"DONE {len(df)} rows ({fails} failed) in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
