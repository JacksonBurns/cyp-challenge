#!/usr/bin/env python3
"""Parse mmCIF _atom_site tables: find HEM iron, ligand centroids."""
import sys
from collections import defaultdict
import numpy as np

def parse_atom_site(path):
    cols = None
    rows = []
    in_atom_site = False
    with open(path) as f:
        for line in f:
            stripped = line.rstrip('\n')
            if stripped.startswith("loop_"):
                in_atom_site = False
                cols = None
                continue
            if stripped.startswith("_atom_site."):
                if cols is None:
                    cols = []
                    in_atom_site = True
                name = stripped.split()[0].replace("_atom_site.", "")
                cols.append(name)
                continue
            if in_atom_site and cols:
                if stripped.startswith("_") or stripped.startswith("loop_") or stripped == "":
                    break
                parts = stripped.split()
                if len(parts) >= len(cols):
                    rows.append(dict(zip(cols, parts)))
    return rows

AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
SKIP = {"HOH", "HEM", "SO4", "ZN", "ZINC", "NA", "CL", "K", "MG", "CA", "GOL", "DOE", "EPS"}

def main(cif):
    rows = parse_atom_site(cif)
    print(f"=== {cif}: {len(rows)} atoms ===")
    if not rows:
        print("  PARSE FAILED - no rows")
        return

    comps = defaultdict(list)
    for r in rows:
        comp = r.get("label_comp_id", "?")
        comps[comp].append(r)

    nonstd = {k: len(v) for k, v in comps.items() if k not in AA and k not in SKIP}
    print(f"  non-standard comps: {nonstd}")

    # HEM iron
    hem = comps.get("HEM", [])
    fe = [r for r in hem if r.get("type_symbol", "") == "Fe"]
    fe_pos = None
    if fe:
        f = fe[0]
        fe_pos = (float(f["Cartn_x"]), float(f["Cartn_y"]), float(f["Cartn_z"]))
        print(f"  HEM Fe: ({fe_pos[0]:.2f}, {fe_pos[1]:.2f}, {fe_pos[2]:.2f})  chain={f.get('label_asym_id')}")
    if hem:
        h = np.array([[float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])] for r in hem])
        print(f"  HEM centroid: ({h[:,0].mean():.2f}, {h[:,1].mean():.2f}, {h[:,2].mean():.2f})  n_atoms={len(hem)}")

    # Ligands
    for comp, atoms in sorted(comps.items()):
        if comp in AA or comp in SKIP:
            continue
        if len(atoms) < 5:
            continue
        coords = np.array([[float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])] for r in atoms])
        c = coords.mean(axis=0)
        chain = atoms[0].get('label_asym_id', '?')
        line = f"  LIGAND {comp}: n={len(atoms)} centroid=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) chain={chain}"
        if fe_pos:
            d = np.linalg.norm(c - np.array(fe_pos))
            line += f"  dist_to_Fe={d:.2f}A"
        print(line)

if __name__ == "__main__":
    for cif in sys.argv[1:]:
        main(cif)
        print()
