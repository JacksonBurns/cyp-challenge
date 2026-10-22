#!/usr/bin/env python3
"""Convert mmCIF _atom_site to PDB, selecting one chain (heme-bearing).
Writes protein-only (no waters) PDB for docking prep.
Usage: cif2pdb.py <in.cif> <chain> <out_protein.pdb> <out_full.pdb>
"""
import sys
from parse_cif import parse_atom_site

def main(cif, chain, out_prot, out_full):
    rows = parse_atom_site(cif)
    sel = [r for r in rows if r.get("label_asym_id") == chain]
    print(f"{cif} chain {chain}: {len(sel)} atoms")
    from collections import defaultdict
    comps = defaultdict(int)
    for r in sel:
        comps[r.get("label_comp_id","?")] += 1
    print("  comps:", dict(comps))

    def writepdb(subset, path):
        with open(path, "w") as f:
            for i, r in enumerate(subset, 1):
                el = r.get("type_symbol", "X")
                # PDB element goes in cols 77-78
                f.write(
                    f"ATOM  {i:5d}  {r.get('label_atom_id','???'):<4s} "
                    f"{r.get('label_comp_id','???'):3s} {chain}   "
                    f"{r.get('label_seq_id','?'):>4s}    "
                    f"{float(r['Cartn_x']):8.3f}{float(r['Cartn_y']):8.3f}{float(r['Cartn_z']):8.3f}"
                    f"{float(r.get('occupancy',1.0)):6.2f}{float(r.get('B_iso_or_equiv',0.0)):7.2f}      "
                    f"{el:<2s}\n"
                )
        print(f"  wrote {path} ({len(subset)} atoms)")

    # full (incl HEM, excl HOH/waters and other crystal stuff)
    SKIP = {"HOH", "SO4", "ZN", "NA", "CL", "K", "MG", "CA"}
    full = [r for r in sel if r.get("label_comp_id") not in SKIP]
    writepdb(full, out_full)
    # protein-only (20 AA), for inspection
    AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
    prot = [r for r in sel if r.get("label_comp_id") in AA]
    writepdb(prot, out_prot)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
