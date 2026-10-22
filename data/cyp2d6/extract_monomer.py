#!/usr/bin/env python3
"""Extract a complete CYP2D6 monomer (protein chain + paired heme chain) to PDB.
Usage: extract_monomer.py <in.cif> <prot_chain> <hem_chain> <out.pdb>
"""
import sys
from parse_cif import parse_atom_site

def pdb_line(serial, name, comp, chain, seq, x, y, z, el):
    """Write a PDB ATOM line with element at cols 77-78 (80-char total)."""
    header = f"ATOM  {serial:5d}  {name:<4s} {comp:3s} {chain:1s} {seq:>4s}    "
    header = header[:30].ljust(30)
    coords = f"{x:8.3f}{y:8.3f}{z:8.3f}"
    occ_b = "  1.00  0.00"
    gap = " " * 10
    elem = f"{el:>2s}"
    return (header + coords + occ_b + gap + elem)[:80].ljust(80) + "\n"

def main(cif, prot_ch, hem_ch, out):
    rows = parse_atom_site(cif)
    SKIP = {"HOH", "SO4", "ZN", "NA", "CL", "K", "MG", "CA"}
    sel = [r for r in rows if r.get("label_asym_id") in (prot_ch, hem_ch)
           and r.get("label_comp_id") not in SKIP]
    fe = [r for r in sel if r.get("type_symbol") == "FE"]
    with open(out, "w") as f:
        for i, r in enumerate(sel, 1):
            el = r.get("type_symbol", "X")
            f.write(pdb_line(
                i,
                r.get("label_atom_id", "???"),
                r.get("label_comp_id", "???"),
                r.get("label_asym_id", "A"),
                r.get("label_seq_id", "0"),
                float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]),
                el,
            ))
    print(f"wrote {out}: {len(sel)} atoms (protein {prot_ch} + heme {hem_ch})")
    if fe:
        r = fe[0]
        print(f"  HEM Fe: ({r['Cartn_x']}, {r['Cartn_y']}, {r['Cartn_z']})")
    # HEM atom count check
    nhem = sum(1 for r in sel if r.get("label_comp_id")=="HEM")
    nprot = sum(1 for r in sel if r.get("label_comp_id")!="HEM")
    print(f"  HEM={nhem} prot={nprot}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
