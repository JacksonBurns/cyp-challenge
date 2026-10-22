#!/usr/bin/env python3
"""Normalize an Open Babel PDBQT into a Vina RIGID-receptor PDBQT.

Open Babel writes proteins in tree notation (ROOT/BRANCH/ENDBRANCH) with a
non-zero TORSDOF (it detects rotatable bonds). The Vina C++ parser rejects both
for a rigid receptor. We produce a flat ATOM-only file with TORSDOF 0.

Usage: strip_branches.py <in.pdbqt> <out.pdbqt>
"""
import sys

def main(inp, outp):
    atoms = []
    with open(inp) as fin:
        for line in fin:
            s = line.rstrip("\n")
            if s.startswith(("ROOT", "BRANCH", "ENDBRANCH", "REMARK", "TORSDOF")):
                continue
            if s.startswith("ATOM"):
                atoms.append(s)
    with open(outp, "w") as fout:
        for a in atoms:
            fout.write(a + "\n")
    print(f"{inp} -> {outp}: {len(atoms)} ATOM lines, TORSDOF 0, flat (no branches)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
