#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
rm -f cyp2d6_target.pdbqt
echo "=== receptor PDBQT (no H-add) ==="
time obabel cyp2d6_monomer.pdb -O cyp2d6_target.pdbqt 2>&1 | tail -3
echo "=== verify ==="
ls -la cyp2d6_target.pdbqt
echo "atoms: $(grep -cE '^[A-Z]' cyp2d6_target.pdbqt)"
echo "HEM lines: $(grep -c HEM cyp2d6_target.pdbqt)"
echo "FE line:"
grep -i "FE " cyp2d6_target.pdbqt | head -1
