#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
smi="CN1CC[C@@]2(c3ccccc3O2)C1"
echo "=== obabel SMILES -> pdbqt (-x -p 7.4 --gen3d) ==="
/usr/bin/obabel "smi=${smi}" -O dm_test.pdbqt -x -p 7.4 --gen3d 2>&1 | tail -2
echo "ATOM count: $(grep -cE '^ATOM' dm_test.pdbqt)"
echo "H (HD) count: $(grep -E '^ATOM' dm_test.pdbqt | awk '{print $12}' | grep -c HD)"
echo "=== AD4 types (col 12) ==="
grep -E '^ATOM' dm_test.pdbqt | awk '{print $12}' | sort | uniq -c
echo "=== has ROOT/TORSDOF? ==="
grep -cE '^(ROOT|TORSDOF)' dm_test.pdbqt
