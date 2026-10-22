#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
echo "CN1CC[C@@]2(c3ccccc3O2)C1" > dm.smi
echo "=== obabel smiles file -> pdbqt (-x -p 7.4 -ismi --gen3d) ==="
/usr/bin/obabel dm.smi -ismi -O dm_test.pdbqt -x -p 7.4 --gen3d 2>&1 | tail -2
echo "ATOM count: $(grep -cE '^ATOM' dm_test.pdbqt)"
echo "H (HD) count: $(grep -E '^ATOM' dm_test.pdbqt | awk '{print $12}' | grep -c HD)"
echo "=== AD4 types (col 12) ==="
grep -E '^ATOM' dm_test.pdbqt | awk '{print $12}' | sort | uniq -c
echo "=== ROOT/TORSDOF present? ==="
grep -E '^(ROOT|TORSDOF|BRANCH)' dm_test.pdbqt
echo "=== try docking with official vina CLI ==="
~/miniforge3/envs/cyp/bin/vina --receptor cyp2d6_target_flat.pdbqt --ligand dm_test.pdbqt --config box.cfg --out dm_out.pdbqt --log dm_log.txt 2>&1 | tail -12
