#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
echo "=== ligand with -xh -p 7.4 (from H-bearing molblock) ==="
/usr/bin/obabel dm_h.mol -O dm_test.pdbqt -xh -p 7.4 2>&1 | tail -1
echo "ATOM count: $(grep -cE '^ATOM' dm_test.pdbqt)  (expect ~26)"
echo "N type (NA=neutral amine, should be protonated at 7.4 -> check):"
grep -E '^ATOM' dm_test.pdbqt | awk '{print $2, $12}' | grep -E " N " 
echo
echo "=== official vina CLI dock (dextromethorphan) ==="
~/miniforge3/envs/cyp/bin/vina --receptor cyp2d6_target_flat.pdbqt --ligand dm_test.pdbqt \
  --center_x -12.796 --center_y 50.784 --center_z -6.164 \
  --size_x 22 --size_y 20 --size_z 20 \
  --num_modes 3 --exhaustiveness 8 \
  --out dm_out.pdbqt --log dm_log.txt 2>&1 | tail -14
