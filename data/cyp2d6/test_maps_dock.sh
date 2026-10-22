#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
VINA=~/miniforge3/envs/cyp/bin/vina
MAPS=/home/jackson/cyp-challenge/data/cyp2d6/cyp2d6_maps
echo "=== maps present? ==="
ls -la ${MAPS}.*.map 2>/dev/null | head
echo "=== dock dm_test.pdbqt with --maps ==="
$VINA --maps $MAPS --ligand dm_test.pdbqt \
  --center_x -12.796 --center_y 50.784 --center_z -6.164 \
  --size_x 22 --size_y 20 --size_z 20 --spacing 0.375 \
  --num_modes 1 --exhaustiveness 8 --cpu 1 --out /tmp/dm_maps_out.pdbqt 2>&1 | tail -15
