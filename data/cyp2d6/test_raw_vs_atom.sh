#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
echo "=== A) RAW obabel ligand (kss1.pdbqt, has ROOT/BRANCH) -> write_maps ==="
rm -f /tmp/mA.*.map
~/miniforge3/envs/cyp/bin/vina --receptor cyp2d6_target_flat.pdbqt --ligand kss1.pdbqt \
  --write_maps /tmp/mA --center_x -12.796 --center_y 50.784 --center_z -6.164 \
  --size_x 22 --size_y 20 --size_z 20 --spacing 0.375 --force_even_voxels >/dev/null 2>&1
echo "  maps: $(ls /tmp/mA.*.map 2>/dev/null | wc -l)"
echo "=== B) ATOM-only ligand (stripped) -> write_maps ==="
grep -E '^ATOM' kss1.pdbqt > /tmp/atomonly.pdbqt
rm -f /tmp/mB.*.map
~/miniforge3/envs/cyp/bin/vina --receptor cyp2d6_target_flat.pdbqt --ligand /tmp/atomonly.pdbqt \
  --write_maps /tmp/mB --center_x -12.796 --center_y 50.784 --center_z -6.164 \
  --size_x 22 --size_y 20 --size_z 20 --spacing 0.375 --force_even_voxels 2>&1 | grep -iE "error|maps" | head -2
echo "  maps: $(ls /tmp/mB.*.map 2>/dev/null | wc -l)"
echo "=== C) RAW obabel ligand -> actual dock ==="
~/miniforge3/envs/cyp/bin/vina --receptor cyp2d6_target_flat.pdbqt --ligand kss1.pdbqt \
  --center_x -12.796 --center_y 50.784 --center_z -6.164 \
  --size_x 22 --size_y 20 --size_z 20 --num_modes 1 --exhaustiveness 8 --cpu 1 \
  --out /tmp/dockC.pdbqt 2>&1 | grep -iE "mode|error|affinity" | head -4
