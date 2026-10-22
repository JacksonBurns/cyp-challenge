#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
# Test which reference structures produce which Vina grid map names.
# Goal: cover O_DA, N_P, and everything docking ligands look up.
test_ref () {
  local smi="$1"; local tag="$2"
  echo "$smi" > /tmp/r_$tag.smi
  /usr/bin/obabel /tmp/r_$tag.smi -ismi -O /tmp/r_$tag.pdbqt -xh --gen3d >/dev/null 2>&1
  rm -f /tmp/m_$tag.*.map
  ~/miniforge3/envs/cyp/bin/vina --receptor cyp2d6_target_flat.pdbqt --ligand /tmp/r_$tag.pdbqt \
    --write_maps /tmp/m_$tag --center_x -12.796 --center_y 50.784 --center_z -6.164 \
    --size_x 22 --size_y 20 --size_z 20 --spacing 0.375 --force_even_voxels >/dev/null 2>&1
  echo "$tag: $(ls /tmp/m_$tag.*.map 2>/dev/null | sed 's/.*m_'$tag'\.//' | sort | tr '\n' ' ')"
}
# carboxylic acid (O in O_DA state?), phenol, ether
test_ref 'OC(=O)Cc1ccccc1O' acid_phenol
# amide + pyridine + amine
test_ref 'CC(=O)Nc1ccncc1CCN' amide_pyridine
# combine: acid + phenol + amide + pyridine + amine + S + halogens
test_ref 'OC(=O)CC(=O)Nc1ccnc(NCC)cc1OS' full_combo
echo "=== now dock a real failing ligand and see ALL grid types it looks up ==="
# use a ligand with O, N, S, halogen
echo 'CC(=O)Oc1ccc(O)cc1' > /tmp/docktest.smi
/usr/bin/obabel /tmp/docktest.smi -ismi -O /tmp/docktest.pdbqt -xh --gen3d >/dev/null 2>&1
rm -f /tmp/mdock.*.map
# generate maps from full_combo, then dock
cp /tmp/m_full_combo.*.map /tmp/mdock.*.map 2>/dev/null
~/miniforge3/envs/cyp/bin/vina --maps /tmp/mdock --ligand /tmp/docktest.pdbqt \
  --center_x -12.796 --center_y 50.784 --center_z -6.164 \
  --size_x 22 --size_y 20 --size_z 20 --num_modes 1 --exhaustiveness 8 --cpu 1 \
  --out /tmp/docktest_out.pdbqt 2>&1 | grep -iE "not present|mode" | head -3
