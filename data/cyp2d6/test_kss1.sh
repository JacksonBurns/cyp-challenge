#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
# single-molecule kitchen sink: aromatic C(A), aliphatic C(C), amide(N_T,O_N),
# amine(N_D/N_A), aromatic N(N_P), OH(O_A), S, F, Cl
SMI='CC(=O)Nc1ccc(NC(C)C)c(F)c1SCCCl'
echo "$SMI" > kss.smi
/usr/bin/obabel kss.smi -ismi -O kss1.pdbqt -xh --gen3d 2>&1 | tail -1
echo "=== kss1.pdbqt atom types ==="
grep -E '^ATOM' kss1.pdbqt | awk '{print $NF}' | sort | uniq -c
echo "=== n atoms: $(grep -cE '^ATOM' kss1.pdbqt) ==="
echo "=== try generating maps from this single-molecule reference ==="
~/miniforge3/envs/cyp/bin/vina --receptor cyp2d6_target_flat.pdbqt --ligand kss1.pdbqt \
  --write_maps /tmp/testmaps \
  --center_x -12.796 --center_y 50.784 --center_z -6.164 \
  --size_x 22 --size_y 20 --size_z 20 --spacing 0.375 --force_even_voxels 2>&1 | tail -6
echo "=== maps produced ==="
ls /tmp/testmaps.*.map 2>/dev/null | sed 's/.*maps\.//' | sort | tr '\n' ' '
echo
