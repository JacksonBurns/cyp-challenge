#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
# single molecule spanning the full AD4 type space:
#  BrCC(F)Cl  -> Br, aliphatic C, F, Cl
#  S          -> thioether S
#  C(=O)N     -> amide (C, O_N, N_T)
#  c1ccnc..c1 -> aromatic ring + N (A, N_P)
#  N(C)C      -> tertiary amine (N_A)
#  CO         -> hydroxyl (O_A)
#  H's        -> HD
SMI='BrCC(F)ClSCC(=O)Nc1ccnc(N(C)C)c1CO'
echo "$SMI" > kss.smi
/usr/bin/obabel kss.smi -ismi -O kss1.pdbqt -xh --gen3d 2>&1 | tail -1
echo "=== obabel atom types in reference ==="
grep -E '^ATOM' kss1.pdbqt | awk '{print $NF}' | sort | uniq -c
echo "=== generate maps ==="
rm -f /tmp/testmaps.*.map
~/miniforge3/envs/cyp/bin/vina --receptor cyp2d6_target_flat.pdbqt --ligand kss1.pdbqt \
  --write_maps /tmp/testmaps \
  --center_x -12.796 --center_y 50.784 --center_z -6.164 \
  --size_x 22 --size_y 20 --size_z 20 --spacing 0.375 --force_even_voxels 2>&1 | tail -3
echo "=== maps produced (need to cover: C_H C_P N_A N_D N_P N_T O_A O_N S_H/F/Cl/Br HD) ==="
ls /tmp/testmaps.*.map 2>/dev/null | sed 's/.*testmaps\.//' | sort | tr '\n' ' '
echo
