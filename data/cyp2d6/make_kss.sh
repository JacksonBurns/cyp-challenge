#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
# kitchen-sink ligand: amide (N_T, O_N), pyridine (N_P), aromatic (C_P),
# aliphatic (C_H), protonated amine (N_D), hydroxyl (O_A), sulfur (S_)
echo 'CS(=O)Nc1ccnc(N)c1C(=O)NCC[NH3+]CO' > kss.smi
echo "=== kitchen-sink ligand atom types ==="
/usr/bin/obabel kss.smi -ismi -O kss.pdbqt -xh -p 7.4 --gen3d 2>&1 | tail -1
grep -E '^ATOM' kss.pdbqt | awk '{print $12}' | sort | uniq -c
echo "=== need at least: C_H C_P N_A N_D N_T N_P O_A O_N S_ HD ==="
