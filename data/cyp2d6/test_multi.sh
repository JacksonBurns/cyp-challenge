#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
# Build a multi-molecule RAW obabel reference (each ligand keeps its own
# MODEL/ROOT/BRANCH/TORSDOF tree) and test if Vina --write_maps accepts it.
cat > kss_multi.smi <<'EOF'
BrCC(F)ClSCC(=O)Nc1ccnc(N(C)C)c1CO
CNCCc1ccc(CC(O)CNC(C)C)cc1
CC(=O)Nc1ccc(Cl)cc1
c1ccncc1
CCSCC
BrCCO
CN1CC[C@@]2(c3ccccc3O2)C1
NC(=O)c1ccccc1
EOF
/usr/bin/obabel kss_multi.smi -ismi -O kss_multi.pdbqt -xh --gen3d 2>&1 | tail -1
echo "=== multi-molecule RAW reference: n molecules (MODEL count) ==="
grep -c '^MODEL' kss_multi.pdbqt
echo "=== atom types across all molecules ==="
grep -E '^ATOM' kss_multi.pdbqt | awk '{print $NF}' | sort | uniq -c
echo "=== test --write_maps with multi-molecule RAW reference ==="
rm -f /tmp/mm.*.map
~/miniforge3/envs/cyp/bin/vina --receptor cyp2d6_target_flat.pdbqt --ligand kss_multi.pdbqt \
  --write_maps /tmp/mm --center_x -12.796 --center_y 50.784 --center_z -6.164 \
  --size_x 22 --size_y 20 --size_z 20 --spacing 0.375 --force_even_voxels 2>&1 | grep -iE "error|maps|Reading" | head -3
echo "  maps produced: $(ls /tmp/mm.*.map 2>/dev/null | wc -l)"
ls /tmp/mm.*.map 2>/dev/null | sed 's/.*mm\.//' | sort | tr '\n' ' '; echo
