#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
# Build a kitchen-sink reference PDBQT from a REAL obabel-converted ligand
# (guaranteed Vina-valid column format), then append baseline dummy atoms for
# any AD4 type not present (so all maps get generated).
set -e
# a ligand with diverse atom types (aromatic, amide, amine, OH, S, halogens)
echo 'CC(=O)Nc1ccc(NC(C)C)cc1SCCBr' > kss.smi
/usr/bin/obabel kss.smi -ismi -O kss_base.pdbqt -xh --gen3d 2>&1 | tail -1
# keep only ATOM lines (drop ROOT/BRANCH/TORSDOF/REMARK)
grep -E '^ATOM' kss_base.pdbqt > kss_ref.pdbqt
echo "=== base atom types (obabel, Vina-valid) ==="
awk '{print $NF}' kss_ref.pdbqt | sort | uniq -c
echo "=== total ATOM lines: $(wc -l < kss_ref.pdbqt) ==="
