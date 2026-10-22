#!/bin/bash
cd /home/jackson/cyp-challenge
echo "=== build kitchen-sink reference (real obabel ligands) ==="
timeout 120 ~/miniforge3/envs/cyp/bin/python data/cyp2d6/make_ref.py 2>&1 | grep -vE "does not have explicit|Warning"
echo "=== clear old maps + pycache ==="
rm -f data/cyp2d6/cyp2d6_maps.*.map
find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "=== run 30-ligand pilot (rebuilds maps) ==="
timeout 500 ~/miniforge3/envs/cyp/bin/python src/dock_features.py --smiles /tmp/pilot_smis.txt --out /tmp/dockfeat_test.csv --n 30 --workers 12 2>&1 | grep -vE "does not have explicit|Warning"
