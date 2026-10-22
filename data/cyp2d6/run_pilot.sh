#!/bin/bash
cd /home/jackson/cyp-challenge
echo "=== regenerate kitchen-sink reference ==="
timeout 300 ~/miniforge3/envs/cyp/bin/python data/cyp2d6/make_kitchen_sink.py /tmp/all_smis.txt 150 data/cyp2d6/kss_ref.pdbqt 2>&1 | grep -vE "does not have explicit|Warning" | tail -3
echo "=== reference types ==="
grep -E '^ATOM' data/cyp2d6/kss_ref.pdbqt | awk '{print $NF}' | sort | tr '\n' ' '
echo
echo "=== clear old maps + pycache ==="
rm -f data/cyp2d6/cyp2d6_maps.*.map
find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "=== run 30-ligand pilot (rebuilds maps) ==="
timeout 500 ~/miniforge3/envs/cyp/bin/python src/dock_features.py --smiles /tmp/pilot_smis.txt --out /tmp/dockfeat_test.csv --n 30 --workers 12 2>&1 | grep -vE "does not have explicit|Warning"
