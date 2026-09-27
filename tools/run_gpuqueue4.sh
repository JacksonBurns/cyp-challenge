#!/usr/bin/env bash
# Step 1b GPU queue (sec 16): TabICL on cpchm embeddings, then D-MPNN classifier.
# One tracked process; sequential so only one heavy GPU job runs at a time.
set -euo pipefail
cd /home/jackson/cyp-challenge

echo "=== [1/2] TabICL on chemprop_chemeleon (cpchm) $(date -u +%H:%M:%S)"
~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl_cp2.py --src cpchm
echo "=== [2/2] D-MPNN classifier seed 7 $(date -u +%H:%M:%S)"
~/miniforge3/envs/chemprop-dev/bin/python src/tdi_dmpnn.py --seed 7
echo "=== QUEUE DONE $(date -u +%H:%M:%S)"
