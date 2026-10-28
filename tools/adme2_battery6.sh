#!/bin/bash
# Endpoint-group ablation (last feature-side readout).
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/cyp/bin/python
$PY src/adme2_ablate.py > cache/adme2_ablate_out.txt 2>&1; echo "ABL=$?"
