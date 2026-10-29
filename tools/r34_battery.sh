#!/bin/bash
# R3-4 battery: avg+shuf extw005, then zoo13 pool gate (real, shuf).
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/cyp/bin/python
$PY tools/shuf_extw005.py > cache/extw005_avg_out.txt 2>&1; echo "AVG=$?"
$PY src/zoo13_gate.py real > cache/zoo13_real_out.txt 2>&1; echo "Z13R=$?"
$PY src/zoo13_gate.py shuf > cache/zoo13_shuf_out.txt 2>&1; echo "Z13S=$?"
