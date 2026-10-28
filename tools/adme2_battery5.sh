#!/bin/bash
# zoo12 band pair on 14-seed averaged family (decisive pool test).
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/cyp/bin/python
$PY src/zoo12_gate.py band > cache/zoo12_band14_out.txt 2>&1; echo "Z12B14=$?"
$PY src/zoo12_gate.py bandshuf > cache/zoo12_band14shuf_out.txt 2>&1; echo "Z12B14S=$?"
