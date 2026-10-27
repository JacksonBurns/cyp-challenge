#!/bin/bash
# zoo12 seed-averaged family gates. Each ~15-20 min CPU.
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/cyp/bin/python
$PY src/zoo12_gate.py band > cache/zoo12_band_out.txt 2>&1; echo "Z12BAND=$?"
$PY src/zoo12_gate.py bandshuf > cache/zoo12_bandshuf_out.txt 2>&1; echo "Z12BANDS=$?"
$PY src/zoo12_gate.py w002 > cache/zoo12_w002_out.txt 2>&1; echo "Z12W002=$?"
$PY src/zoo12_gate.py w002shuf > cache/zoo12_w002shuf_out.txt 2>&1; echo "Z12W002S=$?"
