#!/bin/bash
# zoo12 9-seed band pair on rebuilt family.
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/cyp/bin/python
$PY src/zoo12_gate.py band > cache/zoo12_band9_out.txt 2>&1; echo "Z12B9=$?"
$PY src/zoo12_gate.py bandshuf > cache/zoo12_band9shuf_out.txt 2>&1; echo "Z12B9S=$?"
$PY src/zoo12_gate.py w002 > cache/zoo12_w002b_out.txt 2>&1; echo "Z12W002B=$?"
$PY src/zoo12_gate.py w002shuf > cache/zoo12_w002bshuf_out.txt 2>&1; echo "Z12W002BS=$?"
