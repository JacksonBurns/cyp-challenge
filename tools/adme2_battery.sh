#!/bin/bash
# ADME2 CPU battery: TDI classification gate, then zoo10 pool gate pair.
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/cyp/bin/python
$PY src/tdi_adme2_gate.py > cache/tdi_adme2_out.txt 2>&1; echo "TDI=$?"
$PY src/zoo10_gate.py real > cache/zoo10_gate_out.txt 2>&1; echo "Z10=$?"
$PY src/zoo10_gate.py shuf > cache/zoo10p_gate_out.txt 2>&1; echo "Z10P=$?"
