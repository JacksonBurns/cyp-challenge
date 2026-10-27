#!/bin/bash
# zoo11 pool pair (ft_adme2 real + shuffled control), then A2+ alerts gate.
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/cyp/bin/python
$PY src/zoo11_gate.py real > cache/zoo11_gate_out.txt 2>&1; echo "Z11=$?"
$PY src/zoo11_gate.py shuf > cache/zoo11p_gate_out.txt 2>&1; echo "Z11P=$?"
$PY src/adme2_gates3.py > cache/adme2_gates3_out.txt 2>&1; echo "G3=$?"
