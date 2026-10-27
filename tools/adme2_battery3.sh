#!/bin/bash
# zoo11 avg-family gates, queued behind battery2 (zoo11 real+shuf).
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/cyp/bin/python
while ! [ -f cache/hinge_blend_zoo11p.json ]; do sleep 60; done
$PY src/zoo11_gate.py avg > cache/zoo11a_gate_out.txt 2>&1; echo "Z11A=$?"
while ! [ -f cache/hinge_blend_zoo11p.json ]; do sleep 30; done
$PY src/zoo11_gate.py avgshuf > cache/zoo11ap_gate_out.txt 2>&1; echo "Z11AP=$?"
