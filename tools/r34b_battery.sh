#!/bin/bash
# zoo14 dual-family pool gates; fires after zoo13 real gate lands (avg csv exists).
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/cyp/bin/python
while [ ! -f cache/ft_oof_extw005_avg.csv ]; do sleep 60; done
for m in ext extshuf both bothshuf; do
  $PY src/zoo14_gate.py $m > cache/zoo14_${m}_out.txt 2>&1; echo "Z14_$m=$?"
done
