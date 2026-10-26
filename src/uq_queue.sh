#!/bin/bash
# GPU queue: runs after pcg1 (pcgrad_arm full) exits; one job at a time.
set -u
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemeleon/bin/python

# 1) wait for pcg1 (pid from arg)
PCG_PID=${1:-}
if [ -n "$PCG_PID" ]; then
  while kill -0 "$PCG_PID" 2>/dev/null; do sleep 20; done
  echo "[queue] pcg1 exited; $(date)"
fi

echo "[queue] uqw baseline ctl seed0 $(date)"
$PY src/uq_weight_arm.py --baseline --seed 0 2>&1 | grep -v Warning | tail -12

echo "[queue] uqw weighted q90 w50 seed0 $(date)"
$PY src/uq_weight_arm.py --q 0.9 --wdw 0.5 --iso all --seed 0 2>&1 | grep -v Warning | tail -12

echo "[queue] uqw weighted 2D6-only q90 w50 seed0 $(date)"
$PY src/uq_weight_arm.py --q 0.9 --wdw 0.5 --iso CYP2D6 --seed 0 2>&1 | grep -v Warning | tail -12

echo "[queue] ALL UQ JOBS DONE $(date)"
