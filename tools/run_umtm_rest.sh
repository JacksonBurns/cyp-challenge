#!/bin/bash
# UMTM lineage chain: L1 strict gate -> L1 plan x4 seeds -> L1 eval -> L3 x4 -> L4 x4.
# One GPU job at a time; appends to logs/. Safe to re-run (skips existing outputs).
set -e
cd ~/cyp-challenge
CHE=~/miniforge3/envs/chemeleon/bin/python

if [ ! -f cache/umtm_oof_L1_ext.csv ]; then
  echo "=== GATE L1 ext (strict lineage repro of ft_ext) ==="
  $CHE src/umtm_module.py --lineage L1 --seed 0 --weights ext --fold reg --no-test > logs/umtm_gate_L1_ext.log 2>&1 || { echo GATE_L1_EXT_FAILED; exit 1; }
  ~/miniforge3/envs/cyp/bin/python src/umtm_eval_direct.py cache/umtm_oof_L1_ext.csv cache/ft_oof_ext.csv | tee logs/umtm_gate_L1_ext_eval.txt
else echo "GATE L1 ext already present"; fi

for LIN in L1 L3 L4; do
  for S in 0 1 2 3; do
    TAG=$LIN; [ $S -gt 0 ] && TAG=${LIN}_s$S
    if [ -f "cache/umtm_oof_${TAG}.csv" ]; then echo "$TAG complete, skip"; continue; fi
    echo "=== UMTM $LIN seed $S ==="
    $CHE src/umtm_module.py --lineage $LIN --seed $S --weights plan --fold reg > "logs/umtm_${TAG}.log" 2>&1 || { echo "UMTM_${TAG}_FAILED"; exit 1; }
  done
  echo "UMTM_${LIN}_CHAIN_COMPLETE"
done
echo ALL_LINEAGES_COMPLETE
