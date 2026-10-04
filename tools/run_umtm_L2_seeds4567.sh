#!/bin/bash
# Phase 6 (plan sec 3): 4 more seeds for the winning lineage L2 (nested still
# climbing: cp13 +0.0063, cp14 +0.0018, cp15/cp16 hold). One GPU job at a time.
set -e
cd ~/cyp-challenge
CHE=~/miniforge3/envs/chemeleon/bin/python
for S in 4 5 6 7; do
    if [ -f "cache/umtm_oof_L2_s${S}.csv" ]; then echo "L2_s${S} complete, skip"; continue; fi
    echo "=== UMTM L2 seed $S ==="
    $CHE src/umtm_module.py --lineage L2 --seed $S --weights plan --fold reg > "logs/umtm_L2_s${S}.log" 2>&1 || { echo "UMTM_L2_s${S}_FAILED"; exit 1; }
done
echo UMTM_L2_SEEDS_4567_COMPLETE
