#!/bin/bash
# UMTM Phase 1 gates -> Phase 2 L2 chain (FINAL_PUSH_PLAN sec 3).
# One GPU job at a time; never kill anything else. Logs -> logs/.
set -e
cd ~/cyp-challenge
CHE=~/miniforge3/envs/chemeleon/bin/python
CYP=~/miniforge3/envs/cyp/bin/python

run_gate() {  # $1=weights $2=baseline csv
  if [ -f "cache/umtm_oof_L2_$1.csv" ]; then echo "GATE $1 already present, skipping train"; return 0; fi
  $CHE src/umtm_module.py --lineage L2 --seed 0 --weights "$1" --fold reg --no-test \
    > "logs/umtm_gate_L2_$1.log" 2>&1 || { echo "GATE_${1}_TRAIN_FAILED"; exit 1; }
  $CYP src/umtm_eval_direct.py "cache/umtm_oof_L2_$1.csv" "$2" | tee "logs/umtm_gate_L2_${1}_eval.txt"
}

echo "=== GATE ext (zeroed new heads; must reproduce ft_ext OOF within noise) ==="
run_gate ext cache/ft_oof_ext.csv
echo "=== GATE primary (direct-only; must reproduce ft_fullft OOF within noise) ==="
run_gate primary cache/ft_oof_fullft.csv

for S in 0 1 2 3; do
  if [ -f "cache/umtm_oof_L2_s${S}.csv" ] && [ -f "cache/umtm_test_L2_s${S}.csv" ]; then
    echo "L2 seed $S already complete, skipping"; continue
  fi
  echo "=== PHASE 2: L2 plan seed $S ==="
  $CHE src/umtm_module.py --lineage L2 --seed $S --weights plan --fold reg \
    > "logs/umtm_L2_s${S}.log" 2>&1 || { echo "UMTM_L2_SEED${S}_FAILED"; exit 1; }
done
echo "UMTM_L2_CHAIN_COMPLETE"
