#!/bin/bash
# Strict lineage confirmation: UMTM --weights ext on the L1 (nazarov) encoder
# must reproduce ft_ext's OOF within run-to-run noise (gate sanity, FINAL_PUSH_PLAN 1.2).
set -e
cd ~/cyp-challenge
~/miniforge3/envs/chemeleon/bin/python src/umtm_module.py --lineage L1 --seed 0 --weights ext --fold reg --no-test \
  > logs/umtm_gate_L1_ext.log 2>&1 || { echo "GATE_L1_EXT_FAILED"; exit 1; }
~/miniforge3/envs/cyp/bin/python src/umtm_eval_direct.py cache/umtm_oof_L1_ext.csv cache/ft_oof_ext.csv \
  | tee logs/umtm_gate_L1_ext_eval.txt
echo GATE_L1_EXT_COMPLETE
