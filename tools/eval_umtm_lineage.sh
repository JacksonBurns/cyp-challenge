#!/bin/bash
# Phase-2-style evaluation of a lineage's UMTM OOF files on BOTH tracks:
#  - regression: direct heads as standalone singles (+ via cp13 nested blend, run separately)
#  - TDI: is_TDI heads vs the legacy singles bar (go/no-go), per-seed + seed-avg
# Usage: bash tools/eval_umtm_lineage.sh L2
set -e
cd ~/cyp-challenge
LIN=$1
CYP=~/miniforge3/envs/cyp/bin/python
FILES=$(ls cache/umtm_oof_${LIN}_s[0-9].csv cache/umtm_oof_${LIN}.csv 2>/dev/null || true)
[ -z "$FILES" ] && { echo "no OOF files for $LIN"; exit 1; }
echo "== $LIN standalone direct-head singles (regression track) =="
$CYP src/umtm_eval_direct.py $FILES cache/ft_oof_admchm.csv cache/ft_oof_admmed.csv cache/ft_oof_ext.csv
echo "== $LIN TDI heads vs legacy bar (go/no-go; nested on seed-7) =="
$CYP src/umtm_eval_tdi.py $FILES cache/umtm_eval_legacy_lgbm.csv cache/umtm_eval_legacy_ft_chm.csv
echo "-- seed-averaged probabilities --"
$CYP src/umtm_eval_tdi.py --avg $FILES
