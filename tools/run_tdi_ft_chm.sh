#!/usr/bin/env bash
# Task 1 chain: CheMeLeon encoder fine-tuned on TDI binary task, seed 0.
# fullft first, then frozen head-only (both ~20-40 min each on the shared GPU).
set -x
cd "$HOME/cyp-challenge"
PY="$HOME/miniforge3/envs/chemeleon/bin/python"
$PY src/tdi_ft_chemeleon.py --seed 0 --variant fullft || { echo "TASK1_FULLFT_FAILED"; exit 1; }
$PY src/tdi_ft_chemeleon.py --seed 0 --variant frozen || { echo "TASK1_FROZEN_FAILED"; exit 1; }
echo TASK1_CHAIN_COMPLETE
