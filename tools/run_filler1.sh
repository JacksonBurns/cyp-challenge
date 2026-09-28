#!/usr/bin/env bash
# Task 4 filler (GPU idle slots): ft_ext seed 5 + d2d6 seed 10. ~+0.001 each.
set -x
cd "$HOME/cyp-challenge"
CYP="$HOME/miniforge3/envs/chemprop-dev/bin/python"
CHE="$HOME/miniforge3/envs/chemeleon/bin/python"
$CHE src/ft_ext.py --full-ft --seed 5 || { echo "FILLER_EXT5_FAILED"; exit 1; }
$CYP src/dmpnn_2d6.py --seed 10 || { echo "FILLER_D2D6_10_FAILED"; exit 1; }
echo TASK4_FILLER_COMPLETE
