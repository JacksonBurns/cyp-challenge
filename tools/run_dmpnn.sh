#!/bin/bash
# Stage 3: D-MPNN multitask, scaffold folds, seed 7 (GPU).
cd /home/jackson/cyp-challenge
echo "=== dmpnn start $(date -u +%H:%M) ===" > /tmp/cyp_dmpnn.log
~/miniforge3/envs/chemprop-dev/bin/python src/ft_dmpnn.py --seed 7 >> /tmp/cyp_dmpnn.log 2>&1
echo "=== dmpnn exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_dmpnn.log
touch /tmp/cyp_dmpnn.done
