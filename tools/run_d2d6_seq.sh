#!/bin/bash
# CYP2D6-specialist D-MPNN, seeds 7 then 8 (briford lever), sequential, one tracked process.
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemprop-dev/bin/python
for S in 7 8; do
  echo "=== d2d6 seed $S start $(date -u +%H:%M) ===" >> /tmp/cyp_d2d6_seq.log
  $PY src/dmpnn_2d6.py --seed $S >> /tmp/cyp_d2d6_seq.log 2>&1
  echo "=== d2d6 seed $S exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_d2d6_seq.log
done
touch /tmp/cyp_d2d6.done
