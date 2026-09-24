#!/bin/bash
# ft_ext seeds 2-4, sequential, one tracked process chain.
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemeleon/bin/python
for S in 2 3 4; do
  echo "=== seed $S start $(date -u +%H:%M) ===" >> /tmp/cyp_ft_ext_seq.log
  $PY src/ft_ext.py --full-ft --seed $S >> /tmp/cyp_ft_ext_seq.log 2>&1
  echo "=== seed $S exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_ft_ext_seq.log
done
touch /tmp/cyp_ft_ext_s234.done
