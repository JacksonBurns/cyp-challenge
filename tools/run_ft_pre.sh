#!/bin/bash
# Stage 2: ft_ext with pretrained MP init, seed 0 then seed 1 (sequential GPU).
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemeleon/bin/python
for S in 0 1; do
  echo "=== pre seed $S start $(date -u +%H:%M) ===" >> /tmp/cyp_ft_pre_seq.log
  $PY src/ft_ext.py --full-ft --seed $S --pretrained cache/mp_pretrained_cyp.pt --prefix pre >> /tmp/cyp_ft_pre_seq.log 2>&1
  echo "=== pre seed $S exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_ft_pre_seq.log
done
touch /tmp/cyp_ft_pre.done
