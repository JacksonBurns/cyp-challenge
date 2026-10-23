#!/bin/bash
# Loss zoo round 3: seed-averaging the mae family (gate unit = family avg,
# per intarm lesson). Run after batch2 (single GPU, serial).
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemeleon/bin/python
run () {
  local tag="$1"; shift
  echo "=== $tag start $(date +%T)" >> /tmp/zoo_batch3.log
  $PY src/loss_zoo.py --mode mae --seed "$1" >> "/tmp/zoo_mae_s$1.log" 2>&1
  echo "=== $tag exit=$? $(date +%T)" >> /tmp/zoo_batch3.log
}
while pgrep -f zoo_batch2.sh > /dev/null; do sleep 60; done
run mae_s1 1
run mae_s2 2
echo "BATCH3 DONE $(date +%T)" >> /tmp/zoo_batch3.log
