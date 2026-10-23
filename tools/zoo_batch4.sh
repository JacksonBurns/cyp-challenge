#!/bin/bash
# Loss zoo round 4: seed-averaging the hinge_w3 arm (top weight in zoo1b on
# all 4 isoforms) + homarm seeds. Runs after batch3 (single GPU, serial).
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemeleon/bin/python
run () {
  local tag="$1"; shift
  echo "=== $tag start $(date +%T)" >> /tmp/zoo_batch4.log
  $PY src/loss_zoo.py "$@" >> "/tmp/zoo_$tag.log" 2>&1
  echo "=== $tag exit=$? $(date +%T)" >> /tmp/zoo_batch4.log
}
while pgrep -f "zoo_batch[23].sh" > /dev/null; do sleep 60; done
run hinge_w3_s1 --mode hinge --auxw 3.0 --seed 1
run hinge_w3_s2 --mode hinge --auxw 3.0 --seed 2
run homarm_s1 --mode hinge --hom --seed 1
run homarm_s2 --mode hinge --hom --seed 2
echo "BATCH4 DONE $(date +%T)" >> /tmp/zoo_batch4.log
