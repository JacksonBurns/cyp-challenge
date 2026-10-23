#!/bin/bash
# Batch 5: pretrained-encoder arms, gated on cache/pt_encoder.pt existing.
# Runs after all other GPU batches (serial single-GPU discipline).
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemeleon/bin/python
run () {
  local tag="$1"; shift
  echo "=== $tag start $(date +%T)" >> /tmp/zoo_batch5.log
  $PY src/loss_zoo.py "$@" >> "/tmp/zoo_$tag.log" 2>&1
  echo "=== $tag exit=$? $(date +%T)" >> /tmp/zoo_batch5.log
}
while pgrep -f "zoo_batch[234].sh" > /dev/null; do sleep 60; done
until [ -f cache/pt_encoder.pt ]; do sleep 60; done
run pt_mae       --mode mae --pt
run pt_mae_s1    --mode mae --pt --seed 1
run pt_mae_s2    --mode mae --pt --seed 2
run pt_hinge_w3  --mode hinge --auxw 3.0 --pt
run pt_hinge_w3_s1 --mode hinge --auxw 3.0 --pt --seed 1
run pt_hinge_w3_s2 --mode hinge --auxw 3.0 --pt --seed 2
run pt_homarm    --mode hinge --hom --pt
run pt_homarm_s1 --mode hinge --hom --pt --seed 1
run pt_homarm_s2 --mode hinge --hom --pt --seed 2
echo "BATCH5 DONE $(date +%T)" >> /tmp/zoo_batch5.log
