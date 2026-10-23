#!/bin/bash
# Loss zoo round 2: auxw sweep (rerun of crashed arms) + homolog-aux arm.
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemeleon/bin/python
run () {
  local tag="$1"; shift
  echo "=== $tag start $(date +%T)" >> /tmp/zoo_batch2.log
  $PY src/loss_zoo.py "$@" >> "/tmp/zoo_${tag}.log" 2>&1
  echo "=== $tag exit=$? $(date +%T)" >> /tmp/zoo_batch2.log
}
run hinge_w0  --mode hinge --auxw 0.0
run hinge_w1  --mode hinge --auxw 1.0
run hinge_w3  --mode hinge --auxw 3.0
run homarm    --mode hinge --hom
echo "BATCH2 DONE $(date +%T)" >> /tmp/zoo_batch2.log
