#!/bin/bash
# Loss zoo GPU batch (LEDGER.md item 1): serial, one job at a time.
# Gate = marginal cp18i pool gain, not standalone. Screening seed 0.
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemeleon/bin/python
run () {
  local tag="$1"; shift
  echo "=== $tag start $(date +%T)" >> /tmp/zoo_batch.log
  $PY src/loss_zoo.py "$@" >> "/tmp/zoo_${tag}.log" 2>&1
  echo "=== $tag exit=$? $(date +%T)" >> /tmp/zoo_batch.log
}
run mae       --mode mae
run rank      --mode rank
run mse       --mode mse
run hinge_w0  --mode hinge --auxw 0.0
run hinge_w1  --mode hinge --auxw 1.0
run hinge_w3  --mode hinge --auxw 3.0
echo "BATCH DONE $(date +%T)" >> /tmp/zoo_batch.log
