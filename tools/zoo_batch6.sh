#!/bin/bash
# Batch 6: multi-view arms (TDI-cond pIC50 + screening log2fc + is_TDI heads).
# Waits for the GPU to be free (user TabICL job runs ~6.5 days, ends ~Oct 30).
# One arm at a time; arms are ~1.2h each.
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemeleon/bin/python
run () {
  local tag="$1"; shift
  echo "=== $tag start $(date '+%m-%d %T')" >> /tmp/zoo_batch6.log
  $PY src/loss_zoo.py "$@" >> "/tmp/zoo_$tag.log" 2>&1
  echo "=== $tag exit=$? $(date '+%m-%d %T')" >> /tmp/zoo_batch6.log
}
gpu_idle () {
  # yield the GPU: wait until no compute apps are running
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 300; done
}
gpu_idle
run mv           --mode hinge --auxw 3.0 --mv
gpu_idle
run mv_s1        --mode hinge --auxw 3.0 --mv --seed 1
gpu_idle
run mv_s2        --mode hinge --auxw 3.0 --mv --seed 2
gpu_idle
run tdic         --mode hinge --auxw 3.0 --tdic
gpu_idle
run tdic_s1      --mode hinge --auxw 3.0 --tdic --seed 1
gpu_idle
run tdic_s2      --mode hinge --auxw 3.0 --tdic --seed 2
echo "BATCH6 DONE $(date '+%m-%d %T')" >> /tmp/zoo_batch6.log
