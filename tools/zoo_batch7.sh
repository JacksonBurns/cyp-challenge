#!/bin/bash
# Batch 7: classification-head weight experiment + Emax full multitask arm.
# mv at clsw=0.3 tests the starved-2D6 mechanism; mvx adds Emax x4 + is_TDI x4.
cd /home/jackson/cyp-challenge
PY=~/miniforge3/envs/chemeleon/bin/python
gpu_idle () {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 300; done
}
run () {
  tag=$1; shift
  [ -f cache/zoo_eval_$tag.json ] && { echo "skip $tag"; return; }
  gpu_idle
  echo "=== $tag $(date)"
  $PY src/loss_zoo.py --mode hinge --auxw 3.0 "$@" > /tmp/zoo_$tag.log 2>&1 \
    && echo "done $tag" || echo "FAIL $tag"
}
run mvx            --mv --emx
run mvx_cw0.3      --mv --emx --clsw 0.3
run mv_cw0.3       --mv --clsw 0.3
run mv_cw0.3_s1    --mv --clsw 0.3 --seed 1
run mv_cw0.3_s2    --mv --clsw 0.3 --seed 2
run mvx_cw0.3_s1   --mv --emx --clsw 0.3 --seed 1
run mvx_cw0.3_s2   --mv --emx --clsw 0.3 --seed 2
echo BATCH7_DONE
