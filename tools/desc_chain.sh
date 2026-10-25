#!/bin/bash
# Descriptor arm chain: GBM OOF (cyp env) -> ST-RAE eval (chemeleon env).
cd /home/jackson/cyp-challenge
while [ ! -f cache/desc3d_test.csv ]; do sleep 60; done
sleep 5
echo "=== desc_gbm $(date)"
~/miniforge3/envs/cyp/bin/python tools/desc_gbm.py > /tmp/desc_gbm.log 2>&1 \
  && echo gbm_ok || echo gbm_FAIL
echo "=== tag_eval $(date)"
~/miniforge3/envs/chemeleon/bin/python tools/tag_eval.py dsc3d > /tmp/tag_eval.log 2>&1 \
  && echo eval_ok || echo eval_FAIL
echo CHAIN_DONE
