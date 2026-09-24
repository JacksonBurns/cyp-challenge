#!/bin/bash
# After queue2 (dmpnn -> tabicl_cp x2) finishes: cp ridge-probe embeddings for regression.
cd /home/jackson/cyp-challenge
while [ ! -f /tmp/cyp_gpuqueue.done ]; do sleep 60; done
echo "=== cp_embeddings start $(date -u +%H:%M) ===" > /tmp/cyp_gpuqueue4.log
~/miniforge3/envs/tabicl-chemeleon/bin/python src/cp_embeddings.py >> /tmp/cyp_gpuqueue4.log 2>&1
echo "=== cp_embeddings exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue4.log
touch /tmp/cyp_cpemb.done
