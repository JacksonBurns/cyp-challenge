#!/bin/bash
# GPU queue after ft_pre finishes: cp embeddings, D-MPNN seed 7, TabICL-on-cpmed (TDI, both variants).
cd /home/jackson/cyp-challenge
while [ ! -f /tmp/cyp_ft_pre.done ]; do sleep 60; done
echo "=== cp_embeddings start $(date -u +%H:%M) ===" > /tmp/cyp_gpuqueue.log
~/miniforge3/envs/tabicl-chemeleon/bin/python src/cp_embeddings.py >> /tmp/cyp_gpuqueue.log 2>&1
echo "=== cp_embeddings exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue.log
echo "=== dmpnn start $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue.log
~/miniforge3/envs/chemprop-dev/bin/python src/ft_dmpnn.py --seed 7 >> /tmp/cyp_gpuqueue.log 2>&1
echo "=== dmpnn exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue.log
echo "=== tabicl_cp start $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue.log
~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl_cp.py --ext 0 >> /tmp/cyp_gpuqueue.log 2>&1
echo "=== tabicl_cp exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue.log
echo "=== tabicl_cp_ext start $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue.log
~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl_cp.py --ext 1 >> /tmp/cyp_gpuqueue.log 2>&1
echo "=== tabicl_cp_ext exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue.log
touch /tmp/cyp_gpuqueue.done
