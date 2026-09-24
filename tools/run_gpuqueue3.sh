#!/bin/bash
# GPU queue3: cp_embeddings now, then D-MPNN seed 7, then TabICL-cp (plain + ext).
cd /home/jackson/cyp-challenge
echo "=== cp_embeddings start $(date -u +%H:%M) ===" > /tmp/cyp_gpuqueue3.log
~/miniforge3/envs/tabicl-chemeleon/bin/python src/cp_embeddings.py >> /tmp/cyp_gpuqueue3.log 2>&1
echo "=== cp_embeddings exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue3.log
echo "=== dmpnn start $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue3.log
~/miniforge3/envs/chemprop-dev/bin/python src/ft_dmpnn.py --seed 7 >> /tmp/cyp_gpuqueue3.log 2>&1
echo "=== dmpnn exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue3.log
echo "=== tabicl_cp start $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue3.log
~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl_cp.py --ext 0 >> /tmp/cyp_gpuqueue3.log 2>&1
echo "=== tabicl_cp exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue3.log
echo "=== tabicl_cp_ext start $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue3.log
~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl_cp.py --ext 1 >> /tmp/cyp_gpuqueue3.log 2>&1
echo "=== tabicl_cp_ext exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_gpuqueue3.log
touch /tmp/cyp_gpuqueue3.done
