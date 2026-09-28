#!/usr/bin/env bash
# Task 2 chain: jeremy adme_pretrain encoders FULL-TUNED on the ft_ext 12-head
# recipe (regression). ckpt adapted via src/adapt_adme_ckpt.py. Separate tags
# per encoder: admchm (chemprop_chemeleon d_h=2048) / admmed (medium d_h=600),
# seeds 0-1. Run AFTER the Task 1 chain (single shared GPU).
set -x
cd "$HOME/cyp-challenge"
PY="$HOME/miniforge3/envs/chemeleon/bin/python"
for seed in 0 1; do
  sfx=""; [ "$seed" != 0 ] && sfx="_s$seed"
  $PY src/ft_ext.py --full-ft --prefix admchm --seed $seed \
    --pretrained cache/admecd_ckpt_chm.pt || { echo "TASK2_CHM_SEED${seed}_FAILED"; exit 1; }
done
for seed in 0 1; do
  $PY src/ft_ext.py --full-ft --prefix admmed --seed $seed \
    --pretrained cache/admecd_ckpt_med.pt || { echo "TASK2_MED_SEED${seed}_FAILED"; exit 1; }
done
echo TASK2_CHAIN_COMPLETE
