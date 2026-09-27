#!/usr/bin/env bash
# Re-run audit2 (adds per-fold capped weights to the json) then fraction v6.
set -euo pipefail
cd /home/jackson/cyp-challenge
~/miniforge3/envs/cyp/bin/python src/tdi_blend_family_block2.py > /tmp/fb2_rerun.log
~/miniforge3/envs/cyp/bin/python src/tdi_fraction_opt_v6.py > /tmp/frac_v6_rerun.log
echo DONE
