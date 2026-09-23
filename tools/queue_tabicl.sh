#!/bin/bash
# Wait for the running ft_chemeleon GPU job to finish, then run TabICL TDI.
while pgrep -f "ft_chemeleon.py --full-ft" > /dev/null; do sleep 30; done
cd ~/cyp-challenge && ~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl.py > /tmp/cyp_tdi_tabicl.log 2>&1
echo "TABICL_DONE rc=$?"
