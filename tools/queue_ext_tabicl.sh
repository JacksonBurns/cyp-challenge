#!/bin/bash
# Sequential GPU jobs: ft_ext multitask, then TabICL TDI.
cd ~/cyp-challenge
~/miniforge3/envs/chemeleon/bin/python src/ft_ext.py --full-ft > /tmp/cyp_ft_ext.log 2>&1
echo "FT_EXT_DONE rc=$?"
~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl.py > /tmp/cyp_tdi_tabicl.log 2>&1
echo "TABICL_DONE rc=$?"
