#!/bin/bash
# Sequential GPU: ft_ext all-data test model (folds reused), then TabICL TDI.
cd ~/cyp-challenge
~/miniforge3/envs/chemeleon/bin/python src/ft_ext.py --full-ft --skip-folds > /tmp/cyp_ft_ext_test.log 2>&1
echo "FT_EXT_TEST_DONE rc=$?"
~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl.py > /tmp/cyp_tdi_tabicl.log 2>&1
echo "TABICL_DONE rc=$?"
