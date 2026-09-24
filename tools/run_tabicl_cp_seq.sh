#!/bin/bash
# After cp embeddings done: TabICL-on-cpmed TDI, plain then with AID1851 in-context rows.
cd /home/jackson/cyp-challenge
while [ ! -f /tmp/cyp_cpemb.done ]; do sleep 30; done
echo "=== tabicl_cp start $(date -u +%H:%M) ===" > /tmp/cyp_tabiclcp.log
~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl_cp.py --ext 0 >> /tmp/cyp_tabiclcp.log 2>&1
echo "=== tabicl_cp exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_tabiclcp.log
echo "=== tabicl_cp_ext start $(date -u +%H:%M) ===" >> /tmp/cyp_tabiclcp.log
~/miniforge3/envs/tabicl-chemeleon/bin/python src/tdi_tabicl_cp.py --ext 1 >> /tmp/cyp_tabiclcp.log 2>&1
echo "=== tabicl_cp_ext exit=$? $(date -u +%H:%M) ===" >> /tmp/cyp_tabiclcp.log
touch /tmp/cyp_tabiclcp.done
