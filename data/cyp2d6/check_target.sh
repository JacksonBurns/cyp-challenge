#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
echo "=== receptor PDBQT ==="
echo "atoms: $(grep -cE '^ATOM' cyp2d6_target.pdbqt)"
echo "HEM: $(grep -c HEM cyp2d6_target.pdbqt)"
echo "FE line:"
grep -iE ' FE ' cyp2d6_target.pdbqt | head -1
echo "=== obabel warnings count ==="
grep -c "WARNING" obabel_err.log 2>/dev/null || echo 0
