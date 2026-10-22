#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
echo "=== explicit ammonium [NH3+] -> PDBQT type ==="
echo 'CC[NH3+]' > am.smi
/usr/bin/obabel am.smi -ismi -O am.pdbqt -xh -p 7.4 2>&1 | tail -1
grep -E '^ATOM' am.pdbqt | awk '$2=="N"{print "N: charge="$11, "type="$NF}'
echo "=== neutral amine (no +) -> PDBQT type, -p 7.4 ==="
echo 'CCN' > nam.smi
/usr/bin/obabel nam.smi -ismi -O nam.pdbqt -xh -p 7.4 2>&1 | tail -1
grep -E '^ATOM' nam.pdbqt | awk '$2=="N"{print "N: charge="$11, "type="$NF}'
echo "=== neutral amine, NO -p (default) ==="
/usr/bin/obabel nam.smi -ismi -O nam2.pdbqt -xh 2>&1 | tail -1
grep -E '^ATOM' nam2.pdbqt | awk '$2=="N"{print "N: charge="$11, "type="$NF}'
