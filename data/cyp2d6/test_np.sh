#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
# Find which aromatic-N reference produces N_P.map
try () {
  local smi="$1"; local tag="$2"
  echo "$smi" > /tmp/n_$tag.smi
  /usr/bin/obabel /tmp/n_$tag.smi -ismi -O /tmp/n_$tag.pdbqt -xh --gen3d >/dev/null 2>&1
  rm -f /tmp/nm_$tag.*.map
  ~/miniforge3/envs/cyp/bin/vina --receptor cyp2d6_target_flat.pdbqt --ligand /tmp/n_$tag.pdbqt \
    --write_maps /tmp/nm_$tag --center_x -12.796 --center_y 50.784 --center_z -6.164 \
    --size_x 22 --size_y 20 --size_z 20 --spacing 0.375 --force_even_voxels >/dev/null 2>&1
  echo "$tag: $(ls /tmp/nm_$tag.*.map 2>/dev/null | sed 's/.*nm_'$tag'\.//' | sort | tr '\n' ' ')"
}
try 'c1ccncc1' pyridine
try 'c1ccnc(C)c1' methylpyridine
try 'C1=CN=CN=C1' pyrimidine
try 'c1ccc2ncnc2c1' quinazoline
try 'CCOC(=O)N1CSCC1C(=O)O' failing_ligand
echo "=== obabel types for pyridine N ==="
echo 'c1ccncc1' > /tmp/p.smi; /usr/bin/obabel /tmp/p.smi -ismi -O /tmp/p.pdbqt -xh --gen3d >/dev/null 2>&1
grep -E '^ATOM' /tmp/p.pdbqt | awk '$2=="N"{print "  N type="$NF" charge="$11}'
