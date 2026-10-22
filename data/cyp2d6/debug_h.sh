#!/bin/bash
cd /home/jackson/cyp-challenge/data/cyp2d6
echo "=== raw dm_test.pdbqt (first 20 lines) ==="
head -20 dm_test.pdbqt
echo "=== total lines: $(wc -l < dm_test.pdbqt) ==="
echo
echo "=== TEST 1: molblock WITH H -> pdbqt, explicit -x ==="
~/miniforge3/envs/cyp/bin/python -c "
from rdkit import Chem
from rdkit.Chem import AllChem
mol = Chem.MolFromSmiles('CN1CC[C@@]2(c3ccccc3O2)C1')
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=0)
AllChem.MMFFOptimizeMolecule(mol)
open('dm_h.mol','w').write(Chem.MolToMolBlock(mol))
print('molblock written, atoms:', mol.GetNumAtoms())
" 2>&1 | grep -v "explicit"
echo "  obabel -x:        $(/usr/bin/obabel dm_h.mol -O /tmp/t1.pdbqt -x 2>/dev/null; grep -cE '^ATOM' /tmp/t1.pdbqt)"
echo "  obabel (no -x):   $(/usr/bin/obabel dm_h.mol -O /tmp/t2.pdbqt 2>/dev/null; grep -cE '^ATOM' /tmp/t2.pdbqt)"
echo "  obabel -xh:       $(/usr/bin/obabel dm_h.mol -O /tmp/t3.pdbqt -xh 2>/dev/null; grep -cE '^ATOM' /tmp/t3.pdbqt)"
echo "  obabel --gen3d -x:$(/usr/bin/obabel dm_h.mol -O /tmp/t4.pdbqt --gen3d -x 2>/dev/null; grep -cE '^ATOM' /tmp/t4.pdbqt)"
