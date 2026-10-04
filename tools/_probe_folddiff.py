import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')
from cypfolds import scaffold_groups, make_folds
import run_regression as R  # cyp env has lightgbm
ft = pd.read_csv('cache/ft_data.csv')
g_new = scaffold_groups(ft['SMILES'].tolist())
g_old = R.scaffold_groups(ft['SMILES'].tolist())
diff = [i for i in range(len(g_new)) if g_new[i] != g_old[i]]
print('group diffs:', len(diff))
if diff:
    i = diff[0]
    print(ft['SMILES'].iloc[i], '| new:', repr(g_new[i]), '| old:', repr(g_old[i]))
f_new = make_folds(g_new, seed=0)
print('fold match despite group diff:', (f_new == ft['fold'].values).mean())
