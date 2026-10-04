import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')
from cypfolds import scaffold_groups, make_folds
ft = pd.read_csv('cache/ft_data.csv')
g = scaffold_groups(ft['SMILES'].tolist())
f = make_folds(g, seed=0)
m = (f == ft['fold'].values)
print('env fold match:', m.mean(), '| mismatches:', (~m).sum())
if not m.all():
    bad = np.where(~m)[0]
    for i in bad[:5]:
        print(i, ft['SMILES'].iloc[i], '| grp:', repr(g[i]), '| got', f[i], 'want', ft['fold'].iloc[i])
    # group sizes for mismatched groups
    from collections import Counter
    gb = Counter([g[i] for i in bad])
    print('mismatched group sizes sample:', list(gb.items())[:5])
