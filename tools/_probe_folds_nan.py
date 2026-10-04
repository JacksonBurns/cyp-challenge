import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')

Xtr = pd.read_parquet('cache/X_train.parquet').drop_duplicates('SMILES').reset_index(drop=True)
ft = pd.read_csv('cache/ft_data.csv')
print('Xtr order == ft_data SMILES order:', (Xtr['SMILES'].values == ft['SMILES'].values).all())
import run_regression as R
f7 = R.make_folds(R.scaffold_groups(Xtr['SMILES'].tolist()), seed=7)
f0 = R.make_folds(R.scaffold_groups(Xtr['SMILES'].tolist()))
print('seed7 folds: sizes', np.bincount(f7), 'seed0 sizes', np.bincount(f0))
np.save('/tmp/folds_seed7_umtm.npy', f7)

# NaN handling of chemprop normalize_targets
from chemprop import data, featurizers
from sklearn.preprocessing import StandardScaler
import numpy as np
pts = [data.MoleculeDatapoint.from_smi('CCO', [1.0, np.nan]),
       data.MoleculeDatapoint.from_smi('CCCO', [3.0, 10.0]),
       data.MoleculeDatapoint.from_smi('CCCCO', [np.nan, np.nan])]
d = data.MoleculeDataset(pts, featurizers.SimpleMoleculeMolGraphFeaturizer())
y0 = np.array([p.targets for p in d])
sc = d.normalize_targets()
y1 = np.array([p.targets for p in d])
print('mean_', sc.mean_, 'scale_', sc.scale_)
print('before:', y0[0], ' after:', y1[0])
