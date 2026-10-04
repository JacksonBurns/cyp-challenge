import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))

Xtr = pd.read_parquet('cache/X_train.parquet').drop_duplicates('SMILES').reset_index(drop=True)
inh = pd.read_csv('data/cyp-challenge-TRAIN_inhibition.csv').drop_duplicates('SMILES')
base = pd.DataFrame({'SMILES': Xtr['SMILES'].values})
inh_i = inh.set_index('SMILES')
d1 = inh_i['CYP1A2_pIC50_direct_inhibition'].reindex(base.SMILES)
print('type d1:', type(d1).__name__, '| shape:', np.shape(d1))
v = d1 if not isinstance(d1, pd.DataFrame) else d1.iloc[:, 0]
print('v index head:', v.index[:5].tolist())
print('base index head:', base.index[:5].tolist())
base['t'] = v
print('assigned notna:', int(base['t'].notna().sum()))
# also test duplicate columns
print('inh dup cols:', inh.columns.duplicated().sum())
