import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')
import umtm_data as U

Xtr = pd.read_parquet('cache/X_train.parquet').drop_duplicates('SMILES').reset_index(drop=True)
inh = pd.read_csv('data/cyp-challenge-TRAIN_inhibition.csv').drop_duplicates('SMILES')
tdi = pd.read_csv('data/cyp-challenge-TRAIN_TDI.csv').drop_duplicates('SMILES')
base = pd.DataFrame({'SMILES': Xtr['SMILES'].values})
inh_i, tdi_i = inh.set_index('SMILES'), tdi.set_index('SMILES')
iso = 'CYP1A2'
d1 = inh_i[f'{iso}_pIC50_direct_inhibition'].reindex(base.SMILES)
print('d1 notna:', int(d1.notna().sum()), '| inh col notna:', int(inh[f'{iso}_pIC50_direct_inhibition'].notna().sum()))
print('inh_i index sample:', inh_i.index[:3].tolist())
print('base sample:', base.SMILES[:3].tolist())
d1v = inh_i[f'{iso}_pIC50_direct_inhibition']
print('d1v dtype:', d1v.dtype)
print('dup index?', inh_i.index.duplicated().sum(), tdi_i.index.duplicated().sum())
