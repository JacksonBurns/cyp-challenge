import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')
tdi = pd.read_csv('data/cyp-challenge-TRAIN_TDI.csv').drop_duplicates('SMILES')
emx = pd.read_csv('data/cyp-challenge-TRAIN_Emax.csv').drop_duplicates('SMILES')
Xtr = pd.read_parquet('cache/X_train.parquet').drop_duplicates('SMILES').reset_index(drop=True)
base = pd.DataFrame({'SMILES': Xtr['SMILES'].values})
lab = pd.concat([tdi[['SMILES'] + [f'{i}_is_TDI' for i in ['CYP2D6', 'CYP3A4']]],
                 emx[['SMILES'] + [f'{i}_is_TDI' for i in ['CYP2D6', 'CYP3A4']]]]).drop_duplicates('SMILES')
print('lab rows:', len(lab), '| dup smiles in tdi+emx overlap ok')
lab = lab.set_index('SMILES')
v = lab['CYP2D6_is_TDI'].reindex(base.SMILES)
print('v notna:', int(v.notna().sum()), '| sample vals:', v.dropna().unique()[:4])
m = v.map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0})
print('mapped notna:', int(m.notna().sum()))
base['is_TDI_2D6'] = m.values
print('assigned notna:', int(base['is_TDI_2D6'].notna().sum()))
