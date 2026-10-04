import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')

# replicate build_master step by step with prints
Xtr = pd.read_parquet('cache/X_train.parquet').drop_duplicates('SMILES').reset_index(drop=True)
inh = pd.read_csv('data/cyp-challenge-TRAIN_inhibition.csv').drop_duplicates('SMILES')
tdi = pd.read_csv('data/cyp-challenge-TRAIN_TDI.csv').drop_duplicates('SMILES')
base = pd.DataFrame({'SMILES': Xtr['SMILES'].values})
inh_i, tdi_i = inh.set_index('SMILES'), tdi.set_index('SMILES')
ISO = ['CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4']
for iso in ISO:
    d1 = inh_i[f'{iso}_pIC50_direct_inhibition'].reindex(base.SMILES)
    d2 = tdi_i[f'{iso}_pIC50_direct_inhibition'].reindex(base.SMILES)
    print(iso, 'd1:', int(d1.notna().sum()), 'd2:', int(d2.notna().sum()))
    v = d1.combine_first(d2)
    print('   combined:', int(v.notna().sum()))
    base[f'{iso}_direct'] = v
print('base direct notna:', int(base['CYP1A2_direct'].notna().sum()))
# now the emx/istdi section
emx = pd.read_csv('data/cyp-challenge-TRAIN_Emax.csv').drop_duplicates('SMILES')
lab = pd.concat([tdi[['SMILES'] + [f'{i}_is_TDI' for i in ['CYP2D6', 'CYP3A4']]],
                 emx[['SMILES'] + [f'{i}_is_TDI' for i in ['CYP2D6', 'CYP3A4']]]]).drop_duplicates('SMILES')
lab = lab.set_index('SMILES')
for i in ['CYP2D6', 'CYP3A4']:
    v = lab[f'{i}_is_TDI'].reindex(base.SMILES)
    base[f'is_TDI_{i}'] = v.map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0})
print('after is_TDI section: direct notna =', int(base['CYP1A2_direct'].notna().sum()))
print('dtypes:', base['CYP1A2_direct'].dtype, base['is_TDI_2D6'].dtype)
