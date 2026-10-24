import pandas as pd
sc = pd.read_csv('data/cyp-challenge-single-concentration-TRAIN.csv')
print('enzymes:', sc['enzyme'].unique().tolist())
print('cpds:', sc['SMILES'].nunique(), 'rows:', len(sc))
t = pd.read_csv('data/cyp-challenge-TRAIN_TDI.csv')
print('tdi cond cols:', [c for c in t.columns if 'TDI_condition' in c and 'conf' not in c and 'std' not in c])
print('non-null tdi cond 1A2:', t['CYP1A2_pIC50_TDI_condition'].notna().sum())
