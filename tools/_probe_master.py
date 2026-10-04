import os, sys
import pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
m = pd.read_csv('cache/umtm_master.csv')
ch = m[m['kind'] == 'challenge']
print('challenge rows:', len(ch))
print('CYP1A2_direct notna in csv:', int(ch['CYP1A2_direct'].notna().sum()))
print(m[['SMILES', 'kind', 'CYP1A2_direct', 'CYP1A2_tdic']].head(3).to_string())
