import os, sys
import pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
inh = pd.read_csv('data/cyp-challenge-TRAIN_inhibition.csv')
tdi = pd.read_csv('data/cyp-challenge-TRAIN_TDI.csv')
emx = pd.read_csv('data/cyp-challenge-TRAIN_Emax.csv')
sc = pd.read_csv('data/cyp-challenge-single-concentration-TRAIN.csv')
print('inh cols:', list(inh.columns))
print('tdi cols:', list(tdi.columns))
print('emx cols:', list(emx.columns))
print('sc cols:', list(sc.columns))
print('enzymes:', sc['enzyme'].value_counts().to_dict() if 'enzyme' in sc else 'N/A')
