import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
tdi = pd.read_csv('data/cyp-challenge-TRAIN_TDI.csv')
emx = pd.read_csv('data/cyp-challenge-TRAIN_Emax.csv')
for df, nm in [(tdi, 'tdi'), (emx, 'emx')]:
    c = df['CYP2D6_is_TDI']
    print(nm, c.dtype, '| uniques:', c.dropna().unique()[:6], '| notna:', int(c.notna().sum()))
# after map
v = tdi['CYP2D6_is_TDI']
m = v.map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0})
print('mapped notna:', int(m.notna().sum()))
