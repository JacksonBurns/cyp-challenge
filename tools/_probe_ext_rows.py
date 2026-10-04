import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')
from umtm_data import HEADS
m = pd.read_csv('cache/umtm_master.csv')
for k in ['ext', 'sc_only']:
    sub = m[m['kind'] == k]
    print(k, len(sub), {c: int(sub[c].notna().sum()) for c in HEADS if sub[c].notna().sum() > 0})
# positive counts for is_TDI
ch = m[m['kind'] == 'challenge']
print('is_TDI_2D6 pos:', float(ch['is_TDI_2D6'].sum()), '| 3A4 pos:', float(ch['is_TDI_3A4'].sum()))
