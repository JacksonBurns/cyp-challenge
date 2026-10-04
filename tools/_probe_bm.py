import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')
from umtm_data import build_master, HEADS
m, test = build_master()
ch = m[m.kind == 'challenge']
print('dup cols in master:', m.columns.duplicated().sum())
print('is_TDI_2D6 notna (challenge):', int(ch['is_TDI_2D6'].notna().sum()))
print('is_TDI_2D6 dtype:', m['is_TDI_2D6'].dtype)
# where do challenge rows' is_TDI live across the whole frame?
print('total notna is_TDI_2D6 all kinds:', int(m['is_TDI_2D6'].notna().sum()))
