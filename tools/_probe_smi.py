import os, sys
import pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')
Xtr = pd.read_parquet('cache/X_train.parquet')
inh = pd.read_csv('data/cyp-challenge-TRAIN_inhibition.csv')
tdi = pd.read_csv('data/cyp-challenge-TRAIN_TDI.csv')
print('X vs inh raw overlap:', len(set(Xtr.SMILES) & set(inh.SMILES)))
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s
print('X vs inh canon overlap:', len(set(Xtr.SMILES) & set(inh.SMILES.map(canon))))
import run_regression as R
print('run_regression SMILES handling:')
import inspect
src = inspect.getsource(R)
import re
for ln in src.splitlines():
    if 'SMILES' in ln and ('canon' in ln.lower() or 'MolToSmiles' in ln or 'drop_dup' in ln):
        print('  ', ln.strip())
