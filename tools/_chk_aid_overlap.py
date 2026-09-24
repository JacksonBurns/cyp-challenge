import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
pc = pd.read_csv('external/pubchem_cyp_qhts_aid1851.csv')
pc['canon'] = pc['smiles'].map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if s and Chem.MolFromSmiles(s) else None)
Xtr = pd.read_parquet('cache/X_train.parquet').drop_duplicates('SMILES')
Xte = pd.read_parquet('cache/X_test.parquet')

def canon_set(smi):
    out = set()
    for s in smi:
        m = Chem.MolFromSmiles(s)
        if m:
            out.add(Chem.MolToSmiles(m))
    return out

tr_s = canon_set(Xtr['SMILES'])
te_s = canon_set(Xte['SMILES'])
cn = pc['canon'].dropna()
print('AID canon unique:', cn.nunique(), 'in train:', cn.isin(tr_s).sum(), 'in TEST:', cn.isin(te_s).sum())
