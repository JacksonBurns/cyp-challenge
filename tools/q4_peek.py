import pandas as pd
d = pd.read_csv('/home/jackson/cyp-challenge/external/chembl_cyp_ic50.csv')
print(d.shape, d.columns.tolist())
print(d.head(3).to_string())
# assay-level replicate variance per isoenzyme (Q4 noise floor)
import numpy as np
key = [c for c in d.columns if 'smiles' in c.lower() or 'Smiles' in c]
print('smiles cols:', key)
ac = [c for c in d.columns if 'assay' in c.lower()]
print('assay cols:', ac)
