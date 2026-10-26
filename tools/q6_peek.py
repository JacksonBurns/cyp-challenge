import pandas as pd
d = pd.read_csv('/home/jackson/cyp-challenge/external/chembl_cyp_homolog_ic50.csv')
print(d.shape, d.columns.tolist())
if 'isoform' in d.columns:
    print(d.isoform.value_counts().to_dict())
