import pandas as pd
em = pd.read_csv("data/cyp-challenge-TRAIN_Emax.csv")
print(len(em), em["SMILES"].nunique())
print(em.columns.tolist())
for c in em.columns:
    if em[c].dtype == bool:
        print(c, "pos", int(em[c].sum()), "/", int(em[c].notna().sum()))
