import pandas as pd

df = pd.read_csv("external/chembl_cyp_ic50.csv")
print(df.columns.tolist())
print(df["assay_type"].value_counts().head(6))
print(df["assay_description"].dropna().sample(8, random_state=0).to_string())
h = pd.read_csv("external/chembl_cyp_homolog_ic50.csv")
print("homolog cols:", h.columns.tolist())
print(h["assay_description"].dropna().sample(6, random_state=0).to_string())
