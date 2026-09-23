import pandas as pd
ch = pd.read_csv("external/chembl_cyp_ic50.csv")
print("chembl isoforms:", ch["isoform"].unique())
print("units:", ch["standard_units"].unique()[:8])
ft = pd.read_csv("cache/ft_data.csv")
print("ft rows:", len(ft), "folds:", ft["fold"].nunique())
