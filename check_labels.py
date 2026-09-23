import numpy as np
import pandas as pd

iso = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
tr = pd.read_csv("data/cyp-challenge-TRAIN_inhibition.csv")
tdi = pd.read_csv("data/cyp-challenge-TRAIN_TDI.csv")

m = tr.merge(tdi, on=["Molecule_Name", "SMILES"], suffixes=("_tr", "_tdi"))
print("merged overlap:", len(m))
for c in iso:
    col = f"{c}_pIC50_direct_inhibition"
    a, b = m[f"{col}_tr"], m[f"{col}_tdi"]
    both = a.notna() & b.notna()
    diff = (a[both] - b[both]).abs().max()
    print(f"{col}: train n={a.notna().sum()} tdi-file n={b.notna().sum()} both={both.sum()} maxdiff={diff:.6f}")
    extra = b.notna() & a.isna()
    print(f"   extra labels in TDI file: {extra.sum()}, mean {b[extra].mean() if extra.sum() else float('nan'):.3f}")

# is_TDI label rates
for c in ["CYP2D6", "CYP3A4"]:
    col = f"{c}_is_TDI"
    print(col, tdi[col].value_counts(dropna=False).to_dict())

# check consistency of extra rows: do extra compounds have direct+TDI both?
extra_cpds = tdi[~tdi.Molecule_Name.isin(tr.Molecule_Name)]
print("\nextra compounds:", len(extra_cpds))
for c in iso:
    print(f"  direct {c}: {extra_cpds[f'{c}_pIC50_direct_inhibition'].notna().sum()}")
for c in ["CYP2D6", "CYP3A4"]:
    print(f"  is_TDI {c}: {extra_cpds[f'{c}_is_TDI'].notna().sum()}")
