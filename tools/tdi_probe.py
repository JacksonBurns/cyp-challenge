import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

def ck(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None

t = pd.read_csv("data/cyp-challenge-TRAIN_TDI.csv")
inh = pd.read_csv("data/cyp-challenge-TRAIN_inhibition.csv")
t["ck"] = t["SMILES"].map(ck)
inh["ck"] = inh["SMILES"].map(ck)
print("TDI parse fails:", t["ck"].isna().sum(), "inh parse fails:", inh["ck"].isna().sum())
tset = set(t["ck"].dropna())
iset = set(inh["ck"].dropna())
print("TDI unique:", len(tset), "inh unique:", len(iset), "overlap:", len(tset & iset))
# raw string overlap for comparison
print("raw overlap:", len(set(t["SMILES"]) & set(inh["SMILES"])))
# label availability on the overlap rows
ov = t[t["ck"].isin(iset)]
print("overlap rows:", len(ov), "unique:", ov["ck"].nunique())
for c in ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]:
    print(c, ov[c].value_counts(dropna=False).to_dict())
