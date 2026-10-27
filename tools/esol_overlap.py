import sys
import pandas as pd
sys.path.insert(0, "/home/jackson/cyp-challenge/tools")
from overlap_check import keys

es = pd.read_csv("/home/jackson/cyp-challenge/external/user_supplied/delaney-processed.csv")
print("ESOL cols:", [c for c in es.columns if "SMILES" in c.upper() or "solub" in c.lower()])
smi_col = [c for c in es.columns if c.upper().startswith("SMILES")][0]
rows = [keys(s) for s in es[smi_col].dropna().unique()]
d = pd.DataFrame(rows, columns=["k_iso", "k_st", "k_ik"]).dropna(subset=["k_iso"])
tr = pd.read_csv("/home/jackson/cyp-challenge/data/cyp-challenge-TRAIN_Emax.csv")
te = pd.read_csv("/home/jackson/cyp-challenge/data/cyp-challenge-TEST-BLINDED.csv")
allch = pd.concat([tr["SMILES"], te["SMILES"]])
ck = pd.DataFrame([keys(s) for s in allch.dropna().unique()],
                  columns=["k_iso", "k_st", "k_ik"]).dropna(subset=["k_iso"])
for k in ["k_iso", "k_st"]:
    ov = set(d[k].dropna()) & set(ck[k].dropna())
    print("ESOL", k, "n=", len(d), "overlap=", len(ov))
