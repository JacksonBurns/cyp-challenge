import pandas as pd

DATA = "/home/jackson/cyp-challenge/data/"
emx = pd.read_csv(DATA + "cyp-challenge-TRAIN_Emax.csv")
cand = [c for c in emx.columns if "pIC50" in c]
print("pIC50 cols:", cand[:12])
for pat in ("chem", "pub", "aux", "ext"):
    print(pat, [c for c in emx.columns if pat in c.lower()][:8])
import os
print(sorted(os.listdir(DATA))[:20])
