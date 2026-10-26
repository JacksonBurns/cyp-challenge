import pandas as pd
import numpy as np

DATA = "/home/jackson/cyp-challenge/data/"
emx = pd.read_csv(DATA + "cyp-challenge-TRAIN_Emax.csv").drop_duplicates("SMILES")
tdi = pd.read_csv(DATA + "cyp-challenge-TRAIN_TDI.csv").drop_duplicates("SMILES")
inh = pd.read_csv(DATA + "cyp-challenge-TRAIN_inhibition.csv").drop_duplicates("SMILES").set_index("SMILES")

for src, name in [(emx, "Emax"), (tdi, "TDIfile")]:
    for iso in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
        c = f"{iso}_is_TDI"
        p = f"{iso}_pIC50_direct_inhibition"
        if c in src.columns and p in inh.columns:
            m = src.set_index("SMILES")[[c]].join(inh[[p]], how="inner").dropna()
            g = m.groupby(c)[p].agg(["mean", "count"])
            d = {int(k): (round(float(v["mean"]), 2), int(v["count"]))
                 for k, v in g.iterrows()}
            print(name, iso, d)
