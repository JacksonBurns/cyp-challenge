import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/jackson/cyp-challenge/src')
import tdi_fraction_opt_v7 as F

tdi = pd.read_csv('/home/jackson/cyp-challenge/data/cyp-challenge-TRAIN_TDI.csv')
emx = pd.read_csv('/home/jackson/cyp-challenge/data/cyp-challenge-TRAIN_Emax.csv')
X = pd.concat([tdi, emx], ignore_index=True)
is_t = X["SMILES"].isin(set(tdi["SMILES"]))
y2 = F.label(tdi)
y2d6 = y2.loc[y2["CYP2D6"] >= 0, "CYP2D6"]
print("X rows", len(X), "is_t", int(is_t.sum()), "y2d6 labeled", len(y2d6))
c = np.load('/home/jackson/cyp-challenge/cache/tdi_fb4_v6b_merged_CYP2D6cap.npy')
print("cap", c.shape, "mean", round(float(c.mean()), 4))
print("y2d6 index min/max", y2d6.index.min(), y2d6.index.max())
