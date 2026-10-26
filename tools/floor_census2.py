"""Final checks: TDI_condition label floor pileup, single-concentration
structure, DI vs external-ChEMBL label disagreement on overlap."""
import os

import numpy as np
import pandas as pd

DATA = "/home/jackson/cyp-challenge/data/"
EXT = "/home/jackson/cyp-challenge/external/"

inh = pd.read_csv(DATA + "cyp-challenge-TRAIN_inhibition.csv")
cols = [c for c in inh.columns if "pIC50" in c and "TDI_condition" in c
        and "conf" not in c]
print("TDI_condition pIC50 cols:", cols)
for c in cols:
    s = inh[c].values.astype(float)
    s = s[~np.isnan(s)]
    print(f" {c}: n={len(s)} min={s.min():.2f} "
          f"frac<=1.1={(s<=1.1).mean():.3f} frac<=2={(s<=2).mean():.3f} "
          f"mode~{pd.Series(s).mode().iloc[0]:.2f}" if len(s) else f" {c}: empty")

sc = pd.read_csv(DATA + "cyp-challenge-single-concentration-TRAIN.csv")
print("single-conc cols:", list(sc.columns)[:12], sc.shape)
for c in sc.columns:
    if sc[c].dtype != object:
        s = sc[c].values.astype(float)
        s = s[~np.isnan(s)]
        if len(s):
            print(f"  {c}: n={len(s)} min={s.min():.2f} max={s.max():.2f} "
                  f"frac<=1.1={(s<=1.1).mean():.2f}")
