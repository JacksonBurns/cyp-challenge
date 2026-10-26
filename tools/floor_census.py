"""Floor census: how many labeled rows sit at the assay reporting floor,
per isoform, per label family (direct_inhibition / Emax / chembl / pubchem).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402

DATA = H.R.DATA

for iso in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
    col = f"{iso}_pIC50_direct_inhibition"
    y = H.targets[col].values.astype(float)
    lab = y[~np.isnan(y)]
    print(f"DI {iso}: n={len(lab)} min={lab.min():.2f} "
          f"p1={np.percentile(lab,1):.2f} "
          f"frac<=1.5={(lab<=1.5).mean():.4f} "
          f"frac<=2={(lab<=2).mean():.3f}")

# 1A2 label columns available
print("targets cols w/ CYP1A2:", [c for c in H.targets.columns if "CYP1A2" in c])

emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
for c in emx.columns:
    if "CYP1A2" in c and emx[c].dtype != object:
        s = emx[c].values.astype(float)
        s = s[~np.isnan(s)]
        if len(s):
            print(f"EMX {c}: n={len(s)} min={s.min():.2f} "
                  f"frac<=1.5={(s<=1.5).mean():.4f}")
