"""Export SMILES + direct-inhibition targets + scaffold folds for the CheMeLeon
fine-tune script (which runs in the chemeleon env without our data plumbing).

Writes cache/ft_data.csv: SMILES, CYP1A2, CYP2C9, CYP2D6, CYP3A4, fold
(rows = X_train.parquet dedup order, identical to run_regression.build_all).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
folds = R.make_folds(R.scaffold_groups(Xtr["SMILES"].tolist()))
df = pd.DataFrame({"SMILES": Xtr["SMILES"].values})
for iso in R.ISOFORMS:
    df[iso] = targets[R.DIRECT[iso]].values
df["fold"] = folds
df.to_csv(os.path.join(CACHE, "ft_data.csv"), index=False)
print(df.shape, "labeled per iso:", {i: int(df[i].notna().sum()) for i in R.ISOFORMS})
Xte[["SMILES"]].to_csv(os.path.join(CACHE, "ft_test_smiles.csv"), index=False)
