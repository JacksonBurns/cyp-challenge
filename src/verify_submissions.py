"""Full re-verification of submission files with the official validators.

Default: the two interim files. Pass paths to check candidates:
  python src/verify_submissions.py [regression.csv] [tdi.csv]
Each file must pass its official validator, have 750 rows, exact columns, and
match data/cyp-challenge-TEST-BLINDED.csv row-for-row in SMILES + Molecule_Name.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from validation.activity_validation import validate_activity_submission
from validation.tdi_validation import validate_tdi_submission

test = pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv")

REG = sys.argv[1] if len(sys.argv) > 1 else "cache/raw_regression_submission.csv"
TDI = sys.argv[2] if len(sys.argv) > 2 else "cache/tdi_submission.csv"

print(f"== REGRESSION: {REG}")
ok, errs = validate_activity_submission(REG, expected_ids=set(test["Molecule_Name"].astype(str)))
print("  official validator:", ok, errs)
reg = pd.read_csv(REG)
print("  rows:", len(reg), "| cols:", list(reg.columns))
print("  row-for-row SMILES match:", bool((reg["SMILES"].values == test["SMILES"].values).all()))
print("  row-for-row Molecule_Name match:", bool(
    (reg["Molecule_Name"].values == test["Molecule_Name"].values).all()))
pred_cols = [c for c in reg.columns if c.endswith("_pIC50_direct_inhibition")]
print("  pred col stats:")
for c in pred_cols:
    print(f"    {c}: mean={reg[c].mean():.3f} std={reg[c].std():.4f} "
          f"min={reg[c].min():.2f} max={reg[c].max():.2f} NaN={int(reg[c].isna().sum())}")

print()
print(f"== TDI: {TDI}")
ok, errs = validate_tdi_submission(TDI, expected_ids=set(test["Molecule_Name"].astype(str)))
print("  official validator:", ok, errs)
tdi = pd.read_csv(TDI)
print("  rows:", len(tdi), "| cols:", list(tdi.columns))
print("  row-for-row SMILES match:", bool((tdi["SMILES"].values == test["SMILES"].values).all()))
print("  row-for-row Molecule_Name match:", bool(
    (tdi["Molecule_Name"].values == test["Molecule_Name"].values).all()))
for c in ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]:
    print(f"  {c}: dtype={tdi[c].dtype} pos={int(tdi[c].sum())}/750 rate={tdi[c].mean():.4f}")
print("  NaN anywhere:", int(tdi.isna().sum().sum()))
