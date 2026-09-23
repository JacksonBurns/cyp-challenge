"""Full re-verification of both submission files with the official validators."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from validation.activity_validation import validate_activity_submission
from validation.tdi_validation import validate_tdi_submission

test = pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv")

print("== REGRESSION: cache/raw_regression_submission.csv")
ok, errs = validate_activity_submission("cache/raw_regression_submission.csv",
                                        expected_ids=set(test["Molecule_Name"].astype(str)))
print("  official validator:", ok, errs)
reg = pd.read_csv("cache/raw_regression_submission.csv")
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
print("== TDI: cache/tdi_submission.csv")
ok, errs = validate_tdi_submission("cache/tdi_submission.csv",
                                   expected_ids=set(test["Molecule_Name"].astype(str)))
print("  official validator:", ok, errs)
tdi = pd.read_csv("cache/tdi_submission.csv")
print("  rows:", len(tdi), "| cols:", list(tdi.columns))
print("  row-for-row SMILES match:", bool((tdi["SMILES"].values == test["SMILES"].values).all()))
print("  row-for-row Molecule_Name match:", bool(
    (tdi["Molecule_Name"].values == test["Molecule_Name"].values).all()))
for c in ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]:
    print(f"  {c}: dtype={tdi[c].dtype} pos={int(tdi[c].sum())}/750 rate={tdi[c].mean():.4f}")
print("  NaN anywhere:", int(tdi.isna().sum().sum()))
