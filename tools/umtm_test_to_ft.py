"""Convert UMTM test-pred OOF-style files to ft_test_preds_<tag>.csv schema.

  python tools/umtm_test_to_ft.py cache/umtm_test_L2.csv   -> cache/ft_test_preds_umtmL2.csv
  python tools/umtm_test_to_ft.py cache/umtm_test_L2_s1.csv -> cache/ft_test_preds_umtmL2_s1.csv

Row order must already equal run_regression's test-frame SMILES order (asserted).
"""
import os
import re
import sys

import pandas as pd

ROOT = os.path.expanduser("~/cyp-challenge")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

test = pd.read_csv(os.path.join(ROOT, "data", "cyp-challenge-TEST-BLINDED.csv"))
test_smi = test["SMILES"].tolist()

for path in sys.argv[1:]:
    df = pd.read_csv(path)
    m = re.search(r"umtm_test_(L\d)(?:_s(\d+))?\.csv", os.path.basename(path))
    assert m, f"unrecognized filename {path}"
    tag = f"umtm{m.group(1)}" + (f"_s{m.group(2)}" if m.group(2) else "")
    out = pd.DataFrame({"SMILES": df["SMILES"].values})
    for iso in ISO:
        out[iso] = df[f"{iso}_direct"].values
    assert out["SMILES"].tolist() == test_smi, f"{path}: row order != TEST-BLINDED"
    dst = os.path.join(ROOT, "cache", f"ft_test_preds_{tag}.csv")
    out.to_csv(dst, index=False)
    print("wrote", dst, out.shape)
